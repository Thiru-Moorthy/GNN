from __future__ import annotations

from app.graph.features import FeatureSchema
from app.models.dataset import GraphExample, graph_to_tensors
from app.models.gnn import GNNDimensions, build_model
from app.schemas.prediction import EdgePrediction


def fallback_predictions(example: GraphExample, model_version: str = "gnn-unavailable") -> list[EdgePrediction]:
    predictions: list[EdgePrediction] = []
    for edge in example.edges:
        score = edge.baseline_score if edge.baseline_score is not None else 0.0
        predictions.append(
            EdgePrediction(
                edge_id=edge.edge_id,
                model_name="baseline_fallback",
                model_version=model_version,
                probability=float(score),
                predicted_class="connected" if score >= 0.5 else "not_connected",
                decision_status="fallback",
            )
        )
    return predictions


def predict_with_checkpoint(example: GraphExample, checkpoint_path: str) -> list[EdgePrediction]:
    try:
        import torch
    except Exception as exc:
        raise RuntimeError("GNN inference requires torch.") from exc
    from app.models.checkpoints import load_checkpoint

    checkpoint = load_checkpoint(checkpoint_path)
    schema = FeatureSchema.model_validate(checkpoint["feature_schema"])
    data = graph_to_tensors(example, schema)
    dims = GNNDimensions(node_features=data.x.shape[1], edge_features=data.edge_attr.shape[1], hidden_dim=checkpoint["metadata"]["hidden_dim"])
    model = build_model(dims)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    with torch.no_grad():
        logits = model(data.x, data.edge_index, data.edge_index, data.edge_attr)
        probs = torch.sigmoid(logits).cpu().tolist()
    seen: set[str] = set()
    predictions: list[EdgePrediction] = []
    for edge_id, prob in zip(data.edge_ids, probs, strict=True):
        if edge_id in seen:
            continue
        seen.add(edge_id)
        predictions.append(
            EdgePrediction(
                edge_id=edge_id,
                model_name="GraphSAGEEdgeClassifier",
                model_version=str(checkpoint["metadata"].get("model_version", "gnn-v1")),
                probability=float(prob),
                predicted_class="connected" if prob >= checkpoint["metadata"].get("threshold", 0.5) else "not_connected",
            )
        )
    return predictions
