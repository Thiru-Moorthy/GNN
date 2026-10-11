from __future__ import annotations

import argparse
import json
from pathlib import Path

from app.baseline.geometry import score_edges
from app.config import load_config, ensure_directories
from app.evaluation.metrics import binary_metrics, labeled_edge_arrays
from app.evaluation.reports import write_markdown_report
from app.models.dataset import read_jsonl, validate_label_integrity
from app.models.predict import fallback_predictions, predict_with_checkpoint
from app.schemas.prediction import EdgePrediction
from app.validation.topology import validate_predictions


def benchmark(config_path: str | None = None) -> dict[str, object]:
    config = load_config(config_path)
    ensure_directories(config)
    examples = read_jsonl(config.dataset_path)
    label_counts = validate_label_integrity(examples)
    all_edges = []
    baseline_predictions: list[EdgePrediction] = []
    gnn_predictions: list[EdgePrediction] = []
    validation_predictions: list[EdgePrediction] = []

    checkpoint_available = Path(config.checkpoint_path).exists()
    for example in examples:
        scored_edges = score_edges(example.edges, config.baseline)
        example = example.model_copy(update={"edges": scored_edges})
        all_edges.extend(scored_edges)
        baseline_predictions.extend(fallback_predictions(example, "geometry-baseline-v1"))
        if checkpoint_available:
            try:
                gnn_predictions.extend(predict_with_checkpoint(example, str(config.checkpoint_path)))
            except RuntimeError:
                gnn_predictions.extend(fallback_predictions(example, "gnn-unavailable"))
        else:
            gnn_predictions.extend(fallback_predictions(example, "gnn-checkpoint-missing"))
        relationships = validate_predictions(scored_edges, gnn_predictions[-len(scored_edges) :])
        for rel in relationships:
            probability = float(rel.attributes.get("gnn_probability") or 0.0)
            if rel.validation_status == "ACCEPT":
                probability = max(probability, config.model.threshold)
            elif rel.validation_status == "REJECT":
                probability = min(probability, config.model.threshold - 0.01)
            validation_predictions.append(
                EdgePrediction(
                    edge_id=rel.relationship_id.replace("rel-", "", 1),
                    model_name="gnn_plus_validation",
                    model_version=rel.model_version or "validation-v1",
                    probability=probability,
                    predicted_class="connected" if rel.validation_status == "ACCEPT" else "not_connected",
                    decision_status="predicted",
                )
            )

    methods: dict[str, object] = {}
    for name, predictions in {
        "geometry_baseline": baseline_predictions,
        "gnn": gnn_predictions,
        "gnn_plus_validation": validation_predictions,
    }.items():
        y_true, y_score = labeled_edge_arrays(all_edges, predictions)
        methods[name] = binary_metrics(y_true, y_score, config.model.threshold) if y_true else {"status": "no_labeled_edges"}

    results: dict[str, object] = {
        "is_synthetic": all(example.is_synthetic for example in examples),
        "methods": methods,
        "counts": {"documents": len(examples), "candidate_edges": len(all_edges), **label_counts},
        "checkpoint_used": checkpoint_available,
        "limitations": [
            "Synthetic results do not represent real P&ID performance.",
            "Real benchmarking requires verified annotated P&ID candidate edges.",
        ],
    }
    write_markdown_report(config.report_path, results)
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/evaluation.yaml")
    args = parser.parse_args()
    print(json.dumps(benchmark(args.config), indent=2))


if __name__ == "__main__":
    main()
