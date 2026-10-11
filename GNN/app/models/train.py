from __future__ import annotations

import argparse
import csv
import json
import logging
from pathlib import Path

from app.config import load_config, ensure_directories
from app.graph.features import FeatureSchema, fit_edge_normalization
from app.models.checkpoints import save_checkpoint
from app.models.dataset import dataset_manifest, graph_to_tensors, read_jsonl, split_examples, validate_label_integrity
from app.models.gnn import GNNDimensions, build_model

logger = logging.getLogger(__name__)


def train(config_path: str | None = None) -> dict[str, object]:
    config = load_config(config_path)
    ensure_directories(config)
    try:
        import torch
        import torch.nn.functional as F
    except Exception as exc:
        raise RuntimeError("Training requires torch and torch_geometric. Install requirements.txt.") from exc

    examples = read_jsonl(config.dataset_path)
    counts = validate_label_integrity(examples)
    train_examples, val_examples, test_examples = split_examples(examples, seed=config.model.batch_seed)
    train_edges = [edge for example in train_examples for edge in example.supervised_edges()]
    if not train_edges:
        raise ValueError("No supervised connected/not_connected labels available for training.")
    schema = fit_edge_normalization(train_edges, FeatureSchema())
    train_data = [graph_to_tensors(example, schema) for example in train_examples]
    val_data = [graph_to_tensors(example, schema) for example in val_examples] or train_data

    dims = GNNDimensions(
        node_features=train_data[0].x.shape[1],
        edge_features=train_data[0].edge_attr.shape[1],
        hidden_dim=config.model.hidden_dim,
        dropout=config.model.dropout,
    )
    torch.manual_seed(config.model.batch_seed)
    model = build_model(dims)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.model.learning_rate)
    best_val = float("inf")
    best_state = None
    stale = 0
    epoch_rows: list[dict[str, float | int]] = []

    for epoch in range(1, config.model.epochs + 1):
        model.train()
        total_loss = 0.0
        total_edges = 0
        for data in train_data:
            optimizer.zero_grad()
            logits = model(data.x, data.edge_index, data.edge_index, data.edge_attr)
            mask = data.supervised_mask
            loss = F.binary_cross_entropy_with_logits(logits[mask], data.y[mask])
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * int(mask.sum())
            total_edges += int(mask.sum())
        train_loss = total_loss / max(total_edges, 1)
        model.eval()
        with torch.no_grad():
            val_losses = []
            for data in val_data:
                logits = model(data.x, data.edge_index, data.edge_index, data.edge_attr)
                mask = data.supervised_mask
                val_losses.append(float(F.binary_cross_entropy_with_logits(logits[mask], data.y[mask]).item()))
            val_loss = sum(val_losses) / len(val_losses)
        epoch_rows.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        if val_loss < best_val:
            best_val = val_loss
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if stale >= config.model.patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    metadata = {
        "model_version": "gnn-v1",
        "hidden_dim": config.model.hidden_dim,
        "dropout": config.model.dropout,
        "threshold": config.model.threshold,
        "dataset_manifest": dataset_manifest(examples),
        "label_counts": counts,
        "splits": {"train": len(train_examples), "validation": len(val_examples), "test": len(test_examples)},
        "best_val_loss": best_val,
    }
    save_checkpoint(config.checkpoint_path, model, metadata, schema)
    metrics_path = config.checkpoint_path.parent / "training_metrics.csv"
    with metrics_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["epoch", "train_loss", "val_loss"])
        writer.writeheader()
        writer.writerows(epoch_rows)
    summary = {"checkpoint": str(config.checkpoint_path), "metrics": str(metrics_path), "metadata": metadata}
    (config.checkpoint_path.parent / "training_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train.yaml")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    result = train(args.config)
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
