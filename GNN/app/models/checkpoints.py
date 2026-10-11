from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from app.graph.features import FeatureSchema


def save_checkpoint(path: str | Path, model: Any, metadata: dict[str, Any], feature_schema: FeatureSchema) -> None:
    try:
        import torch
    except Exception as exc:
        raise RuntimeError("Saving GNN checkpoints requires torch.") from exc
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": model.state_dict(), "metadata": metadata, "feature_schema": feature_schema.model_dump()}, out)
    out.with_suffix(".metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    out.with_suffix(".features.json").write_text(feature_schema.model_dump_json(indent=2), encoding="utf-8")


def load_checkpoint(path: str | Path) -> dict[str, Any]:
    try:
        import torch
    except Exception as exc:
        raise RuntimeError("Loading GNN checkpoints requires torch.") from exc
    src = Path(path)
    if not src.exists():
        raise FileNotFoundError(f"Checkpoint not found: {src}")
    return torch.load(src, map_location="cpu")
