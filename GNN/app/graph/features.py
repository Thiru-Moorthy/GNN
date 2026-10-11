from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np
from pydantic import BaseModel, Field

from app.schemas.edge import CandidateEdge
from app.schemas.node import GraphNode, NodeType


NODE_TYPES = [item.value for item in NodeType]
EDGE_FEATURE_NAMES = [
    "distance_norm",
    "dx_norm",
    "dy_norm",
    "bbox_gap_norm",
    "alignment_score",
    "same_page",
    "line_evidence",
    "junction_evidence",
    "source_confidence",
    "target_confidence",
]


class FeatureSchema(BaseModel):
    node_type_mapping: dict[str, int] = Field(default_factory=lambda: {name: idx for idx, name in enumerate(NODE_TYPES)})
    edge_feature_names: list[str] = Field(default_factory=lambda: list(EDGE_FEATURE_NAMES))
    node_feature_names: list[str] = Field(
        default_factory=lambda: ["node_type_id", "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2", "cx", "cy", "w", "h", "confidence"]
    )
    means: dict[str, float] = Field(default_factory=dict)
    stds: dict[str, float] = Field(default_factory=dict)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(self.model_dump_json(indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "FeatureSchema":
        return cls.model_validate_json(Path(path).read_text(encoding="utf-8"))


def node_feature_vector(node: GraphNode, page_size: tuple[float, float], schema: FeatureSchema | None = None) -> list[float]:
    feature_schema = schema or FeatureSchema()
    width, height = page_size
    x1, y1, x2, y2 = node.bbox
    cx, cy = node.center
    return [
        float(feature_schema.node_type_mapping.get(node.node_type.value, feature_schema.node_type_mapping["unknown"])),
        x1 / width,
        y1 / height,
        x2 / width,
        y2 / height,
        cx / width,
        cy / height,
        (x2 - x1) / width,
        (y2 - y1) / height,
        float(node.detection_confidence),
    ]


def edge_feature_vector(edge: CandidateEdge, schema: FeatureSchema | None = None) -> list[float]:
    feature_schema = schema or FeatureSchema()
    values: list[float] = []
    for name in feature_schema.edge_feature_names:
        value = edge.geometric_features.get(name, 0.0)
        values.append(float(value) if not isinstance(value, bool) else float(value))
    return values


def fit_edge_normalization(edges: Iterable[CandidateEdge], schema: FeatureSchema | None = None) -> FeatureSchema:
    fitted = schema or FeatureSchema()
    rows = np.array([edge_feature_vector(edge, fitted) for edge in edges], dtype=float)
    if rows.size == 0:
        return fitted
    for idx, name in enumerate(fitted.edge_feature_names):
        fitted.means[name] = float(rows[:, idx].mean())
        std = float(rows[:, idx].std())
        fitted.stds[name] = std if std > 1e-8 else 1.0
    return fitted


def normalized_edge_features(edge: CandidateEdge, schema: FeatureSchema) -> list[float]:
    raw = edge_feature_vector(edge, schema)
    return [
        (value - schema.means.get(name, 0.0)) / schema.stds.get(name, 1.0)
        for value, name in zip(raw, schema.edge_feature_names, strict=True)
    ]


def write_feature_manifest(schema: FeatureSchema, path: str | Path) -> None:
    Path(path).write_text(json.dumps(schema.model_dump(), indent=2), encoding="utf-8")
