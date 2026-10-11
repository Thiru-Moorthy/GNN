from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator


class NodeType(str, Enum):
    equipment = "equipment"
    valve = "valve"
    instrument = "instrument"
    pipe_segment = "pipe_segment"
    junction = "junction"
    text = "text"
    off_page_connector = "off_page_connector"
    unknown = "unknown"


class GraphNode(BaseModel):
    node_id: str
    page_id: str
    node_type: NodeType
    class_id: str
    tag: str | None = None
    bbox: tuple[float, float, float, float]
    center: tuple[float, float]
    orientation: float | None = None
    detection_confidence: float = 1.0
    source_evidence: list[str] = Field(default_factory=list)
    attributes: dict[str, Any] = Field(default_factory=dict)
    annotation_status: Literal["observed", "synthetic", "manual", "unknown"] = "observed"

    @model_validator(mode="after")
    def validate_bbox(self) -> "GraphNode":
        x1, y1, x2, y2 = self.bbox
        if x2 < x1 or y2 < y1:
            raise ValueError(f"Invalid bbox for node {self.node_id}: {self.bbox}")
        return self
