from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field


class LabelStatus(str, Enum):
    connected = "connected"
    not_connected = "not_connected"
    ambiguous = "ambiguous"
    unlabeled = "unlabeled"


class RelationshipType(str, Enum):
    physical_pipe = "physical_pipe"
    signal = "signal"
    candidate = "candidate"
    unknown = "unknown"


class CandidateEdge(BaseModel):
    edge_id: str
    source_node_id: str
    target_node_id: str
    candidate_relation_type: RelationshipType = RelationshipType.physical_pipe
    geometric_features: dict[str, float | int | bool | str] = Field(default_factory=dict)
    baseline_score: float | None = None
    baseline_decision: str | None = None
    baseline_explanation: list[str] = Field(default_factory=list)
    ground_truth_label: LabelStatus = LabelStatus.unlabeled
    label_status: LabelStatus = LabelStatus.unlabeled
    source_evidence: list[str] = Field(default_factory=list)

    @property
    def is_supervised_label(self) -> bool:
        return self.label_status in {LabelStatus.connected, LabelStatus.not_connected}


class ValidatedRelationship(BaseModel):
    relationship_id: str
    source_node_id: str
    target_node_id: str
    relation_type: RelationshipType = RelationshipType.physical_pipe
    validation_status: Literal["ACCEPT", "REVIEW", "REJECT"]
    validation_reasons: list[str] = Field(default_factory=list)
    evidence_references: list[str] = Field(default_factory=list)
    model_version: str | None = None
    rule_version: str = "rules-v1"
    reviewer_status: Literal["not_required", "pending", "approved", "rejected"] = "not_required"
    attributes: dict[str, Any] = Field(default_factory=dict)
