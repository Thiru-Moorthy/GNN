from __future__ import annotations

from app.config import ValidationConfig
from app.schemas.edge import CandidateEdge, RelationshipType, ValidatedRelationship
from app.schemas.prediction import EdgePrediction


def validate_edge(
    edge: CandidateEdge,
    prediction: EdgePrediction | None,
    config: ValidationConfig | None = None,
) -> ValidatedRelationship:
    cfg = config or ValidationConfig()
    probability = prediction.probability if prediction else None
    baseline = edge.baseline_score if edge.baseline_score is not None else 0.0
    reasons: list[str] = []
    reviewer_status = "not_required"

    if probability is None:
        status = "REVIEW"
        reviewer_status = "pending"
        reasons.append("gnn_prediction_unavailable")
    elif probability >= cfg.accept_probability and baseline >= cfg.min_baseline_for_accept:
        status = "ACCEPT"
        reasons.append(f"probability={probability:.3f}_baseline={baseline:.3f}")
    elif probability < cfg.review_probability and baseline < cfg.min_baseline_for_accept:
        status = "REJECT"
        reasons.append(f"low_probability={probability:.3f}_baseline={baseline:.3f}")
    else:
        status = "REVIEW"
        reviewer_status = "pending"
        reasons.append(f"ambiguous_probability={probability:.3f}_baseline={baseline:.3f}")

    if not bool(edge.geometric_features.get("same_page", True)):
        status = "REVIEW"
        reviewer_status = "pending"
        reasons.append("cross_page_requires_explicit_connector_evidence")

    return ValidatedRelationship(
        relationship_id=f"rel-{edge.edge_id}",
        source_node_id=edge.source_node_id,
        target_node_id=edge.target_node_id,
        relation_type=RelationshipType.physical_pipe,
        validation_status=status,  # type: ignore[arg-type]
        validation_reasons=reasons,
        evidence_references=edge.source_evidence,
        model_version=prediction.model_version if prediction else None,
        reviewer_status=reviewer_status,  # type: ignore[arg-type]
        attributes={"baseline_score": baseline, "gnn_probability": probability},
    )
