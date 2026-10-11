from __future__ import annotations

from app.schemas.edge import CandidateEdge


def incompatible_cross_page_without_connector(edge: CandidateEdge) -> bool:
    return not bool(edge.geometric_features.get("same_page", True))


def rule_explanations(edge: CandidateEdge) -> list[str]:
    reasons: list[str] = []
    if incompatible_cross_page_without_connector(edge):
        reasons.append("cross_page_candidate_without_off_page_connector_evidence")
    if not bool(edge.geometric_features.get("line_evidence", False)):
        reasons.append("missing_line_evidence")
    return reasons
