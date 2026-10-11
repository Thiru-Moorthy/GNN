from __future__ import annotations

from app.config import BaselineConfig
from app.schemas.edge import CandidateEdge


def score_edge(edge: CandidateEdge, config: BaselineConfig | None = None) -> CandidateEdge:
    cfg = config or BaselineConfig()
    gf = edge.geometric_features
    distance_px = float(gf.get("distance_px", 9999.0))
    gap_px = float(gf.get("bbox_gap_px", distance_px))
    alignment = float(gf.get("alignment_score", 0.0))
    line_evidence = bool(gf.get("line_evidence", False))
    distance_score = max(0.0, 1.0 - distance_px / cfg.max_distance_px)
    gap_score = max(0.0, 1.0 - gap_px / cfg.max_distance_px)
    score = 0.45 * distance_score + 0.25 * gap_score + 0.2 * alignment + (0.1 if line_evidence else 0.0)
    reasons = [
        f"distance_px={distance_px:.1f}",
        f"bbox_gap_px={gap_px:.1f}",
        f"alignment_score={alignment:.2f}",
        f"line_evidence={line_evidence}",
    ]
    decision = "connected" if score >= cfg.accept_threshold else "review" if score >= cfg.review_threshold else "not_connected"
    return edge.model_copy(update={"baseline_score": float(score), "baseline_decision": decision, "baseline_explanation": reasons})


def score_edges(edges: list[CandidateEdge], config: BaselineConfig | None = None) -> list[CandidateEdge]:
    return [score_edge(edge, config) for edge in edges]
