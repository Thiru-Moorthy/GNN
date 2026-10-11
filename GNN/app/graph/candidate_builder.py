from __future__ import annotations

import itertools
import math
from dataclasses import dataclass

from app.config import CandidateConfig
from app.perception.line_extractor import LineSegment
from app.schemas.edge import CandidateEdge, LabelStatus, RelationshipType
from app.schemas.node import GraphNode


@dataclass(frozen=True)
class CandidateGraph:
    nodes: list[GraphNode]
    edges: list[CandidateEdge]
    page_sizes: dict[str, tuple[float, float]]


def _distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.dist(a, b)


def _bbox_gap(a: GraphNode, b: GraphNode) -> float:
    ax1, ay1, ax2, ay2 = a.bbox
    bx1, by1, bx2, by2 = b.bbox
    dx = max(bx1 - ax2, ax1 - bx2, 0)
    dy = max(by1 - ay2, ay1 - by2, 0)
    return (dx * dx + dy * dy) ** 0.5


def _near_line(node: GraphNode, lines: list[LineSegment]) -> bool:
    cx, cy = node.center
    for line in lines:
        x1, y1 = line.start
        x2, y2 = line.end
        if min(x1, x2) - 25 <= cx <= max(x1, x2) + 25 and min(y1, y2) - 25 <= cy <= max(y1, y2) + 25:
            return True
    return False


def build_candidate_graph(
    nodes: list[GraphNode],
    page_sizes: dict[str, tuple[float, float]],
    lines: list[LineSegment] | None = None,
    config: CandidateConfig | None = None,
) -> CandidateGraph:
    cfg = config or CandidateConfig()
    line_items = lines or []
    candidates: list[CandidateEdge] = []
    for source, target in itertools.combinations(sorted(nodes, key=lambda n: n.node_id), 2):
        if cfg.require_same_page and source.page_id != target.page_id:
            continue
        page_w, page_h = page_sizes[source.page_id]
        diag = (page_w**2 + page_h**2) ** 0.5
        distance = _distance(source.center, target.center)
        gap = _bbox_gap(source, target)
        if distance > cfg.hard_negative_distance_px:
            continue
        dy = abs(source.center[1] - target.center[1])
        dx = abs(source.center[0] - target.center[0])
        alignment = 1.0 - min(dy / max(dx, 1.0), 1.0)
        line_evidence = _near_line(source, line_items) and _near_line(target, line_items)
        edge_id = f"edge-{source.node_id}--{target.node_id}"
        candidates.append(
            CandidateEdge(
                edge_id=edge_id,
                source_node_id=source.node_id,
                target_node_id=target.node_id,
                candidate_relation_type=RelationshipType.physical_pipe,
                geometric_features={
                    "distance_norm": distance / diag,
                    "dx_norm": (target.center[0] - source.center[0]) / page_w,
                    "dy_norm": (target.center[1] - source.center[1]) / page_h,
                    "bbox_gap_norm": gap / diag,
                    "alignment_score": alignment,
                    "same_page": source.page_id == target.page_id,
                    "line_evidence": line_evidence,
                    "junction_evidence": False,
                    "source_confidence": source.detection_confidence,
                    "target_confidence": target.detection_confidence,
                    "distance_px": distance,
                    "bbox_gap_px": gap,
                },
                ground_truth_label=LabelStatus.unlabeled,
                label_status=LabelStatus.unlabeled,
                source_evidence=[source.node_id, target.node_id],
            )
        )
    return CandidateGraph(nodes=nodes, edges=candidates, page_sizes=page_sizes)
