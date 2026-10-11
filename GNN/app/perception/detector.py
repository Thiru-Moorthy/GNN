from __future__ import annotations

from pathlib import Path

from app.schemas.document import PageRecord
from app.schemas.node import GraphNode, NodeType


class ComponentDetector:
    """YOLO-compatible detector facade with a deterministic demo fallback."""

    def __init__(self, weights_path: str | None = None, class_map: dict[int, str] | None = None) -> None:
        self.weights_path = weights_path
        self.class_map = class_map or {}
        self.model_version = "demo-detector-v1"

    def detect(self, page: PageRecord) -> list[GraphNode]:
        if self.weights_path:
            weights = Path(self.weights_path)
            if not weights.exists():
                raise FileNotFoundError(f"Detector weights not found: {weights}")
            raise NotImplementedError(
                "YOLO inference hook is configured but not bundled in the POC. "
                "Install ultralytics and adapt ComponentDetector.detect for project weights."
            )
        return self._demo_detections(page)

    def _demo_detections(self, page: PageRecord) -> list[GraphNode]:
        w, h = page.width, page.height
        specs = [
            ("pump-1", NodeType.equipment, "pump", "P-101", (0.10, 0.40, 0.22, 0.55)),
            ("valve-1", NodeType.valve, "valve", "V-101", (0.39, 0.43, 0.45, 0.52)),
            ("tank-1", NodeType.equipment, "vessel", "T-101", (0.68, 0.35, 0.84, 0.60)),
            ("inst-1", NodeType.instrument, "instrument", "PI-101", (0.38, 0.18, 0.46, 0.27)),
        ]
        nodes: list[GraphNode] = []
        for suffix, node_type, class_id, tag, rel in specs:
            x1, y1, x2, y2 = rel
            bbox = (x1 * w, y1 * h, x2 * w, y2 * h)
            center = ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)
            nodes.append(
                GraphNode(
                    node_id=f"{page.page_id}-{suffix}",
                    page_id=page.page_id,
                    node_type=node_type,
                    class_id=class_id,
                    tag=tag,
                    bbox=bbox,
                    center=center,
                    detection_confidence=0.99,
                    source_evidence=[f"{page.page_id}:demo_detection"],
                    annotation_status="synthetic",
                )
            )
        return nodes
