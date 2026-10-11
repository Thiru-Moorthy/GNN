from __future__ import annotations

import json
from pathlib import Path

from app.schemas.edge import CandidateEdge, ValidatedRelationship
from app.schemas.node import GraphNode
from app.schemas.prediction import EdgePrediction


def save_visualization_json(
    path: str | Path,
    nodes: list[GraphNode],
    candidates: list[CandidateEdge],
    predictions: list[EdgePrediction],
    relationships: list[ValidatedRelationship],
) -> None:
    prediction_map = {item.edge_id: item.model_dump(mode="json") for item in predictions}
    relationship_map = {"edge-" + rel.source_node_id + "--" + rel.target_node_id: rel.model_dump(mode="json") for rel in relationships}
    payload = {
        "legend": {
            "candidate": "gray dashed",
            "accepted": "green solid",
            "review": "orange dotted",
            "rejected": "red faded",
        },
        "nodes": [node.model_dump(mode="json") for node in nodes],
        "edges": [
            {
                **edge.model_dump(mode="json"),
                "prediction": prediction_map.get(edge.edge_id),
                "validation": relationship_map.get(edge.edge_id),
            }
            for edge in candidates
        ],
    }
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
