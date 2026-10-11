from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from app.schemas.edge import ValidatedRelationship
from app.schemas.node import GraphNode


def build_knowledge_graph_payload(nodes: list[GraphNode], relationships: list[ValidatedRelationship]) -> dict[str, Any]:
    return {
        "nodes": [node.model_dump(mode="json") for node in nodes],
        "relationships": [rel.model_dump(mode="json") for rel in relationships],
        "metadata": {"graph_backend": "json_networkx_compatible", "accepted_relationships": sum(r.validation_status == "ACCEPT" for r in relationships)},
    }


def save_knowledge_graph(path: str | Path, nodes: list[GraphNode], relationships: list[ValidatedRelationship]) -> dict[str, Any]:
    payload = build_knowledge_graph_payload(nodes, relationships)
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def export_graphml(path: str | Path, nodes: list[GraphNode], relationships: list[ValidatedRelationship]) -> None:
    try:
        import networkx as nx  # type: ignore
    except Exception as exc:
        raise RuntimeError("GraphML export requires networkx.") from exc
    graph = nx.Graph()
    for node in nodes:
        graph.add_node(node.node_id, **node.model_dump(mode="json"))
    for rel in relationships:
        if rel.validation_status == "ACCEPT":
            graph.add_edge(rel.source_node_id, rel.target_node_id, **rel.model_dump(mode="json"))
    nx.write_graphml(graph, path)
