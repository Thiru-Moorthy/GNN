from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from app.graph.candidate_builder import CandidateGraph
from app.graph.features import FeatureSchema, edge_feature_vector, node_feature_vector
from app.schemas.edge import CandidateEdge, LabelStatus
from app.schemas.node import GraphNode


class GraphExample(BaseModel):
    document_id: str
    is_synthetic: bool = True
    nodes: list[GraphNode]
    edges: list[CandidateEdge]
    page_sizes: dict[str, tuple[float, float]]
    metadata: dict[str, Any] = Field(default_factory=dict)

    def supervised_edges(self) -> list[CandidateEdge]:
        return [edge for edge in self.edges if edge.is_supervised_label]


def write_jsonl(examples: list[GraphExample], path: str | Path) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        for example in examples:
            fh.write(example.model_dump_json() + "\n")


def read_jsonl(path: str | Path) -> list[GraphExample]:
    src = Path(path)
    if not src.exists():
        raise FileNotFoundError(f"Dataset does not exist: {src}")
    examples: list[GraphExample] = []
    with src.open("r", encoding="utf-8") as fh:
        for line_number, line in enumerate(fh, start=1):
            if not line.strip():
                continue
            try:
                examples.append(GraphExample.model_validate_json(line))
            except Exception as exc:
                raise ValueError(f"Invalid graph example at {src}:{line_number}: {exc}") from exc
    return examples


def validate_label_integrity(examples: list[GraphExample]) -> dict[str, int]:
    counts = {status.value: 0 for status in LabelStatus}
    for example in examples:
        for edge in example.edges:
            counts[edge.label_status.value] += 1
            if edge.label_status in {LabelStatus.ambiguous, LabelStatus.unlabeled} and edge.ground_truth_label in {
                LabelStatus.connected,
                LabelStatus.not_connected,
            }:
                raise ValueError(f"Edge {edge.edge_id} has unresolved label_status but supervised ground_truth_label.")
    return counts


def split_examples(
    examples: list[GraphExample],
    seed: int = 42,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
) -> tuple[list[GraphExample], list[GraphExample], list[GraphExample]]:
    shuffled = list(examples)
    random.Random(seed).shuffle(shuffled)
    n = len(shuffled)
    train_end = max(1, int(n * train_ratio)) if n else 0
    val_end = min(n, train_end + max(1, int(n * val_ratio))) if n > 2 else train_end
    return shuffled[:train_end], shuffled[train_end:val_end], shuffled[val_end:]


def graph_to_tensors(example: GraphExample, schema: FeatureSchema):
    try:
        import torch
        from torch_geometric.data import Data  # type: ignore
    except Exception as exc:
        raise RuntimeError("GNN tensors require torch and torch_geometric. Install requirements.txt.") from exc

    node_index = {node.node_id: idx for idx, node in enumerate(example.nodes)}
    x_rows = [node_feature_vector(node, example.page_sizes[node.page_id], schema) for node in example.nodes]
    edge_pairs: list[list[int]] = []
    edge_attr_rows: list[list[float]] = []
    labels: list[float] = []
    supervised_mask: list[bool] = []
    edge_ids: list[str] = []
    for edge in example.edges:
        if edge.source_node_id not in node_index or edge.target_node_id not in node_index:
            raise ValueError(f"Edge {edge.edge_id} references unknown nodes.")
        src = node_index[edge.source_node_id]
        dst = node_index[edge.target_node_id]
        edge_pairs.append([src, dst])
        edge_pairs.append([dst, src])
        edge_attr = edge_feature_vector(edge, schema)
        edge_attr_rows.append(edge_attr)
        edge_attr_rows.append(edge_attr)
        label = 1.0 if edge.label_status == LabelStatus.connected else 0.0
        supervised = edge.is_supervised_label
        labels.extend([label, label])
        supervised_mask.extend([supervised, supervised])
        edge_ids.extend([edge.edge_id, edge.edge_id])
    data = Data(
        x=torch.tensor(x_rows, dtype=torch.float32),
        edge_index=torch.tensor(edge_pairs, dtype=torch.long).t().contiguous() if edge_pairs else torch.empty((2, 0), dtype=torch.long),
        edge_attr=torch.tensor(edge_attr_rows, dtype=torch.float32) if edge_attr_rows else torch.empty((0, len(schema.edge_feature_names)), dtype=torch.float32),
        y=torch.tensor(labels, dtype=torch.float32),
        supervised_mask=torch.tensor(supervised_mask, dtype=torch.bool),
    )
    data.edge_ids = edge_ids
    return data


def candidate_graph_to_example(document_id: str, graph: CandidateGraph, is_synthetic: bool = False) -> GraphExample:
    return GraphExample(document_id=document_id, nodes=graph.nodes, edges=graph.edges, page_sizes=graph.page_sizes, is_synthetic=is_synthetic)


def dataset_manifest(examples: list[GraphExample]) -> dict[str, Any]:
    counts = validate_label_integrity(examples)
    return {
        "num_documents": len(examples),
        "num_nodes": sum(len(example.nodes) for example in examples),
        "num_candidate_edges": sum(len(example.edges) for example in examples),
        "label_counts": counts,
        "synthetic_documents": sum(example.is_synthetic for example in examples),
    }
