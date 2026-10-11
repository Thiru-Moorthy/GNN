from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.baseline.geometry import score_edges
from app.config import CandidateConfig
from app.graph.candidate_builder import build_candidate_graph
from app.models.dataset import GraphExample, write_jsonl
from app.perception.detector import ComponentDetector
from app.perception.line_extractor import LineExtractor
from app.schemas.document import PageRecord
from app.schemas.edge import LabelStatus


def create_demo_examples(path: str | Path = "data/processed/synthetic_graphs.jsonl", count: int = 8) -> list[GraphExample]:
    examples: list[GraphExample] = []
    for idx in range(count):
        doc_id = f"synthetic-doc-{idx:03d}"
        page = PageRecord(page_id=f"{doc_id}-p0001", document_id=doc_id, page_number=1, width=1000, height=700)
        nodes = ComponentDetector().detect(page)
        # Slightly perturb tags through attributes without changing topology.
        for node in nodes:
            node.attributes["synthetic_document_index"] = idx
        lines = LineExtractor().extract(page)
        graph = build_candidate_graph(
            nodes,
            {page.page_id: (page.width, page.height)},
            lines,
            CandidateConfig(hard_negative_distance_px=1000.0),
        )
        positive_pairs = {
            tuple(sorted([f"{page.page_id}-pump-1", f"{page.page_id}-valve-1"])),
            tuple(sorted([f"{page.page_id}-tank-1", f"{page.page_id}-valve-1"])),
        }
        labeled_edges = []
        for edge in graph.edges:
            pair = tuple(sorted([edge.source_node_id, edge.target_node_id]))
            if pair in positive_pairs:
                label = LabelStatus.connected
            elif "inst-1" in edge.source_node_id or "inst-1" in edge.target_node_id:
                label = LabelStatus.ambiguous if idx % 3 == 0 else LabelStatus.unlabeled
            else:
                label = LabelStatus.not_connected
            labeled_edges.append(edge.model_copy(update={"ground_truth_label": label, "label_status": label}))
        scored_edges = score_edges(labeled_edges)
        examples.append(
            GraphExample(
                document_id=doc_id,
                is_synthetic=True,
                nodes=nodes,
                edges=scored_edges,
                page_sizes={page.page_id: (page.width, page.height)},
                metadata={"source": "explicit_synthetic_topology", "expected_connected_edges": sorted(["pump-valve", "valve-tank"])},
            )
        )
    write_jsonl(examples, path)
    return examples


if __name__ == "__main__":
    examples = create_demo_examples()
    print(f"Wrote {len(examples)} synthetic graph examples to data/processed/synthetic_graphs.jsonl")
