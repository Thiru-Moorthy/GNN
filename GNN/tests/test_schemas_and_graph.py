from app.baseline.geometry import score_edges
from app.graph.candidate_builder import build_candidate_graph
from app.graph.features import FeatureSchema, edge_feature_vector
from app.perception.detector import ComponentDetector
from app.perception.line_extractor import LineExtractor
from app.schemas.document import PageRecord
from app.schemas.edge import CandidateEdge, LabelStatus
from app.schemas.node import GraphNode, NodeType


def test_schema_rejects_invalid_bbox():
    try:
        GraphNode(node_id="n1", page_id="p1", node_type=NodeType.equipment, class_id="pump", bbox=(10, 10, 1, 1), center=(5, 5))
    except ValueError as exc:
        assert "Invalid bbox" in str(exc)
    else:
        raise AssertionError("Expected invalid bbox to fail")


def test_candidate_graph_and_baseline_scores():
    page = PageRecord(page_id="p1", document_id="d1", page_number=1, width=1000, height=700)
    nodes = ComponentDetector().detect(page)
    lines = LineExtractor().extract(page)
    graph = build_candidate_graph(nodes, {"p1": (1000, 700)}, lines)
    assert graph.edges
    scored = score_edges(graph.edges)
    assert all(edge.baseline_score is not None for edge in scored)


def test_feature_mapping_is_deterministic():
    edge = CandidateEdge(
        edge_id="e1",
        source_node_id="a",
        target_node_id="b",
        geometric_features={"distance_norm": 0.1, "alignment_score": 0.8, "same_page": True},
        label_status=LabelStatus.connected,
    )
    schema = FeatureSchema()
    assert edge_feature_vector(edge, schema) == edge_feature_vector(edge, schema)
    assert len(edge_feature_vector(edge, schema)) == len(schema.edge_feature_names)
