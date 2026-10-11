from app.evaluation.metrics import binary_metrics, labeled_edge_arrays
from app.models.dataset import GraphExample, validate_label_integrity
from app.schemas.edge import CandidateEdge, LabelStatus
from app.schemas.node import GraphNode, NodeType
from app.schemas.prediction import EdgePrediction


def _node(node_id: str) -> GraphNode:
    return GraphNode(node_id=node_id, page_id="p1", node_type=NodeType.equipment, class_id="pump", bbox=(0, 0, 10, 10), center=(5, 5))


def test_unlabeled_and_ambiguous_are_not_supervised():
    edges = [
        CandidateEdge(edge_id="e1", source_node_id="a", target_node_id="b", label_status=LabelStatus.connected),
        CandidateEdge(edge_id="e2", source_node_id="a", target_node_id="c", label_status=LabelStatus.ambiguous),
        CandidateEdge(edge_id="e3", source_node_id="b", target_node_id="c", label_status=LabelStatus.unlabeled),
    ]
    example = GraphExample(document_id="d1", nodes=[_node("a"), _node("b"), _node("c")], edges=edges, page_sizes={"p1": (100, 100)})
    assert [edge.edge_id for edge in example.supervised_edges()] == ["e1"]
    counts = validate_label_integrity([example])
    assert counts["ambiguous"] == 1
    assert counts["unlabeled"] == 1


def test_metric_edge_cases():
    metrics = binary_metrics([1, 0, 1, 0], [0.9, 0.8, 0.2, 0.1], 0.5)
    assert metrics["confusion_matrix"] == [[1, 1], [1, 1]]
    assert metrics["precision"] == 0.5


def test_labeled_edge_arrays_excludes_unknown_labels():
    edges = [
        CandidateEdge(edge_id="e1", source_node_id="a", target_node_id="b", label_status=LabelStatus.connected),
        CandidateEdge(edge_id="e2", source_node_id="a", target_node_id="c", label_status=LabelStatus.unlabeled),
    ]
    preds = [
        EdgePrediction(edge_id="e1", model_name="m", model_version="v", probability=0.7, predicted_class="connected"),
        EdgePrediction(edge_id="e2", model_name="m", model_version="v", probability=0.1, predicted_class="not_connected"),
    ]
    assert labeled_edge_arrays(edges, preds) == ([1], [0.7])
