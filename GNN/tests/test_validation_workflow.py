from pathlib import Path

from PIL import Image

from app.ingestion.image_loader import load_image_document
from app.orchestration.state import WorkflowState
from app.orchestration.workflow import run_local_workflow
from app.schemas.edge import CandidateEdge
from app.schemas.prediction import EdgePrediction
from app.validation.confidence import validate_edge


def test_validation_routes_review_without_prediction():
    edge = CandidateEdge(edge_id="e1", source_node_id="a", target_node_id="b", baseline_score=0.5)
    rel = validate_edge(edge, None)
    assert rel.validation_status == "REVIEW"
    assert rel.reviewer_status == "pending"


def test_validation_accepts_high_confidence():
    edge = CandidateEdge(edge_id="e1", source_node_id="a", target_node_id="b", baseline_score=0.7)
    pred = EdgePrediction(edge_id="e1", model_name="m", model_version="v", probability=0.9, predicted_class="connected")
    rel = validate_edge(edge, pred)
    assert rel.validation_status == "ACCEPT"


def test_image_ingestion_and_local_workflow(tmp_path: Path):
    image_path = tmp_path / "sample.png"
    Image.new("RGB", (640, 480), "white").save(image_path)
    document = load_image_document(image_path)
    assert document.pages[0].width == 640
    state = run_local_workflow(WorkflowState(run_id="pytest-workflow", document_path=str(image_path)))
    assert state.status == "completed"
    assert "knowledge_graph" in state.artifact_refs
