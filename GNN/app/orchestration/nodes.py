from __future__ import annotations

from pathlib import Path

from app.baseline.geometry import score_edges
from app.graph.candidate_builder import build_candidate_graph
from app.graph.knowledge_graph import save_knowledge_graph
from app.graph.visualization import save_visualization_json
from app.ingestion.image_loader import SUPPORTED_IMAGE_EXTENSIONS, load_image_document
from app.ingestion.pdf_loader import load_pdf_document
from app.models.dataset import GraphExample
from app.models.predict import fallback_predictions, predict_with_checkpoint
from app.perception.detector import ComponentDetector
from app.perception.line_extractor import LineExtractor
from app.perception.ocr import OCRExtractor
from app.validation.topology import validate_predictions
from app.orchestration.state import WorkflowState


def ingest_document(state: WorkflowState) -> WorkflowState:
    if not state.document_path:
        raise ValueError("document_path is required")
    path = Path(state.document_path)
    if path.suffix.lower() == ".pdf":
        document = load_pdf_document(path)
    elif path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS:
        document = load_image_document(path)
    else:
        raise ValueError(f"Unsupported document type: {path.suffix}")
    state.document_id = document.document_id
    state.data["document"] = document
    state.counts["pages"] = len(document.pages)
    return state


def normalize_pages(state: WorkflowState) -> WorkflowState:
    state.artifact_refs["coordinate_system"] = "pixel_top_left_normalized_metadata"
    return state


def detect_components(state: WorkflowState) -> WorkflowState:
    document = state.data["document"]
    nodes = []
    detector = ComponentDetector()
    for page in document.pages:
        nodes.extend(detector.detect(page))
    state.data["nodes"] = nodes
    state.counts["nodes"] = len(nodes)
    return state


def extract_text_and_lines(state: WorkflowState) -> WorkflowState:
    document = state.data["document"]
    texts = []
    lines = []
    for page in document.pages:
        texts.extend(OCRExtractor().extract(page))
        lines.extend(LineExtractor().extract(page))
    state.data["ocr"] = texts
    state.data["lines"] = lines
    state.counts["ocr_texts"] = len(texts)
    state.counts["line_segments"] = len(lines)
    return state


def build_candidate_graph_node(state: WorkflowState) -> WorkflowState:
    document = state.data["document"]
    page_sizes = {page.page_id: (page.width, page.height) for page in document.pages}
    graph = build_candidate_graph(state.data["nodes"], page_sizes, state.data.get("lines", []))
    state.data["candidate_graph"] = graph
    state.counts["candidate_edges"] = len(graph.edges)
    return state


def run_baseline(state: WorkflowState) -> WorkflowState:
    graph = state.data["candidate_graph"]
    graph = graph.__class__(nodes=graph.nodes, edges=score_edges(graph.edges), page_sizes=graph.page_sizes)
    state.data["candidate_graph"] = graph
    return state


def run_gnn(state: WorkflowState) -> WorkflowState:
    graph = state.data["candidate_graph"]
    example = GraphExample(document_id=state.document_id or state.run_id, is_synthetic=False, nodes=graph.nodes, edges=graph.edges, page_sizes=graph.page_sizes)
    checkpoint = Path("artifacts/checkpoints/best_model.pt")
    if checkpoint.exists():
        try:
            predictions = predict_with_checkpoint(example, str(checkpoint))
        except RuntimeError as exc:
            state.errors.append(str(exc))
            predictions = fallback_predictions(example)
    else:
        state.errors.append("GNN checkpoint unavailable; using baseline fallback probabilities.")
        predictions = fallback_predictions(example)
    state.data["predictions"] = predictions
    return state


def validate_topology(state: WorkflowState) -> WorkflowState:
    graph = state.data["candidate_graph"]
    state.data["relationships"] = validate_predictions(graph.edges, state.data["predictions"])
    return state


def route_uncertain_edges(state: WorkflowState) -> WorkflowState:
    relationships = state.data["relationships"]
    state.counts["review_edges"] = sum(rel.validation_status == "REVIEW" for rel in relationships)
    return state


def publish_knowledge_graph(state: WorkflowState) -> WorkflowState:
    graph = state.data["candidate_graph"]
    path = Path("artifacts/graphs") / f"{state.run_id}_knowledge_graph.json"
    save_knowledge_graph(path, graph.nodes, state.data["relationships"])
    state.artifact_refs["knowledge_graph"] = str(path)
    return state


def evaluate_connectivity(state: WorkflowState) -> WorkflowState:
    # Real benchmark is run separately on labeled datasets; live document workflow may be unlabeled.
    state.artifact_refs["evaluation_note"] = "No ground truth labels supplied for this workflow run."
    return state


def generate_outputs(state: WorkflowState) -> WorkflowState:
    graph = state.data["candidate_graph"]
    path = Path("artifacts/visualizations") / f"{state.run_id}_visualization.json"
    save_visualization_json(path, graph.nodes, graph.edges, state.data["predictions"], state.data["relationships"])
    state.artifact_refs["visualization"] = str(path)
    state.status = "completed"
    return state
