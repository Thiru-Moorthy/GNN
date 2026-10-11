from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.baseline.geometry import score_edges
from app.config import load_config, ensure_directories
from app.evaluation.benchmark import benchmark
from app.graph.knowledge_graph import save_knowledge_graph
from app.graph.visualization import save_visualization_json
from app.models.dataset import read_jsonl
from app.models.predict import fallback_predictions, predict_with_checkpoint
from app.validation.topology import validate_predictions
from scripts.create_demo_data import create_demo_examples


def run_demo() -> dict[str, object]:
    config = load_config("configs/train.yaml")
    ensure_directories(config)
    create_demo_examples(config.dataset_path)
    training_result: dict[str, object]
    try:
        from app.models.train import train

        training_result = train("configs/train.yaml")
        trained = True
    except RuntimeError as exc:
        training_result = {"status": "skipped", "reason": str(exc)}
        trained = False

    examples = read_jsonl(config.dataset_path)
    first = examples[0]
    scored_edges = score_edges(first.edges, config.baseline)
    first = first.model_copy(update={"edges": scored_edges})
    if trained and Path(config.checkpoint_path).exists():
        predictions = predict_with_checkpoint(first, str(config.checkpoint_path))
    else:
        predictions = fallback_predictions(first, "gnn-unavailable")
    relationships = validate_predictions(scored_edges, predictions)
    kg_path = Path("artifacts/graphs/demo_knowledge_graph.json")
    viz_path = Path("artifacts/visualizations/demo_graph.json")
    save_knowledge_graph(kg_path, first.nodes, relationships)
    save_visualization_json(viz_path, first.nodes, scored_edges, predictions, relationships)
    benchmark_result = benchmark("configs/evaluation.yaml")
    return {
        "trained_gnn": trained,
        "training": training_result,
        "benchmark": benchmark_result,
        "knowledge_graph": str(kg_path),
        "visualization": str(viz_path),
    }


if __name__ == "__main__":
    print(json.dumps(run_demo(), indent=2, default=str))
