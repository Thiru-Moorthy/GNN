from __future__ import annotations

from app.schemas.edge import CandidateEdge, LabelStatus
from app.schemas.prediction import EdgePrediction


def binary_metrics(y_true: list[int], y_score: list[float], threshold: float = 0.5) -> dict[str, float | list[list[int]]]:
    y_pred = [1 if score >= threshold else 0 for score in y_score]
    tp = sum(yt == 1 and yp == 1 for yt, yp in zip(y_true, y_pred, strict=True))
    tn = sum(yt == 0 and yp == 0 for yt, yp in zip(y_true, y_pred, strict=True))
    fp = sum(yt == 0 and yp == 1 for yt, yp in zip(y_true, y_pred, strict=True))
    fn = sum(yt == 1 and yp == 0 for yt, yp in zip(y_true, y_pred, strict=True))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "confusion_matrix": [[tn, fp], [fn, tp]]}


def labeled_edge_arrays(edges: list[CandidateEdge], predictions: list[EdgePrediction]) -> tuple[list[int], list[float]]:
    by_id = {prediction.edge_id: prediction for prediction in predictions}
    y_true: list[int] = []
    scores: list[float] = []
    for edge in edges:
        if edge.label_status not in {LabelStatus.connected, LabelStatus.not_connected}:
            continue
        if edge.edge_id not in by_id:
            continue
        y_true.append(1 if edge.label_status == LabelStatus.connected else 0)
        scores.append(by_id[edge.edge_id].probability)
    return y_true, scores


def candidate_generation_recall(edges: list[CandidateEdge], expected_connected_edge_ids: set[str] | None = None) -> float | None:
    if expected_connected_edge_ids is None:
        return None
    if not expected_connected_edge_ids:
        return 1.0
    generated = {edge.edge_id for edge in edges}
    return len(generated & expected_connected_edge_ids) / len(expected_connected_edge_ids)
