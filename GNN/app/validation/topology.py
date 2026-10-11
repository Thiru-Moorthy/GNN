from __future__ import annotations

from app.schemas.edge import CandidateEdge, ValidatedRelationship
from app.schemas.prediction import EdgePrediction
from app.validation.confidence import validate_edge


def validate_predictions(edges: list[CandidateEdge], predictions: list[EdgePrediction]) -> list[ValidatedRelationship]:
    by_id = {prediction.edge_id: prediction for prediction in predictions}
    return [validate_edge(edge, by_id.get(edge.edge_id)) for edge in edges]
