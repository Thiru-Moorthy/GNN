from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, Field


class EdgePrediction(BaseModel):
    edge_id: str
    model_name: str
    model_version: str
    probability: float
    predicted_class: Literal["connected", "not_connected"]
    decision_status: Literal["predicted", "fallback", "unavailable"] = "predicted"
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
