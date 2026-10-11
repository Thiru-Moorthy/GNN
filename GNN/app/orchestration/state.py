from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class WorkflowState(BaseModel):
    run_id: str
    document_path: str | None = None
    document_id: str | None = None
    status: Literal["new", "running", "completed", "failed"] = "new"
    errors: list[str] = Field(default_factory=list)
    artifact_refs: dict[str, str] = Field(default_factory=dict)
    counts: dict[str, int] = Field(default_factory=dict)
    data: dict[str, Any] = Field(default_factory=dict)
