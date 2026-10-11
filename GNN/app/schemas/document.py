from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field


class PageRecord(BaseModel):
    page_id: str
    document_id: str
    page_number: int
    width: float
    height: float
    coordinate_system: str = "pixel_top_left"
    image_path: str | None = None
    transform: dict[str, float | str] = Field(default_factory=dict)


class DocumentRecord(BaseModel):
    document_id: str
    project_id: str = "pid-gnn-poc"
    revision: str = "demo"
    file_hash: str
    source_path: str
    pages: list[PageRecord] = Field(default_factory=list)
    processing_status: Literal["new", "ingested", "processed", "failed"] = "new"
    is_synthetic: bool = False

    @property
    def source_file(self) -> Path:
        return Path(self.source_path)
