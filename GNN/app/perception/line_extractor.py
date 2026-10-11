from __future__ import annotations

from pydantic import BaseModel

from app.schemas.document import PageRecord


class LineSegment(BaseModel):
    line_id: str
    page_id: str
    start: tuple[float, float]
    end: tuple[float, float]
    confidence: float = 1.0
    line_type: str = "physical_pipe"
    source: str = "demo_line_extractor"


class LineExtractor:
    def __init__(self, mode: str = "demo") -> None:
        self.mode = mode

    def extract(self, page: PageRecord) -> list[LineSegment]:
        if self.mode != "demo":
            raise NotImplementedError("OpenCV line extraction hook is intentionally replaceable in this POC.")
        w, h = page.width, page.height
        y = 0.48 * h
        return [
            LineSegment(line_id=f"{page.page_id}-line-1", page_id=page.page_id, start=(0.22 * w, y), end=(0.39 * w, y), confidence=0.9),
            LineSegment(line_id=f"{page.page_id}-line-2", page_id=page.page_id, start=(0.45 * w, y), end=(0.68 * w, y), confidence=0.9),
            LineSegment(line_id=f"{page.page_id}-signal-1", page_id=page.page_id, start=(0.42 * w, 0.27 * h), end=(0.42 * w, 0.43 * h), confidence=0.75, line_type="signal"),
        ]
