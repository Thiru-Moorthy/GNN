from __future__ import annotations

from pydantic import BaseModel

from app.schemas.document import PageRecord


class OCRText(BaseModel):
    text_id: str
    page_id: str
    text: str
    bbox: tuple[float, float, float, float]
    confidence: float
    source: str = "demo_ocr"


class OCRExtractor:
    def __init__(self, engine: str = "demo") -> None:
        self.engine = engine

    def extract(self, page: PageRecord) -> list[OCRText]:
        if self.engine != "demo":
            raise NotImplementedError("OCR engine is replaceable; configure a project OCR adapter here.")
        w, h = page.width, page.height
        return [
            OCRText(text_id=f"{page.page_id}-txt-P101", page_id=page.page_id, text="P-101", bbox=(0.1 * w, 0.34 * h, 0.22 * w, 0.39 * h), confidence=0.95),
            OCRText(text_id=f"{page.page_id}-txt-L101", page_id=page.page_id, text="LINE-101", bbox=(0.46 * w, 0.55 * h, 0.62 * w, 0.60 * h), confidence=0.91),
        ]
