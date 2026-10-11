from __future__ import annotations

from app.schemas.document import PageRecord


def normalize_bbox(bbox: tuple[float, float, float, float], page: PageRecord) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = bbox
    return (x1 / page.width, y1 / page.height, x2 / page.width, y2 / page.height)


def normalize_point(point: tuple[float, float], page: PageRecord) -> tuple[float, float]:
    x, y = point
    return (x / page.width, y / page.height)


def page_diagonal(page: PageRecord) -> float:
    return (page.width**2 + page.height**2) ** 0.5
