from __future__ import annotations

import hashlib
from pathlib import Path

from app.config import IngestionConfig
from app.schemas.document import DocumentRecord, PageRecord


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_pdf_document(
    path: str | Path,
    output_dir: str | Path = "data/processed/pages",
    project_id: str = "pid-gnn-poc",
    config: IngestionConfig | None = None,
) -> DocumentRecord:
    cfg = config or IngestionConfig()
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f"Input PDF does not exist: {source}")
    if source.suffix.lower() != ".pdf":
        raise ValueError(f"Expected a PDF file, got {source.suffix}")
    try:
        import fitz  # type: ignore
    except Exception as exc:
        raise RuntimeError("PDF ingestion requires PyMuPDF (`pip install PyMuPDF`).") from exc

    file_hash = _sha256(source)
    doc_id = f"doc-{file_hash[:12]}"
    page_dir = Path(output_dir) / doc_id
    page_dir.mkdir(parents=True, exist_ok=True)
    pages: list[PageRecord] = []
    zoom = cfg.image_dpi / 72.0
    matrix = fitz.Matrix(zoom, zoom)
    with fitz.open(source) as pdf:
        for idx, page in enumerate(pdf, start=1):
            pix = page.get_pixmap(matrix=matrix, alpha=False)
            image_path = page_dir / f"page_{idx:04d}.png"
            pix.save(str(image_path))
            pages.append(
                PageRecord(
                    page_id=f"{doc_id}-p{idx:04d}",
                    document_id=doc_id,
                    page_number=idx,
                    width=float(pix.width),
                    height=float(pix.height),
                    coordinate_system=cfg.coordinate_system,
                    image_path=str(image_path),
                    transform={"dpi": cfg.image_dpi, "zoom": zoom, "source": "pdf_render"},
                )
            )
    return DocumentRecord(
        document_id=doc_id,
        project_id=project_id,
        file_hash=file_hash,
        source_path=str(source),
        pages=pages,
        processing_status="ingested",
    )
