from __future__ import annotations

import hashlib
from pathlib import Path

from PIL import Image

from app.schemas.document import DocumentRecord, PageRecord

SUPPORTED_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_image_document(path: str | Path, project_id: str = "pid-gnn-poc") -> DocumentRecord:
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f"Input image does not exist: {source}")
    if source.suffix.lower() not in SUPPORTED_IMAGE_EXTENSIONS:
        raise ValueError(f"Unsupported image extension {source.suffix}. Supported: {sorted(SUPPORTED_IMAGE_EXTENSIONS)}")
    try:
        with Image.open(source) as img:
            width, height = img.size
    except Exception as exc:
        raise ValueError(f"Could not open image {source}: {exc}") from exc
    file_hash = file_sha256(source)
    doc_id = f"doc-{file_hash[:12]}"
    page = PageRecord(
        page_id=f"{doc_id}-p0001",
        document_id=doc_id,
        page_number=1,
        width=float(width),
        height=float(height),
        image_path=str(source),
    )
    return DocumentRecord(
        document_id=doc_id,
        project_id=project_id,
        file_hash=file_hash,
        source_path=str(source),
        pages=[page],
        processing_status="ingested",
    )
