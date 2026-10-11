from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class PathsConfig(BaseModel):
    data_dir: Path = Path("data")
    artifacts_dir: Path = Path("artifacts")
    processed_dir: Path = Path("data/processed")


class IngestionConfig(BaseModel):
    image_dpi: int = 200
    coordinate_system: str = "pixel_top_left"


class CandidateConfig(BaseModel):
    max_distance_px: float = 180.0
    hard_negative_distance_px: float = 320.0
    require_same_page: bool = True


class BaselineConfig(BaseModel):
    accept_threshold: float = 0.65
    review_threshold: float = 0.4
    max_distance_px: float = 180.0


class ModelConfig(BaseModel):
    hidden_dim: int = 128
    dropout: float = 0.2
    learning_rate: float = 0.005
    epochs: int = 30
    patience: int = 8
    batch_seed: int = 42
    threshold: float = 0.5
    device: str = "auto"


class ValidationConfig(BaseModel):
    accept_probability: float = 0.72
    review_probability: float = 0.35
    min_baseline_for_accept: float = 0.35


class AppConfig(BaseModel):
    project_id: str = "pid-gnn-poc"
    run_id: str = "local-demo"
    paths: PathsConfig = Field(default_factory=PathsConfig)
    ingestion: IngestionConfig = Field(default_factory=IngestionConfig)
    candidates: CandidateConfig = Field(default_factory=CandidateConfig)
    baseline: BaselineConfig = Field(default_factory=BaselineConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    validation: ValidationConfig = Field(default_factory=ValidationConfig)
    dataset_path: Path = Path("data/processed/synthetic_graphs.jsonl")
    checkpoint_path: Path = Path("artifacts/checkpoints/best_model.pt")
    report_path: Path = Path("artifacts/reports/benchmark_report.md")


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml  # type: ignore
    except Exception as exc:  # pragma: no cover - depends on optional dependency
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            raise RuntimeError("YAML config requires PyYAML unless the file uses JSON-compatible YAML.") from exc
        if not isinstance(data, dict):
            raise ValueError(f"Config {path} must contain a mapping.")
        return data
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config {path} must contain a mapping.")
    return data


def load_config(path: str | Path | None = None) -> AppConfig:
    if path is None:
        return AppConfig()
    cfg_path = Path(path)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config file does not exist: {cfg_path}")
    if cfg_path.suffix.lower() in {".yaml", ".yml"}:
        raw = _load_yaml(cfg_path)
    else:
        raw = json.loads(cfg_path.read_text(encoding="utf-8"))
    return AppConfig.model_validate(raw)


def ensure_directories(config: AppConfig) -> None:
    for path in [
        config.paths.data_dir,
        config.paths.artifacts_dir,
        config.paths.processed_dir,
        config.checkpoint_path.parent,
        config.report_path.parent,
    ]:
        path.mkdir(parents=True, exist_ok=True)
