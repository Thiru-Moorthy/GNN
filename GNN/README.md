# P&ID AI Platform - GNN Connectivity POC

This repository is a runnable proof of concept for Piping and Instrumentation Diagram connectivity analysis. It demonstrates the complete local pipeline:

P&ID PDF/image -> ingestion -> perception stubs or adapters -> candidate graph -> geometry baseline -> GraphSAGE edge classifier -> validation -> knowledge graph -> benchmark report -> optional Streamlit/API/workflow execution.

The core ML task is candidate-edge classification: predict whether two graph elements are physically connected. The code keeps observed detections, candidate relationships, ground-truth labels, GNN predictions, validated relationships, ambiguous records, and unknown records separate.

## Current Status

- The repository was empty when this POC was created.
- No pretrained YOLO weights, OCR engine, real annotations, or production P&ID dataset are bundled.
- The synthetic demo data is explicitly marked synthetic and must not be interpreted as real P&ID performance.
- PDF ingestion requires `PyMuPDF`. Raster image ingestion works through Pillow.
- The GNN implementation requires `torch` and `torch-geometric`. If they are absent, scripts report that the GNN path is unavailable and use baseline fallback probabilities for non-training workflow runs.

## Install

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

On Linux/macOS:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Commands

Generate synthetic demo data:

```bash
python scripts/create_demo_data.py
```

Run the end-to-end demo:

```bash
python scripts/run_demo.py
```

Train the GNN:

```bash
python -m app.models.train --config configs/train.yaml
```

Evaluate baseline versus GNN:

```bash
python -m app.evaluation.benchmark --config configs/evaluation.yaml
```

Run tests:

```bash
pytest -q
```

Optional UI:

```bash
streamlit run app/main.py
```

Optional API:

```bash
uvicorn app.api.routes:create_app --factory --reload
```

## Repository Structure

```text
app/
  api/                 Optional FastAPI app factory
  baseline/            Geometry and rule baseline
  evaluation/          Metrics, benchmark CLI, Markdown/JSON reports
  graph/               Candidate graph, features, KG persistence, visualization JSON
  ingestion/           PDF/image loading and coordinate metadata
  models/              Dataset loader, GraphSAGE model, training, inference, checkpoints
  orchestration/       LangGraph-compatible workflow and local fallback runner
  perception/          YOLO/OCR/line extraction interfaces with demo fallback
  schemas/             Pydantic document/node/edge/prediction schemas
  validation/          Confidence and topology validation
configs/               JSON-compatible YAML configs
data/                  Sample, annotation, and processed data folders
scripts/               Demo data, training, evaluation, end-to-end demo scripts
tests/                 Pytest coverage for core POC behavior
artifacts/             Checkpoints, reports, graphs, visualizations
```

## Data Model

Labels use explicit states:

- `connected`
- `not_connected`
- `ambiguous`
- `unlabeled`

Only `connected` and `not_connected` are used for supervised loss. Ambiguous and unlabeled edges are preserved but excluded from training and labeled metrics.

Validated relationships use:

- `ACCEPT`
- `REVIEW`
- `REJECT`

Uncertain connections are routed to review and remain uncertain unless a reviewer or later evidence resolves them.

## Annotation Format

The training loader expects JSONL records matching `app.models.dataset.GraphExample`:

```json
{
  "document_id": "drawing-001",
  "is_synthetic": false,
  "nodes": [],
  "edges": [],
  "page_sizes": {"page-1": [1000, 700]},
  "metadata": {"project": "example"}
}
```

Each edge should include `label_status`. Do not encode missing annotations as `not_connected`.

## Outputs

Typical artifacts:

- `data/processed/synthetic_graphs.jsonl`
- `artifacts/checkpoints/best_model.pt`
- `artifacts/checkpoints/best_model.metadata.json`
- `artifacts/checkpoints/best_model.features.json`
- `artifacts/checkpoints/training_metrics.csv`
- `artifacts/reports/benchmark_report.md`
- `artifacts/reports/benchmark_report.json`
- `artifacts/graphs/demo_knowledge_graph.json`
- `artifacts/visualizations/demo_graph.json`

## Limitations and Next Steps

Real P&ID evaluation requires verified annotations. To evaluate on real drawings:

1. Add real PDFs/images under `data/sample/`.
2. Provide YOLO weights or implement the detector adapter in `app/perception/detector.py`.
3. Add an OCR adapter if tag extraction is needed beyond demo mode.
4. Convert verified annotations to the JSONL schema.
5. Run training and benchmark commands on project-level splits.
6. Inspect false positives/false negatives in the generated report and visualization JSON.
