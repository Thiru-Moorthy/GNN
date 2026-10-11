from __future__ import annotations

from app.orchestration.state import WorkflowState
from app.orchestration.workflow import run_local_workflow


def create_app():
    try:
        from fastapi import FastAPI
        from pydantic import BaseModel
    except Exception as exc:
        raise RuntimeError("FastAPI API requires fastapi. Install requirements.txt.") from exc

    class ProcessRequest(BaseModel):
        document_path: str
        run_id: str = "api-run"

    api = FastAPI(title="P&ID GNN Connectivity POC")

    @api.post("/documents/{document_id}/process")
    def process_document(document_id: str, request: ProcessRequest):
        state = run_local_workflow(WorkflowState(run_id=request.run_id, document_path=request.document_path, document_id=document_id))
        return state.model_dump(mode="json")

    @api.get("/health")
    def health():
        return {"status": "ok"}

    return api
