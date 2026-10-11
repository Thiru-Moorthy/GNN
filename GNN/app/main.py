from __future__ import annotations

from pathlib import Path


def main() -> None:
    try:
        import streamlit as st
    except Exception as exc:
        raise RuntimeError("Streamlit UI requires streamlit. Install requirements.txt.") from exc
    from app.orchestration.state import WorkflowState
    from app.orchestration.workflow import run_local_workflow

    st.set_page_config(page_title="P&ID GNN Connectivity POC", layout="wide")
    st.title("P&ID GNN Connectivity POC")
    uploaded = st.file_uploader("Upload PDF, PNG, or JPEG", type=["pdf", "png", "jpg", "jpeg"])
    if uploaded:
        upload_dir = Path("data/uploads")
        upload_dir.mkdir(parents=True, exist_ok=True)
        path = upload_dir / uploaded.name
        path.write_bytes(uploaded.getvalue())
        if st.button("Run POC"):
            state = run_local_workflow(WorkflowState(run_id="streamlit-run", document_path=str(path)))
            st.json(state.model_dump(mode="json", exclude={"data"}))
            for label, artifact in state.artifact_refs.items():
                if Path(artifact).exists():
                    st.download_button(label=f"Download {label}", data=Path(artifact).read_bytes(), file_name=Path(artifact).name)


if __name__ == "__main__":
    main()
