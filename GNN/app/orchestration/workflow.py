from __future__ import annotations

from collections.abc import Callable

from app.orchestration import nodes
from app.orchestration.state import WorkflowState

WORKFLOW_NODE_SEQUENCE: list[Callable[[WorkflowState], WorkflowState]] = [
    nodes.ingest_document,
    nodes.normalize_pages,
    nodes.detect_components,
    nodes.extract_text_and_lines,
    nodes.build_candidate_graph_node,
    nodes.run_baseline,
    nodes.run_gnn,
    nodes.validate_topology,
    nodes.route_uncertain_edges,
    nodes.publish_knowledge_graph,
    nodes.evaluate_connectivity,
    nodes.generate_outputs,
]


def run_local_workflow(initial_state: WorkflowState) -> WorkflowState:
    state = initial_state.model_copy(update={"status": "running"})
    try:
        for node in WORKFLOW_NODE_SEQUENCE:
            state = node(state)
    except Exception as exc:
        state.status = "failed"
        state.errors.append(f"{type(exc).__name__}: {exc}")
    return state


def build_langgraph_workflow():
    try:
        from langgraph.graph import END, StateGraph  # type: ignore
    except Exception as exc:
        raise RuntimeError("LangGraph is optional. Install langgraph or use run_local_workflow.") from exc

    graph = StateGraph(WorkflowState)
    previous = None
    for node_fn in WORKFLOW_NODE_SEQUENCE:
        graph.add_node(node_fn.__name__, node_fn)
        if previous is not None:
            graph.add_edge(previous, node_fn.__name__)
        previous = node_fn.__name__
    graph.set_entry_point(WORKFLOW_NODE_SEQUENCE[0].__name__)
    graph.add_edge(WORKFLOW_NODE_SEQUENCE[-1].__name__, END)
    return graph.compile()
