from typing import Any, Dict

from langgraph.graph import StateGraph, END

from .nodes import (
    planner_node,
    retriever_node,
    compressor_node,
    critic_node,
    refine_retrieve_node,
    synthesizer_node,
    citation_pruner_node,
)


async def run_rag_query(
    query: str,
    thread_id: str | None,
    cross_doc: bool,
    selected_doc_ids: list[str],
    uploaded_doc_ids: list[str],
) -> Dict[str, Any]:
    '''Run a simple RAG graph similar to deep_rag.'''

    builder = StateGraph(dict)

    builder.add_node("planner", planner_node)
    builder.add_node("retriever", retriever_node)
    builder.add_node("compressor", compressor_node)
    builder.add_node("critic", critic_node)
    builder.add_node("refine_retrieve", refine_retrieve_node)
    builder.add_node("synthesizer", synthesizer_node)
    builder.add_node("citation_pruner", citation_pruner_node)

    builder.set_entry_point("planner")
    builder.add_edge("planner", "retriever")
    builder.add_edge("retriever", "compressor")
    builder.add_edge("compressor", "critic")
    builder.add_edge("critic", "synthesizer")
    builder.add_edge("synthesizer", "citation_pruner")
    builder.add_edge("citation_pruner", END)

    graph = builder.compile()

    initial_state: Dict[str, Any] = {
        "thread_id": thread_id,
        "query": query,
        "cross_doc": cross_doc,
        "selected_doc_ids": selected_doc_ids,
        "uploaded_doc_ids": uploaded_doc_ids,
    }

    result = await graph.ainvoke(initial_state)
    return result
