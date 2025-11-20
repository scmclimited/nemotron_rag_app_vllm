"""
LangGraph RAG Pipeline - Mimics deep_rag architecture.

Graph flow:
  planner → retriever → compressor → critic → (refine_retrieve OR synthesizer)
                                              ↓
                                        citation_pruner → END

The critic node routes to either:
  - refine_retrieve: If confidence is low and iterations remaining
  - synthesizer: If confidence is high or max iterations reached
"""
import logging
from typing import Any, Dict

from langgraph.graph import StateGraph, END

from ..config import settings
from .state import GraphState
from .nodes import (
    planner_node,
    retriever_node,
    compressor_node,
    critic_node,
    refine_retrieve_node,
    synthesizer_node,
    citation_pruner_node,
)

logger = logging.getLogger(__name__)


def route_after_critic(state: GraphState) -> str:
    """
    Route from critic to either refine_retrieve or synthesizer.
    
    Decision based on confidence and iteration count.
    """
    action = state.get("action", "synthesize")
    
    if action == "refine":
        logger.debug("Routing: critic → refine_retrieve")
        return "refine_retrieve"
    else:
        logger.debug("Routing: critic → synthesizer")
        return "synthesizer"


async def run_rag_query(
    query: str,
    thread_id: str | None = None,
    cross_doc: bool = True,
    selected_doc_ids: list[str] | None = None,
    uploaded_doc_ids: list[str] | None = None,
    doc_id: str | None = None,
    embedding_model: str | None = None,
) -> Dict[str, Any]:
    """
    Run the RAG pipeline with LangGraph.
    
    Mimics deep_rag's critic/supervisor pipeline with:
      - Planner: Decompose question
      - Retriever: Hybrid search
      - Compressor: Summarize evidence
      - Critic: Evaluate and route
      - Refine/Synthesizer: Answer or refine
      - Citation Pruner: Final output
    
    Args:
        query: User question
        thread_id: Optional thread/conversation ID
        cross_doc: Whether to search across documents
        selected_doc_ids: Specific documents to search
        uploaded_doc_ids: Recently uploaded documents
        doc_id: Single document to scope to
        embedding_model: Override default embedding model ("clip" or "nemotron")
    
    Returns:
        Dictionary with answer, citations, and metadata
    """
    logger.info("=" * 80)
    logger.info("Starting RAG Query Execution")
    logger.info(f"Query: {query}")
    logger.info(f"Embedding model: {embedding_model or settings.embedding_model}")
    logger.info("=" * 80)
    
    # Build graph
    builder = StateGraph(GraphState)
    
    # Add all nodes
    builder.add_node("planner", planner_node)
    builder.add_node("retriever", retriever_node)
    builder.add_node("compressor", compressor_node)
    builder.add_node("critic", critic_node)
    builder.add_node("refine_retrieve", refine_retrieve_node)
    builder.add_node("synthesizer", synthesizer_node)
    builder.add_node("citation_pruner", citation_pruner_node)
    
    # Linear flow: planner → retriever → compressor → critic
    builder.set_entry_point("planner")
    builder.add_edge("planner", "retriever")
    builder.add_edge("retriever", "compressor")
    builder.add_edge("compressor", "critic")
    
    # Conditional routing from critic
    builder.add_conditional_edges(
        "critic",
        route_after_critic,
        {
            "refine_retrieve": "refine_retrieve",
            "synthesizer": "synthesizer",
        }
    )
    
    # After refine, loop back to compressor
    builder.add_edge("refine_retrieve", "compressor")
    
    # From synthesizer to citation pruner
    builder.add_edge("synthesizer", "citation_pruner")
    
    # End at citation pruner
    builder.add_edge("citation_pruner", END)
    
    # Compile graph
    graph = builder.compile()
    
    # Initialize state
    initial_state: GraphState = {
        "query": query,
        "thread_id": thread_id,
        "cross_doc": cross_doc,
        "selected_doc_ids": selected_doc_ids or [],
        "uploaded_doc_ids": uploaded_doc_ids or [],
        "doc_id": doc_id,
        "iterations": 0,
        "refinements": [],
        "evidence": [],
        "plan": "",
        "notes": "",
        "answer": "",
        "confidence": 0.0,
        "action": "synthesize",
        "embedding_model": embedding_model or settings.embedding_model,
        "doc_ids": [],
    }
    
    # Execute graph
    try:
        result = await graph.ainvoke(initial_state)
        logger.info("=" * 80)
        logger.info("RAG Query Execution Complete")
        logger.info(f"Answer: {result.get('answer', 'N/A')[:100]}...")
        logger.info(f"Citations: {len(result.get('citations', []))} chunks")
        logger.info("=" * 80)
        return result
    except Exception as e:
        logger.error(f"Error executing RAG graph: {e}", exc_info=True)
        return {
            "answer": "I don't know.",
            "citations": [],
            "error": str(e),
            "metadata": {"embedding_model": embedding_model or settings.embedding_model},
        }
