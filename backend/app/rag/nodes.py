"""
LangGraph RAG pipeline nodes - Mimics deep_rag architecture.

Nodes:
  1. planner: Decompose question into sub-goals
  2. retriever: Fetch relevant chunks from vector DB
  3. compressor: Summarize evidence into notes
  4. critic: Evaluate evidence quality and decide refinement
  5. refine_retrieve: Execute refined queries if needed
  6. synthesizer: Generate final answer from evidence
  7. citation_pruner: Attach citations to answer

Supports per-node model configuration via environment variables:
  - {NODE_NAME}_MODEL: Override LLM model
  - {NODE_NAME}_QUANTIZATION: fp8, bf16, or none
  - {NODE_NAME}_DTYPE: float32, float16, bfloat16
"""
import logging
import re
from typing import Any, Dict, List, Optional

from ..config import settings
from ..inference.router import get_backend
from .state import GraphState
from .prompts import format_template
from .vectorstore import HybridVectorStore
from .node_models import get_node_config

logger = logging.getLogger(__name__)


def _get_node_llm_config(node_name: str) -> Dict[str, Any]:
    """
    Get LLM configuration for a specific node.
    
    Returns a dict with:
    - model: Override model (if configured)
    - quantization: Quantization level (fp8, bf16, none)
    - dtype: Data type (if configured)
    
    Falls back to global config if node-specific config not provided.
    """
    config = get_node_config(node_name)
    return {
        "model": config.get_model(),
        "quantization": config.get_quantization(),
        "dtype": config.get_dtype(),
        "is_quantized": config.is_quantized(),
    }


async def planner_node(state: GraphState) -> GraphState:
    """
    Planner: Decompose the question into sub-goals.
    
    Generates a plan that guides retrieval and helps the critic evaluate evidence.
    Supports per-node model configuration via environment variables.
    """
    logger.info("=" * 80)
    logger.info("GRAPH NODE: Planner - Decomposing question into sub-goals")
    
    # Get node-specific configuration
    node_config = _get_node_llm_config("planner")
    if node_config["model"] or node_config["quantization"] != "none":
        logger.info(f"Node configuration: {node_config}")
    logger.info("=" * 80)
    
    query = state.get("query", "")
    iterations = state.get("iterations", 0)
    cross_doc = state.get("cross_doc", False)
    selected_doc_ids = state.get("selected_doc_ids")
    doc_id = state.get("doc_id")
    
    logger.info(f"Query: {query}")
    logger.info(f"Iterations: {iterations}")
    logger.info(f"Cross-doc: {cross_doc}")
    if selected_doc_ids and len(selected_doc_ids) > 0:
        logger.info(f"Planning for {len(selected_doc_ids)} selected document(s)")
    elif doc_id:
        logger.info(f"Planning for specific document: {doc_id[:8]}...")
    
    # Include doc_id context in prompt if available
    doc_context = ""
    if doc_id:
        doc_context = (
            f"\n\nNote: This question is about a specific document. "
            f"Focus your planning on this document's content."
        )
    
    # Call LLM for planning
    prompt = format_template("planner", question=query, doc_context=doc_context)
    
    backend = get_backend()
    messages = [{"role": "user", "content": prompt}]
    
    try:
        # Get backend response (vLLM or Ollama)
        body = type("Body", (), {})()
        setattr(body, "model_dump", lambda: {
            "messages": messages,
            "max_tokens": 200,
            "temperature": 0.2,
        })
        result = await backend.chat_completion(body)
        plan = result.get("choices", [{}])[0].get("message", {}).get("content", "")
    except Exception as e:
        logger.warning(f"Error calling LLM for planning: {e}")
        plan = query  # Fallback: use query as plan
    
    plan_text = plan.strip()
    logger.info(f"Generated Plan: {plan_text}")
    logger.info("-" * 40)
    
    return {
        "plan": plan_text,
        "iterations": iterations,
    }


async def retriever_node(state: GraphState) -> GraphState:
    """
    Retriever: Fetch relevant chunks from vector database.
    
    Uses hybrid retrieval (lexical + vector) with document filtering.
    """
    logger.info("-" * 40)
    logger.info("GRAPH NODE: Retriever - Fetching relevant chunks")
    logger.info("-" * 40)
    
    query = state.get("query", "")
    plan = state.get("plan", "")
    iterations = state.get("iterations", 0)
    cross_doc = state.get("cross_doc", False)
    selected_doc_ids = state.get("selected_doc_ids")
    uploaded_doc_ids = state.get("uploaded_doc_ids")
    doc_id = state.get("doc_id")
    embedding_model = state.get("embedding_model", settings.embedding_model)
    
    logger.info(f"Iterations: {iterations}, Cross-doc: {cross_doc}")
    logger.info(f"Embedding model: {embedding_model}")
    
    # Combine query with plan for better retrieval
    combined_query = f"{query} {plan}".strip()
    
    # Determine which doc_ids to use for filtering
    doc_ids_to_filter = None
    
    if selected_doc_ids and len(selected_doc_ids) > 0:
        doc_ids_to_filter = list(selected_doc_ids)
    
    if uploaded_doc_ids and len(uploaded_doc_ids) > 0:
        if doc_ids_to_filter is None:
            doc_ids_to_filter = []
        for uploaded_id in uploaded_doc_ids:
            if uploaded_id not in doc_ids_to_filter:
                doc_ids_to_filter.append(uploaded_id)
    
    if doc_id and (doc_ids_to_filter is None or doc_id not in doc_ids_to_filter):
        if doc_ids_to_filter is None:
            doc_ids_to_filter = [doc_id]
        else:
            doc_ids_to_filter.append(doc_id)
    
    # Perform retrieval using specified embedding model
    logger.info(f"Creating vector store with embedding model: {embedding_model}")
    store = HybridVectorStore(embedding_model=embedding_model)
    
    try:
        hits = await store.retrieve(
            query=combined_query,
            cross_doc=cross_doc,
            selected_doc_ids=doc_ids_to_filter or [],
            uploaded_doc_ids=[],
            k_retriever=settings.k_retriever,
            k_vec=settings.k_vec,
            k_lex=settings.k_lex,
        )
    except Exception as e:
        logger.error(f"Error during retrieval: {e}")
        hits = []
    
    logger.info(f"Retrieved {len(hits)} chunks")
    if hits:
        for i, hit in enumerate(hits[:3]):
            logger.info(f"  {i+1}. Score: {hit.get('score', 'N/A')}, Doc: {hit.get('doc_id', 'unknown')[:8]}")
    
    return {
        "evidence": hits,
        "doc_ids": list(set([h.get("doc_id") for h in hits if h.get("doc_id")])),
    }


async def compressor_node(state: GraphState) -> GraphState:
    """
    Compressor: Summarize retrieved evidence into concise notes.
    
    Reduces verbosity of chunks while preserving key facts, numbers, and proper nouns.
    """
    logger.info("=" * 80)
    logger.info("GRAPH NODE: Compressor - Summarizing evidence")
    logger.info("=" * 80)
    
    evidence = state.get("evidence", [])
    iterations = state.get("iterations", 0)
    
    logger.info(f"Compressing {len(evidence)} chunks into notes...")
    logger.info(f"Iterations: {iterations}")
    
    if not evidence:
        logger.warning("No evidence to compress")
        return {
            "notes": "",
        }
    
    # Build snippets from evidence
    snippets = "\n\n".join([
        f"[p{h.get('p0', '?')}-{h.get('p1', '?')}] {h.get('text', '')[:1200]}"
        for h in evidence
    ])
    
    # Call LLM for compression
    prompt = format_template("compressor", snippets=snippets)
    
    backend = get_backend()
    messages = [{"role": "user", "content": prompt}]
    
    try:
        body = type("Body", (), {})()
        setattr(body, "model_dump", lambda: {
            "messages": messages,
            "max_tokens": 400,
            "temperature": 0.1,
        })
        result = await backend.chat_completion(body)
        notes = result.get("choices", [{}])[0].get("message", {}).get("content", "")
    except Exception as e:
        logger.warning(f"Error calling LLM for compression: {e}")
        notes = snippets  # Fallback: use raw snippets
    
    notes_text = notes.strip()
    logger.info(f"Compressed notes ({len(notes_text)} chars)")
    logger.info("-" * 80)
    
    return {
        "notes": notes_text,
    }


async def critic_node(state: GraphState) -> GraphState:
    """
    Critic: Evaluate evidence quality and decide next action.
    
    Routes to either:
      - "refine": Request refined retrieval if confidence is low
      - "synthesize": Proceed to answer generation if confidence is sufficient
    """
    logger.info("-" * 40)
    logger.info("GRAPH NODE: Critic - Evaluating evidence quality")
    logger.info("-" * 40)
    
    evidence = state.get("evidence", [])
    plan = state.get("plan", "")
    notes = state.get("notes", "")
    iterations = state.get("iterations", 0)
    max_iters = settings.max_iters
    
    logger.info(f"Evidence chunks: {len(evidence)}")
    logger.info(f"Iterations: {iterations}/{max_iters}")
    
    # Simple confidence heuristic: based on number and quality of evidence
    # Mimics deep_rag's confidence calculation
    strong_chunks = sum(
        1 for h in evidence
        if float(h.get("score", 0.0)) > settings.confidence_threshold
    )
    
    confidence = min(0.9, 0.4 + 0.1 * strong_chunks)
    
    logger.info(f"Strong chunks: {strong_chunks}/{len(evidence)}")
    logger.info(f"Confidence score: {confidence:.2f}")
    
    # Decision logic
    action = "synthesize"  # Default: proceed to answer
    
    if confidence < 0.6 and iterations < max_iters:
        logger.info(
            f"Confidence {confidence:.2f} < 0.6 and iterations {iterations} < {max_iters} "
            f"→ Requesting refinement..."
        )
        action = "refine"
    elif iterations >= max_iters:
        logger.info(
            f"Max iterations {max_iters} reached → Synthesizing answer with current evidence"
        )
        action = "synthesize"
    
    logger.info(f"Critic decision: {action}")
    logger.info("-" * 40)
    
    return {
        "action": action,
        "confidence": confidence,
        "iterations": iterations,
    }


async def refine_retrieve_node(state: GraphState) -> GraphState:
    """
    Refine Retrieve: Generate sub-queries and execute refined retrieval.
    
    Called when critic decides evidence quality is insufficient.
    Uses LLM to generate better sub-queries based on initial plan and evidence gaps.
    """
    logger.info("=" * 80)
    logger.info("GRAPH NODE: Refine Retrieve - Generating refined queries")
    logger.info("=" * 80)
    
    plan = state.get("plan", "")
    notes = state.get("notes", "")
    iterations = state.get("iterations", 0)
    evidence = state.get("evidence", [])
    refinements = state.get("refinements", [])
    embedding_model = state.get("embedding_model", settings.embedding_model)
    
    logger.info(f"Refinement iteration: {iterations + 1}")
    
    # Call LLM to generate refined queries
    prompt = format_template(
        "critic_standard",
        plan=plan,
        notes=notes
    )
    
    backend = get_backend()
    messages = [{"role": "user", "content": prompt}]
    
    try:
        body = type("Body", (), {})()
        setattr(body, "model_dump", lambda: {
            "messages": messages,
            "max_tokens": 150,
            "temperature": 0.3,
        })
        result = await backend.chat_completion(body)
        refined_queries_raw = result.get("choices", [{}])[0].get("message", {}).get("content", "")
    except Exception as e:
        logger.warning(f"Error generating refined queries: {e}")
        refined_queries_raw = ""
    
    # Parse refined queries from LLM output
    # Expected format: numbered or bulleted list of queries
    refined_queries = [
        q.strip().lstrip("- •*").strip()
        for q in refined_queries_raw.split("\n")
        if q.strip() and not q.strip().startswith(("Summary", "Note"))
    ]
    refined_queries = refined_queries[:2]  # Max 2 refined queries
    
    logger.info(f"Generated {len(refined_queries)} refined queries:")
    for i, q in enumerate(refined_queries):
        logger.info(f"  {i+1}. {q}")
    
    # Re-retrieve with refined queries
    store = HybridVectorStore(embedding_model=embedding_model)
    new_evidence = []
    
    selected_doc_ids = state.get("selected_doc_ids")
    uploaded_doc_ids = state.get("uploaded_doc_ids")
    doc_id = state.get("doc_id")
    
    doc_ids_to_filter = None
    if selected_doc_ids:
        doc_ids_to_filter = list(selected_doc_ids)
    if uploaded_doc_ids:
        if doc_ids_to_filter is None:
            doc_ids_to_filter = []
        for uid in uploaded_doc_ids:
            if uid not in doc_ids_to_filter:
                doc_ids_to_filter.append(uid)
    if doc_id:
        if doc_ids_to_filter is None:
            doc_ids_to_filter = [doc_id]
        elif doc_id not in doc_ids_to_filter:
            doc_ids_to_filter.append(doc_id)
    
    for refined_query in refined_queries:
        try:
            hits = await store.retrieve(
                query=refined_query,
                cross_doc=state.get("cross_doc", False),
                selected_doc_ids=doc_ids_to_filter or [],
                uploaded_doc_ids=[],
                k_retriever=settings.k_retriever,
                k_vec=settings.k_vec,
                k_lex=settings.k_lex,
            )
            new_evidence.extend(hits)
        except Exception as e:
            logger.warning(f"Error retrieving with refined query '{refined_query}': {e}")
    
    # Deduplicate and merge with existing evidence
    existing_ids = {h.get("chunk_id") for h in evidence}
    merged_evidence = list(evidence)
    for hit in new_evidence:
        if hit.get("chunk_id") not in existing_ids:
            merged_evidence.append(hit)
            existing_ids.add(hit.get("chunk_id"))
    
    logger.info(f"Merged evidence: {len(evidence)} → {len(merged_evidence)} chunks")
    logger.info("-" * 80)
    
    return {
        "evidence": merged_evidence,
        "refinements": refinements + refined_queries,
        "iterations": iterations + 1,
    }


async def synthesizer_node(state: GraphState) -> GraphState:
    """
    Synthesizer: Generate final answer from evidence using LLM.
    
    Builds context from compressed notes and queries the LLM with citations.
    """
    logger.info("=" * 80)
    logger.info("GRAPH NODE: Synthesizer - Generating final answer")
    logger.info("=" * 80)
    
    query = state.get("query", "")
    notes = state.get("notes", "")
    evidence = state.get("evidence", [])
    confidence = state.get("confidence", 0.5)
    
    logger.info(f"Query: {query}")
    logger.info(f"Confidence: {confidence:.2f}")
    logger.info(f"Evidence chunks: {len(evidence)}")
    
    # If confidence below threshold, abstain early
    threshold = settings.synthesizer_conf_threshold_explicit_selection
    if confidence < threshold:
        logger.info(f"Confidence {confidence:.2f} < threshold {threshold} → Abstaining")
        return {
            "answer": "I don't know.",
            "llm_raw": {"choices": [{"message": {"content": "I don't know."}}]},
            "confidence": confidence,
        }
    
    # Build context from evidence with citations
    doc_reference_list = "Available Chunks:\n"
    context_lines = []
    citation_map = {}
    
    for idx, chunk in enumerate(evidence[:24]):  # Limit to 24 chunks
        letter = chr(65 + idx)  # A, B, C, ...
        citation_map[idx] = letter
        doc_id = chunk.get("doc_id", "unknown")
        text = chunk.get("text", "")
        doc_reference_list += f"[{letter}] DOC: {doc_id[:8]} - {text[:100]}...\n"
        context_lines.append(text)
    
    context = "\n\n".join(context_lines)
    
    # Format synthesizer prompt
    prompt = format_template(
        "synthesizer_standard",
        doc_reference_list=doc_reference_list,
        question_text=query,
        question_lower=query.lower(),
        num_documents=len(set(h.get("doc_id") for h in evidence)),
        context=context,
        citation_format="[A] DOC: doc_id1\n[B] DOC: doc_id2",
        order_block="",
    )
    
    backend = get_backend()
    messages = [{"role": "user", "content": prompt}]
    
    try:
        body = type("Body", (), {})()
        setattr(body, "model_dump", lambda: {
            "messages": messages,
            "max_tokens": 512,
            "temperature": 0.2,
        })
        result = await backend.chat_completion(body)
        answer = result.get("choices", [{}])[0].get("message", {}).get("content", "I don't know.")
    except Exception as e:
        logger.error(f"Error calling LLM for synthesis: {e}")
        result = {"choices": [{"message": {"content": "I don't know."}}]}
        answer = "I don't know."
    
    answer_text = answer.strip()
    
    # Check if LLM abstained
    if "i don't know" in answer_text.lower():
        logger.info("LLM returned abstention")
        answer_text = "I don't know."
    
    logger.info(f"Generated answer ({len(answer_text)} chars)")
    logger.info("-" * 80)
    
    return {
        "answer": answer_text,
        "llm_raw": result,
        "confidence": confidence,
    }


async def citation_pruner_node(state: GraphState) -> GraphState:
    """
    Citation Pruner: Final processing of answer and citations.
    
    Validates citations and prepares final output.
    """
    logger.info("=" * 80)
    logger.info("GRAPH NODE: Citation Pruner - Processing citations")
    logger.info("=" * 80)
    
    answer = state.get("answer", "I don't know.")
    evidence = state.get("evidence", [])
    llm_raw = state.get("llm_raw", {})
    
    # If abstained, clear citations
    if "i don't know" in answer.lower():
        logger.info("Clearing citations due to abstention")
        citations = []
    else:
        # Use evidence as citations
        citations = evidence
    
    logger.info(f"Final answer: {answer[:100]}...")
    logger.info(f"Citations: {len(citations)} chunks")
    logger.info("=" * 80)
    
    return {
        "answer": answer,
        "citations": citations,
        "metadata": {
            "backend": "nemotron",
            "tokens": llm_raw.get("usage", {}),
            "embedding_model": state.get("embedding_model", settings.embedding_model),
        },
    }
