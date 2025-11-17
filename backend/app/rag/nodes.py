from typing import List, Dict, Any

from ..config import settings
from ..inference.router import get_backend
from .vectorstore import HybridVectorStore


async def planner_node(state: Dict[str, Any]) -> Dict[str, Any]:
    '''Planner: interpret query and decide retrieval strategy.'''
    return state


async def retriever_node(state: Dict[str, Any]) -> Dict[str, Any]:
    '''Retriever: queries the vector DB using K_RETRIEVER, K_VEC, K_LEX.'''
    store = HybridVectorStore()
    hits = await store.retrieve(
        query=state["query"],
        cross_doc=state.get("cross_doc", True),
        selected_doc_ids=state.get("selected_doc_ids") or [],
        uploaded_doc_ids=state.get("uploaded_doc_ids") or [],
    )
    state["retrievals"] = hits
    return state


async def compressor_node(state: Dict[str, Any]) -> Dict[str, Any]:
    '''Compressor: merges and deduplicates retrieved chunks.'''
    state["compressed_context"] = state.get("retrievals", [])
    return state


async def critic_node(state: Dict[str, Any]) -> Dict[str, Any]:
    '''Critic: decides whether we need another retrieval round.'''
    state["needs_refine"] = False
    return state


async def refine_retrieve_node(state: Dict[str, Any]) -> Dict[str, Any]:
    '''Refine retrieve: second retrieval step if needed.'''
    return await retriever_node(state)


async def synthesizer_node(state: Dict[str, Any]) -> Dict[str, Any]:
    '''Synthesizer: calls LLM backend with compressed context + query.'''
    backend = get_backend()
    system_prompt = "You are a helpful RAG assistant. Use the provided context to answer."

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": state["query"]},
    ]
    body = type("Body", (), {})()
    setattr(body, "model_dump", lambda: {
        "model": None,
        "messages": messages,
        "max_tokens": 512,
        "temperature": 0.2,
    })
    result = await backend.chat_completion(body)
    state["llm_raw"] = result
    return state


async def citation_pruner_node(state: Dict[str, Any]) -> Dict[str, Any]:
    '''Citation pruner: attach citations to final answer.'''
    answer_text = ""
    llm_raw = state.get("llm_raw") or {}
    try:
        answer_text = llm_raw["choices"][0]["message"]["content"]
    except Exception:
        answer_text = "[Error extracting content from LLM response]"

    return {
        "answer": answer_text,
        "citations": state.get("retrievals", []),
        "metadata": {
            "backend": "nemotron",
            "tokens": llm_raw.get("usage", {}),
        },
    }
