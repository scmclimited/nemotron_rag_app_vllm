"""
RAG API endpoints with embedding toggle and comparison testing.
"""
import logging
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict, Any

from ..rag.graph import run_rag_query

logger = logging.getLogger(__name__)
router = APIRouter(tags=["rag"])


class RagQuery(BaseModel):
    """RAG query request."""
    thread_id: Optional[str] = None
    query: str
    cross_doc: bool = True
    selected_doc_ids: List[str] = []
    uploaded_doc_ids: List[str] = []
    doc_id: Optional[str] = None
    embedding_model: Optional[str] = None  # "clip" or "nemotron"


class RagComparisonRequest(BaseModel):
    """Request for comparing results with different embeddings."""
    query: str
    thread_id: Optional[str] = None
    cross_doc: bool = True
    selected_doc_ids: List[str] = []
    uploaded_doc_ids: List[str] = []
    doc_id: Optional[str] = None


@router.post("/rag/query")
async def rag_query(body: RagQuery):
    """
    Execute RAG query with optional embedding model override.
    
    If embedding_model is provided, it overrides the default from config.
    """
    try:
        logger.info(f"RAG Query: {body.query[:50]}... (embedding: {body.embedding_model or 'default'})")
        result = await run_rag_query(
            query=body.query,
            thread_id=body.thread_id,
            cross_doc=body.cross_doc,
            selected_doc_ids=body.selected_doc_ids,
            uploaded_doc_ids=body.uploaded_doc_ids,
            doc_id=body.doc_id,
            embedding_model=body.embedding_model,
        )
        return result
    except Exception as e:
        logger.error(f"RAG query error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/rag/query/compare")
async def rag_query_compare(body: RagComparisonRequest):
    """
    Compare RAG results using CLIP and Nemotron embeddings side-by-side.
    
    Returns:
        {
            "query": "...",
            "clip": {
                "answer": "...",
                "confidence": 0.8,
                "citations_count": 5,
                "metadata": {...}
            },
            "nemotron": {
                "answer": "...",
                "confidence": 0.75,
                "citations_count": 4,
                "metadata": {...}
            },
            "comparison": {
                "answer_similarity": 0.85,
                "confidence_delta": -0.05,
                "citation_delta": 1
            }
        }
    """
    try:
        logger.info(f"RAG Comparison Query: {body.query[:50]}...")
        
        # Run with CLIP embeddings
        logger.info("Running with CLIP embeddings...")
        clip_result = await run_rag_query(
            query=body.query,
            thread_id=body.thread_id,
            cross_doc=body.cross_doc,
            selected_doc_ids=body.selected_doc_ids,
            uploaded_doc_ids=body.uploaded_doc_ids,
            doc_id=body.doc_id,
            embedding_model="clip",
        )
        
        # Run with Nemotron embeddings
        logger.info("Running with Nemotron embeddings...")
        nemotron_result = await run_rag_query(
            query=body.query,
            thread_id=body.thread_id,
            cross_doc=body.cross_doc,
            selected_doc_ids=body.selected_doc_ids,
            uploaded_doc_ids=body.uploaded_doc_ids,
            doc_id=body.doc_id,
            embedding_model="nemotron",
        )
        
        # Build comparison response
        clip_answer = clip_result.get("answer", "")
        nemotron_answer = nemotron_result.get("answer", "")
        
        # Simple answer similarity (character overlap ratio)
        answer_similarity = _calculate_similarity(clip_answer, nemotron_answer)
        
        clip_confidence = clip_result.get("confidence", 0.0)
        nemotron_confidence = nemotron_result.get("confidence", 0.0)
        confidence_delta = nemotron_confidence - clip_confidence
        
        clip_citations = len(clip_result.get("citations", []))
        nemotron_citations = len(nemotron_result.get("citations", []))
        citation_delta = nemotron_citations - clip_citations
        
        comparison = {
            "query": body.query,
            "clip": {
                "answer": clip_answer,
                "confidence": clip_confidence,
                "citations_count": clip_citations,
                "metadata": clip_result.get("metadata", {}),
            },
            "nemotron": {
                "answer": nemotron_answer,
                "confidence": nemotron_confidence,
                "citations_count": nemotron_citations,
                "metadata": nemotron_result.get("metadata", {}),
            },
            "deltas": {
                "answer_similarity": answer_similarity,
                "confidence_delta": confidence_delta,
                "citation_delta": citation_delta,
                "note": "Positive delta = Nemotron advantage, Negative = CLIP advantage",
            },
        }
        
        logger.info(f"Comparison complete. Answer similarity: {answer_similarity:.2f}")
        return comparison
        
    except Exception as e:
        logger.error(f"RAG comparison error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


def _calculate_similarity(text1: str, text2: str) -> float:
    """
    Calculate simple character-level similarity between two texts.
    
    Range: 0.0 (completely different) to 1.0 (identical)
    """
    if not text1 or not text2:
        return 0.0 if text1 != text2 else 1.0
    
    # Convert to lowercase for comparison
    text1_lower = text1.lower()
    text2_lower = text2.lower()
    
    # Simple overlap-based similarity
    matches = sum(1 for c1, c2 in zip(text1_lower, text2_lower) if c1 == c2)
    max_len = max(len(text1), len(text2))
    
    return matches / max_len if max_len > 0 else 0.0
