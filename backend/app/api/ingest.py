"""
Ingestion API endpoints for document processing with embedding toggle.

Supports uploading documents and generating embeddings with CLIP or Nemotron.
"""
import logging
from fastapi import APIRouter, HTTPException, UploadFile, File
from pydantic import BaseModel
from typing import List, Optional

from ..config import settings
from ..rag.embeddings import embed_text, get_embedding_model_name, get_embedding_dimension, semantic_chunks_text
from ..rag.vectorstore import HybridVectorStore

logger = logging.getLogger(__name__)
router = APIRouter(tags=["ingestion"])


class IngestTextRequest(BaseModel):
    """Request to ingest plain text."""
    doc_id: str
    text: str
    embedding_model: Optional[str] = None  # "clip" or "nemotron" override


class IngestTextResponse(BaseModel):
    """Response from text ingestion."""
    doc_id: str
    chunks_created: int
    embedding_model: str
    embedding_dimension: int
    metadata: dict


@router.post("/ingest/text")
async def ingest_text(request: IngestTextRequest):
    """
    Ingest plain text and generate embeddings.
    
    Supports embedding model override to test CLIP vs Nemotron.
    
    Args:
        doc_id: Unique document ID
        text: Text content to ingest
        embedding_model: Override embedding model ("clip" or "nemotron")
    
    Returns:
        Ingestion result with chunk count and embedding info
    """
    try:
        logger.info(f"Ingesting document: {request.doc_id}")
        logger.info(f"Embedding model: {request.embedding_model or settings.embedding_model}")
        
        # Get or override embedding model
        embedding_model = request.embedding_model or settings.embedding_model
        
        # Chunk the text
        chunks = semantic_chunks_text(request.text)
        logger.info(f"Created {len(chunks)} chunks")
        
        # Create vector store
        store = HybridVectorStore(embedding_model=embedding_model)
        
        # Prepare chunks for ingestion
        chunk_dicts = []
        for i, (chunk_text, p0, p1, is_ocr, is_figure) in enumerate(chunks):
            chunk_dict = {
                'text': chunk_text,
                'chunk_id': f"{request.doc_id}_chunk_{i}",
                'page_start': p0,
                'page_end': p1,
                'is_ocr': is_ocr,
                'is_figure': is_figure,
            }
            chunk_dicts.append(chunk_dict)
        
        # Ingest all chunks
        ingested_count = await store.ingest_chunks_batch(chunk_dicts, request.doc_id)
        
        logger.info(f"Successfully ingested {ingested_count}/{len(chunk_dicts)} chunks")
        
        return IngestTextResponse(
            doc_id=request.doc_id,
            chunks_created=ingested_count,
            embedding_model=get_embedding_model_name(),
            embedding_dimension=get_embedding_dimension(),
            metadata={
                "total_chunks": len(chunks),
                "embedding_model": embedding_model,
                "text_length": len(request.text),
            }
        )
    
    except Exception as e:
        logger.error(f"Error ingesting text: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


class EmbeddingTestRequest(BaseModel):
    """Request to test embedding generation."""
    text: str
    embedding_model: Optional[str] = None


class EmbeddingTestResponse(BaseModel):
    """Response from embedding test."""
    embedding_model: str
    embedding_dimension: int
    text_length: int
    embedding_norm: float
    sample_values: List[float]  # First 5 values


@router.post("/ingest/test-embedding")
async def test_embedding(request: EmbeddingTestRequest):
    """
    Test embedding generation with text.
    
    Useful for verifying embedding models work correctly and comparing models.
    
    Args:
        text: Text to embed
        embedding_model: Override embedding model ("clip" or "nemotron")
    
    Returns:
        Embedding info (dimension, norm, sample values)
    """
    try:
        logger.info(f"Testing embedding with model: {request.embedding_model or settings.embedding_model}")
        
        # Generate embedding
        embedding = embed_text(request.text, normalize_emb=True)
        
        # Calculate norm (should be ~1.0 for normalized)
        embedding_norm = float((embedding ** 2).sum() ** 0.5)
        
        # Get model info
        model_name = get_embedding_model_name()
        dimension = get_embedding_dimension()
        
        logger.info(f"Generated {dimension}-dim embedding (norm: {embedding_norm:.4f})")
        
        return EmbeddingTestResponse(
            embedding_model=model_name,
            embedding_dimension=dimension,
            text_length=len(request.text),
            embedding_norm=embedding_norm,
            sample_values=embedding[:5].tolist()
        )
    
    except Exception as e:
        logger.error(f"Error testing embedding: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/ingest/compare-embeddings")
async def compare_embeddings_test(request: EmbeddingTestRequest):
    """
    Compare embeddings from CLIP and Nemotron models.
    
    Generates embeddings with both models and returns comparison.
    
    Args:
        text: Text to embed
    
    Returns:
        Embeddings from both models and their similarity
    """
    try:
        import numpy as np
        
        logger.info("Testing both embedding models")
        
        # Get CLIP embedding
        from ..rag.embedding_providers import CLIPEmbeddingProvider, NemotronEmbeddingProvider
        
        clip_provider = CLIPEmbeddingProvider()
        clip_embedding = clip_provider.embed_text(request.text)
        
        # Get Nemotron embedding
        nemotron_provider = NemotronEmbeddingProvider()
        nemotron_embedding = nemotron_provider.embed_text(request.text)
        
        # Calculate cosine similarity
        cosine_sim = float(np.dot(clip_embedding, nemotron_embedding))
        
        logger.info(f"CLIP dimension: {clip_provider.get_embedding_dim()}")
        logger.info(f"Nemotron dimension: {nemotron_provider.get_embedding_dim()}")
        logger.info(f"Cosine similarity: {cosine_sim:.4f}")
        
        return {
            "text": request.text[:100] + "..." if len(request.text) > 100 else request.text,
            "clip": {
                "model": clip_provider.get_model_name(),
                "dimension": clip_provider.get_embedding_dim(),
                "norm": float((clip_embedding ** 2).sum() ** 0.5),
                "sample_values": clip_embedding[:5].tolist()
            },
            "nemotron": {
                "model": nemotron_provider.get_model_name(),
                "dimension": nemotron_provider.get_embedding_dim(),
                "norm": float((nemotron_embedding ** 2).sum() ** 0.5),
                "sample_values": nemotron_embedding[:5].tolist()
            },
            "comparison": {
                "cosine_similarity": cosine_sim,
                "clip_norm": float((clip_embedding ** 2).sum() ** 0.5),
                "nemotron_norm": float((nemotron_embedding ** 2).sum() ** 0.5),
                "dimension_ratio": nemotron_provider.get_embedding_dim() / clip_provider.get_embedding_dim()
            }
        }
    
    except Exception as e:
        logger.error(f"Error comparing embeddings: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/ingest/status")
async def ingestion_status():
    """
    Get ingestion system status and embedding configuration.
    
    Returns:
        Current embedding model and configuration info
    """
    try:
        from ..rag.embeddings import get_active_embedding_provider
        
        provider = get_active_embedding_provider()
        
        return {
            "status": "ready",
            "embedding_model": settings.embedding_model,
            "embedding_model_name": provider.get_model_name(),
            "embedding_dimension": provider.get_embedding_dim(),
            "database_url": settings.database_url,
            "chunking_parameters": {
                "max_words": 25,
                "overlap": 12,
            }
        }
    
    except Exception as e:
        logger.error(f"Error getting ingestion status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

