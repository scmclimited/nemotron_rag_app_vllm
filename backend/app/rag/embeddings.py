"""
Embedding utilities for ingestion pipeline.

Supports toggling between CLIP and Nemotron embeddings via environment variable.
Provides text, batch, and utility functions for document ingestion.
"""
import logging
import numpy as np
from typing import List, Tuple
from ..config import settings
from .embedding_providers import get_embedding_provider, EmbeddingProvider

logger = logging.getLogger(__name__)

# Global provider instance (cached)
_provider: EmbeddingProvider | None = None


def get_active_embedding_provider() -> EmbeddingProvider:
    """
    Get the active embedding provider.
    
    Caches globally to avoid reloading models on every call.
    """
    global _provider
    if _provider is None:
        _provider = get_embedding_provider()
        logger.info(f"Initialized embedding provider: {_provider.get_model_name()}")
    return _provider


def embed_text(text: str, normalize_emb: bool = True) -> np.ndarray:
    """
    Embed a single text chunk.
    
    Args:
        text: Text to embed
        normalize_emb: Whether to normalize the embedding
    
    Returns:
        Normalized embedding vector
    """
    provider = get_active_embedding_provider()
    
    try:
        embedding = provider.embed_text(text)
        
        if normalize_emb and embedding is not None:
            embedding = normalize(embedding)
        
        return embedding
    
    except Exception as e:
        logger.error(f"Error embedding text: {e}")
        raise


def embed_batch(texts: List[str], normalize_emb: bool = True) -> np.ndarray:
    """
    Embed a batch of texts.
    
    Args:
        texts: List of texts to embed
        normalize_emb: Whether to normalize embeddings
    
    Returns:
        Array of embedding vectors (num_texts, embedding_dim)
    """
    if not texts:
        return np.array([])
    
    provider = get_active_embedding_provider()
    
    try:
        embeddings = provider.embed_batch(texts)
        
        if normalize_emb and embeddings is not None:
            embeddings = np.array([normalize(emb) for emb in embeddings])
        
        return embeddings
    
    except Exception as e:
        logger.error(f"Error embedding batch: {e}")
        raise


def normalize(v: np.ndarray) -> np.ndarray:
    """
    L2 normalize a vector.
    
    Args:
        v: Vector to normalize
    
    Returns:
        Normalized vector
    """
    norm = np.linalg.norm(v)
    if norm < 1e-8:
        return v
    return v / norm


def get_embedding_dimension() -> int:
    """
    Get the dimension of embeddings from the active provider.
    
    Returns:
        Embedding dimension (768 for CLIP, 1024 for Nemotron)
    """
    provider = get_active_embedding_provider()
    return provider.get_embedding_dim()


def get_embedding_model_name() -> str:
    """
    Get the name of the active embedding model.
    
    Returns:
        Model name/ID
    """
    provider = get_active_embedding_provider()
    return provider.get_model_name()


def semantic_chunks_text(text: str, max_words: int = 25, overlap: int = 12) -> List[Tuple[str, int, int, bool, bool]]:
    """
    Split text into semantic chunks with overlap.
    
    Similar to PDF chunking but for plain text.
    Respects CLIP's 77-token limit (or Nemotron's higher limit).
    
    Args:
        text: Text to chunk
        max_words: Maximum words per chunk
        overlap: Overlap words between chunks
    
    Returns:
        List of (chunk_text, page_start, page_end, is_ocr, is_figure) tuples
    """
    import re
    chunks = []
    
    # Split by paragraphs first
    paragraphs = re.split(r'\n\s*\n', text)
    buf, count = [], 0
    
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        
        toks = len(para.split())  # Word count
        if count + toks > max_words and buf:
            chunk_text = " ".join(buf)
            chunks.append((chunk_text, 1, 1, False, False))
            
            # Calculate overlap
            overlap_words = " ".join(chunk_text.split()[-overlap:])
            buf, count = [overlap_words, para], len(overlap_words.split()) + toks
        else:
            buf.append(para)
            count += toks
    
    if buf:
        chunk_text = " ".join(buf)
        chunks.append((chunk_text, 1, 1, False, False))
    
    return chunks


def log_embedding_info():
    """Log current embedding configuration."""
    try:
        provider = get_active_embedding_provider()
        logger.info(f"Active embedding model: {provider.get_model_name()}")
        logger.info(f"Embedding dimension: {provider.get_embedding_dim()}")
    except Exception as e:
        logger.warning(f"Could not log embedding info: {e}")
