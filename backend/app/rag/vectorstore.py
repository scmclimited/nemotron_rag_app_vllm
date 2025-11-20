"""
Hybrid vector store abstraction over pgvector.

Supports both lexical and vector-based retrieval using configurable embedding models.
"""
import logging
import numpy as np
from typing import List, Dict, Any, Optional
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError

from ..config import settings
from .embeddings import embed_text, get_embedding_model_name

logger = logging.getLogger(__name__)


class HybridVectorStore:
    """
    Hybrid vector store supporting both lexical and vector-based retrieval.
    
    Uses pgvector for vector similarity and PostgreSQL full-text search for lexical matching.
    Supports embedding model toggle (CLIP/Nemotron).
    """
    
    def __init__(self, embedding_model: Optional[str] = None):
        """
        Initialize vector store with optional embedding model override.
        
        Args:
            embedding_model: Override embedding model ("clip" or "nemotron")
        """
        self.embedding_model = embedding_model
        self.connection_string = settings.database_url
        
        # Will be lazy-initialized
        self.engine = None
    
    async def retrieve(
        self,
        query: str,
        cross_doc: bool = True,
        selected_doc_ids: Optional[List[str]] = None,
        uploaded_doc_ids: Optional[List[str]] = None,
        k_retriever: int = 8,
        k_vec: int = 60,
        k_lex: int = 60,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant chunks using hybrid search.
        
        Args:
            query: Search query
            cross_doc: Whether to search across documents
            selected_doc_ids: Specific document IDs to search
            uploaded_doc_ids: Recently uploaded document IDs
            k_retriever: Final number of chunks to return
            k_vec: Vector search limit
            k_lex: Lexical search limit
        
        Returns:
            List of relevant chunks with scores
        """
        try:
            logger.info(f"Retrieving with query: {query[:50]}...")
            logger.info(f"Embedding model: {self.embedding_model or get_embedding_model_name()}")
            
            # Generate query embedding
            try:
                query_embedding = embed_text(query, normalize_emb=True)
                query_embedding_str = str(query_embedding.tolist())
            except Exception as e:
                logger.error(f"Error generating query embedding: {e}")
                # Fallback: return empty if embedding fails
                return []
            
            # Determine document filter
            doc_filter = self._build_doc_filter(
                cross_doc,
                selected_doc_ids,
                uploaded_doc_ids
            )
            
            # Vector search
            vector_results = await self._vector_search(
                query_embedding_str,
                k_vec,
                doc_filter
            )
            
            # Lexical search (BM25-style using full-text search)
            lexical_results = await self._lexical_search(
                query,
                k_lex,
                doc_filter
            )
            
            # Merge and rank results
            merged = self._merge_results(vector_results, lexical_results)
            
            # Return top k_retriever
            final_results = merged[:k_retriever]
            
            logger.info(f"Retrieved {len(final_results)} chunks")
            return final_results
        
        except Exception as e:
            logger.error(f"Error during hybrid retrieval: {e}", exc_info=True)
            return []
    
    def _build_doc_filter(
        self,
        cross_doc: bool,
        selected_doc_ids: Optional[List[str]],
        uploaded_doc_ids: Optional[List[str]]
    ) -> Optional[str]:
        """
        Build SQL WHERE clause for document filtering.
        
        Returns:
            SQL fragment or None if no filtering
        """
        doc_ids_to_use = []
        
        if selected_doc_ids:
            doc_ids_to_use.extend(selected_doc_ids)
        
        if uploaded_doc_ids:
            doc_ids_to_use.extend(uploaded_doc_ids)
        
        if not cross_doc and doc_ids_to_use:
            # Non-cross-doc: only search in specified documents
            doc_list = "', '".join(doc_ids_to_use)
            return f"doc_id IN ('{doc_list}')"
        elif cross_doc and doc_ids_to_use:
            # Cross-doc: prioritize specified documents but allow others
            # This is handled in the merge/rank function
            return None
        else:
            # No filtering
            return None
    
    async def _vector_search(
        self,
        query_embedding: str,
        limit: int,
        doc_filter: Optional[str]
    ) -> List[Dict[str, Any]]:
        """
        Vector similarity search using pgvector.
        
        Returns:
            List of chunks with vector scores
        """
        try:
            # SQL query for vector search
            sql = f"""
            SELECT 
                chunk_id,
                doc_id,
                text,
                p0,
                p1,
                emb <-> CAST(:embedding AS vector) as distance,
                1.0 / (1.0 + (emb <-> CAST(:embedding AS vector))) as score
            FROM chunks
            """
            
            if doc_filter:
                sql += f" WHERE {doc_filter}"
            
            sql += """
            ORDER BY emb <-> CAST(:embedding AS vector)
            LIMIT :limit
            """
            
            # TODO: Implement actual database query
            # For now, return empty list (placeholder implementation)
            logger.warning("Vector search not yet fully implemented - returning empty results")
            return []
        
        except Exception as e:
            logger.error(f"Error during vector search: {e}")
            return []
    
    async def _lexical_search(
        self,
        query: str,
        limit: int,
        doc_filter: Optional[str]
    ) -> List[Dict[str, Any]]:
        """
        Full-text search using PostgreSQL.
        
        Returns:
            List of chunks with lexical scores
        """
        try:
            # SQL query for lexical search
            sql = f"""
            SELECT 
                chunk_id,
                doc_id,
                text,
                p0,
                p1,
                ts_rank(to_tsvector('english', text), 
                        plainto_tsquery('english', :query)) as score
            FROM chunks
            """
            
            if doc_filter:
                sql += f" WHERE {doc_filter}"
            else:
                sql += " WHERE to_tsvector('english', text) @@ plainto_tsquery('english', :query)"
            
            sql += """
            ORDER BY score DESC
            LIMIT :limit
            """
            
            # TODO: Implement actual database query
            # For now, return empty list (placeholder implementation)
            logger.warning("Lexical search not yet fully implemented - returning empty results")
            return []
        
        except Exception as e:
            logger.error(f"Error during lexical search: {e}")
            return []
    
    def _merge_results(
        self,
        vector_results: List[Dict[str, Any]],
        lexical_results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Merge and rank vector and lexical search results.
        
        Combines scores using RRF (Reciprocal Rank Fusion).
        """
        # Create combined ranking
        scores = {}
        
        # Vector results (60% weight)
        for i, result in enumerate(vector_results):
            chunk_id = result.get('chunk_id')
            if chunk_id not in scores:
                scores[chunk_id] = {'result': result, 'score': 0.0}
            # RRF: 1 / (k + rank)
            rrf_score = 0.6 * (1.0 / (60 + i + 1))
            scores[chunk_id]['score'] += rrf_score
        
        # Lexical results (40% weight)
        for i, result in enumerate(lexical_results):
            chunk_id = result.get('chunk_id')
            if chunk_id not in scores:
                scores[chunk_id] = {'result': result, 'score': 0.0}
            rrf_score = 0.4 * (1.0 / (60 + i + 1))
            scores[chunk_id]['score'] += rrf_score
        
        # Sort by combined score
        sorted_results = sorted(
            scores.values(),
            key=lambda x: x['score'],
            reverse=True
        )
        
        return [item['result'] for item in sorted_results]
    
    async def ingest_chunk(
        self,
        chunk_text: str,
        doc_id: str,
        chunk_id: str,
        page_start: int = 1,
        page_end: int = 1,
        is_ocr: bool = False,
        is_figure: bool = False,
    ) -> bool:
        """
        Ingest a single chunk into the vector store.
        
        Args:
            chunk_text: Text content of chunk
            doc_id: Document ID
            chunk_id: Unique chunk ID
            page_start: Starting page number
            page_end: Ending page number
            is_ocr: Whether this is OCR content
            is_figure: Whether this is figure content
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Generate embedding for chunk
            embedding = embed_text(chunk_text, normalize_emb=True)
            embedding_str = str(embedding.tolist())
            
            # TODO: Insert into database using pgvector
            # For now, just log
            logger.info(f"Ingested chunk {chunk_id} from {doc_id}")
            return True
        
        except Exception as e:
            logger.error(f"Error ingesting chunk: {e}")
            return False
    
    async def ingest_chunks_batch(
        self,
        chunks: List[Dict[str, Any]],
        doc_id: str,
    ) -> int:
        """
        Ingest multiple chunks into the vector store.
        
        Args:
            chunks: List of chunk dictionaries
            doc_id: Document ID
        
        Returns:
            Number of successfully ingested chunks
        """
        success_count = 0
        
        for chunk in chunks:
            success = await self.ingest_chunk(
                chunk_text=chunk.get('text', ''),
                doc_id=doc_id,
                chunk_id=chunk.get('chunk_id'),
                page_start=chunk.get('page_start', 1),
                page_end=chunk.get('page_end', 1),
                is_ocr=chunk.get('is_ocr', False),
                is_figure=chunk.get('is_figure', False),
            )
            
            if success:
                success_count += 1
        
        logger.info(f"Ingested {success_count}/{len(chunks)} chunks for document {doc_id}")
        return success_count
