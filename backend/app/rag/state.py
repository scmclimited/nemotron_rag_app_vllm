"""
State definition for LangGraph RAG pipeline (mimics deep_rag).
"""
from typing import TypedDict, List, Dict, Any, Optional


class GraphState(TypedDict, total=False):
    """Graph state for RAG pipeline matching deep_rag structure."""
    
    # Query and planning
    query: str
    plan: str
    
    # Evidence and retrieval
    evidence: List[Dict[str, Any]]
    notes: str
    
    # Output
    answer: str
    confidence: float
    action: str  # Decision action: "refine", "synthesize", or "abstain"
    
    # Iteration tracking
    iterations: int
    refinements: List[str]
    
    # Document filtering
    thread_id: Optional[str]
    doc_id: Optional[str]  # Primary document ID for document-specific retrieval
    selected_doc_ids: Optional[List[str]]  # Multi-document selection (not cross-doc)
    uploaded_doc_ids: Optional[List[str]]  # Uploaded documents
    doc_ids: List[str]  # All document IDs found during retrieval (for multi-doc tracking)
    cross_doc: bool  # Whether cross-document retrieval is enabled
    
    # Additional metadata
    llm_raw: Optional[Dict[str, Any]]  # Raw LLM response
    citations: Optional[List[Dict[str, Any]]]  # Citation chunks for final answer
    embedding_model: Optional[str]  # Track which embedding model was used

