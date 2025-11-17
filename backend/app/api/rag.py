from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict, Any

from ..rag.graph import run_rag_query

router = APIRouter(tags=["rag"])

class RagQuery(BaseModel):
    thread_id: Optional[str] = None
    query: str
    cross_doc: bool = True
    selected_doc_ids: List[str] = []
    uploaded_doc_ids: List[str] = []

@router.post("/rag/query")
async def rag_query(body: RagQuery):
    try:
        result = await run_rag_query(
            query=body.query,
            thread_id=body.thread_id,
            cross_doc=body.cross_doc,
            selected_doc_ids=body.selected_doc_ids,
            uploaded_doc_ids=body.uploaded_doc_ids,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return result
