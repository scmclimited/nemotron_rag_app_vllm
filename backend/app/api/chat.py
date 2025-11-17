from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Any, Dict, List, Optional

from ..inference.router import get_backend

router = APIRouter(tags=["chat"])

class ChatMessage(BaseModel):
    role: str
    content: Any

class ChatCompletionRequest(BaseModel):
    model: Optional[str] = None
    messages: List[ChatMessage]
    max_tokens: Optional[int] = 512
    temperature: Optional[float] = 0.2

@router.post("/chat/completions")
async def chat_completions(body: ChatCompletionRequest):
    backend = get_backend()
    try:
        response = await backend.chat_completion(body)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return response
