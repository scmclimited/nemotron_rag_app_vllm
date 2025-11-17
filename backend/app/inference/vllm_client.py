import httpx
from typing import Any, Dict

from pydantic import BaseModel

class VLLMBackend:
    '''Simple client for a vLLM OpenAI-style server.'''

    def __init__(self, base_url: str, model_id: str):
        self.base_url = base_url.rstrip("/")
        self.model_id = model_id

    async def chat_completion(self, body: BaseModel) -> Dict[str, Any]:
        payload = body.model_dump()
        if not payload.get("model"):
            payload["model"] = self.model_id

        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(f"{self.base_url}/chat/completions", json=payload)
            resp.raise_for_status()
            return resp.json()
