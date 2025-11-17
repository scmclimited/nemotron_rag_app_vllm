from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .config import settings
from .api import chat, rag

app = FastAPI(title="Nemotron VL Local RAG API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(chat.router, prefix="/v1")
app.include_router(rag.router, prefix="")

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "inference_mode": settings.inference_mode,
        "vllm_base_url": settings.vllm_base_url,
    }
