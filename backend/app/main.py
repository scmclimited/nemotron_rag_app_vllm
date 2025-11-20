from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .config import settings
from .api import chat, rag, ingest

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
app.include_router(ingest.router, prefix="")

@app.get("/health")
async def health():
    return {
        "status": "ok",
        # Embedding configuration
        "embedding_model": settings.embedding_model,
        "clip_model": settings.clip_model,
        "clip_embedding_dim": settings.clip_embedding_dim,
        "nemotron_embedding_model": settings.nemotron_embedding_model,
        "nemotron_embedding_dim": settings.nemotron_embedding_dim,
        # Inference configuration
        "temperature_chat": settings.temperature_chat,
        "inference_mode": settings.inference_mode,
        "vllm_base_url": settings.vllm_base_url,
        "model_dir": settings.model_dir,
        "nemotron_vl_model_id": settings.nemotron_vl_model_id,
        "nemotron_text_model_id": settings.nemotron_text_model_id,
        # Database
        "database_url": settings.database_url,
        # RAG parameters
        "k_retriever": settings.k_retriever,
        "k_vec": settings.k_vec,
        "k_lex": settings.k_lex,
        "max_iters": settings.max_iters,
        "confidence_threshold": settings.confidence_threshold,
        "synthesizer_conf_threshold_default": settings.synthesizer_conf_threshold_default,
        "synthesizer_conf_threshold_explicit_selection": settings.synthesizer_conf_threshold_explicit_selection,
    }
