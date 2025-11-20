import os
from pydantic_settings import BaseSettings
from pydantic import AnyUrl

# Embedding configuration
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "clip")  # "clip" or "nemotron"
CLIP_MODEL = os.getenv("CLIP_MODEL", "openai/clip-vit-large-patch14")
CLIP_EMBEDDING_DIM = int(os.getenv("CLIP_EMBEDDING_DIM", 768))
NEMOTRON_EMBEDDING_MODEL = os.getenv("NEMOTRON_EMBEDDING_MODEL", "nvidia-embed-qa-4")
NEMOTRON_EMBEDDING_DIM = int(os.getenv("NEMOTRON_EMBEDDING_DIM", 1024))

MODEL_DIR = os.getenv("MODEL_DIR", "/mnt/d/Models")
INFERENCE_MODE = os.getenv("INFERENCE_MODE", "vl_fp8")
GPU_OFFLOAD_FRACTION = os.getenv("GPU_OFFLOAD_FRACTION", 0.4)
NEMOTRON_VL_MODEL_ID = os.getenv("NEMOTRON_VL_MODEL_ID", "nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL-FP8")
NEMOTRON_TEXT_MODEL_ID = os.getenv("NEMOTRON_TEXT_MODEL_ID", "nvidia/NVIDIA-Nemotron-Nano-12B-v2")
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql+psycopg://deep_rag:deep_rag@vector_db:5432/deep_rag")
VLLM_BASE_URL = os.getenv("VLLM_BASE_URL", "http://vllm:8000/v1")
K_RETRIEVER = int(os.getenv("K_RETRIEVER", 8))
K_VEC = int(os.getenv("K_VEC", 60))
K_LEX = int(os.getenv("K_LEX", 60))
SYNTHESIZER_CONF_THRESHOLD_DEFAULT = float(os.getenv("SYNTHESIZER_CONF_THRESHOLD_DEFAULT", 0.40))
SYNTHESIZER_CONF_THRESHOLD_EXPLICIT_SELECTION = float(os.getenv("SYNTHESIZER_CONF_THRESHOLD_EXPLICIT_SELECTION", 0.30))
TEMPERATURE_CHAT = float(os.getenv("LLM_TEMPERATURE", 0.15))
MAX_TOKENS_CHAT = int(os.getenv("MAX_TOKENS", 1024))
MAX_ITERS = int(os.getenv("MAX_ITERS", 5))
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", 0.30))

# Model quantization configuration
MODEL_QUANTIZATION = os.getenv("MODEL_QUANTIZATION", "none")  # "fp8", "bf16", or "none"
MODEL_DTYPE = os.getenv("MODEL_DTYPE", "float16")  # "float32", "float16", "bfloat16"

class Settings(BaseSettings):
    # Embedding configuration
    embedding_model: str = EMBEDDING_MODEL  # "clip" or "nemotron"
    clip_model: str = CLIP_MODEL
    clip_embedding_dim: int = CLIP_EMBEDDING_DIM
    nemotron_embedding_model: str = NEMOTRON_EMBEDDING_MODEL
    nemotron_embedding_dim: int = NEMOTRON_EMBEDDING_DIM

    # Database
    database_url: AnyUrl = DATABASE_URL

    # Model storage root
    model_dir: str = MODEL_DIR

    # Model IDs
    nemotron_vl_model_id: str = NEMOTRON_VL_MODEL_ID
    nemotron_text_model_id: str = NEMOTRON_TEXT_MODEL_ID

    # ONNX / GGUF paths (optional – derived from model_dir if not provided)
    nemotron_onnx_path: str | None = None
    gguf_model_path: str | None = None

    # Inference mode
    inference_mode: str = INFERENCE_MODE  # vl_fp8 | text_4bit | cpu_gguf
    gpu_offload_fraction: float = float(GPU_OFFLOAD_FRACTION)

    # vLLM
    vllm_base_url: str = VLLM_BASE_URL

    # RAG parameters
    k_retriever: int = K_RETRIEVER
    k_vec: int = K_VEC
    k_lex: int = K_LEX
    synthesizer_conf_threshold_default: float = SYNTHESIZER_CONF_THRESHOLD_DEFAULT
    synthesizer_conf_threshold_explicit_selection: float = SYNTHESIZER_CONF_THRESHOLD_EXPLICIT_SELECTION
    temperature_chat: float = TEMPERATURE_CHAT
    max_tokens_chat: int = MAX_TOKENS_CHAT
    max_iters: int = MAX_ITERS
    confidence_threshold: float = CONFIDENCE_THRESHOLD
    
    # Model quantization
    model_quantization: str = MODEL_QUANTIZATION
    model_dtype: str = MODEL_DTYPE
    
    class Config:
        env_file = ".env"
        env_prefix = ""
        extra = "ignore"

    def __init__(self, **values):
        super().__init__(**values)
        # Normalize model_dir for cross-platform use
        self.model_dir = self.model_dir.replace("\\", "/")

        # Derive default artifact paths if not set explicitly
        if self.nemotron_onnx_path is None:
            self.nemotron_onnx_path = f"{self.model_dir}/onnx/nemotron_12b_int8.onnx"

        if self.gguf_model_path is None:
            self.gguf_model_path = f"{self.model_dir}/gguf/nemotron_12b_q4_k_m.gguf"


settings = Settings()
