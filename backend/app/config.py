from pydantic_settings import BaseSettings
from pydantic import AnyUrl


class Settings(BaseSettings):
    # Database
    database_url: AnyUrl = "postgresql+psycopg://deep_rag:deep_rag@vector_db:5432/deep_rag"

    # Model IDs
    nemotron_vl_model_id: str = "nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL-FP8"
    nemotron_text_model_id: str = "nvidia/NVIDIA-Nemotron-Nano-12B-v2"

    # ONNX / GGUF paths (optional)
    nemotron_onnx_path: str = "backend_models/nemotron_12b_int8.onnx"
    gguf_model_path: str = "backend_models/nemotron_12b_q4_k_m.gguf"

    # Inference mode
    inference_mode: str = "vl_fp8"  # vl_fp8 | text_4bit | cpu_gguf
    gpu_offload_fraction: float = 0.4

    # vLLM
    vllm_base_url: str = "http://vllm:8000/v1"

    # RAG parameters
    k_retriever: int = 8
    k_vec: int = 60
    k_lex: int = 60
    synthesizer_conf_threshold_default: float = 0.40
    synthesizer_conf_threshold_explicit_selection: float = 0.30

    class Config:
        env_file = ".env"
        env_prefix = ""
        extra = "ignore"


settings = Settings()
