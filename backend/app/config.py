from pydantic_settings import BaseSettings
from pydantic import AnyUrl


class Settings(BaseSettings):
    # Database
    database_url: AnyUrl = "postgresql+psycopg://deep_rag:deep_rag@vector_db:5432/deep_rag"

    # Model storage root
    model_dir: str = "D:/Models"

    # Model IDs
    nemotron_vl_model_id: str = "nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL-FP8"
    nemotron_text_model_id: str = "nvidia/NVIDIA-Nemotron-Nano-12B-v2"

    # ONNX / GGUF paths (optional – derived from model_dir if not provided)
    nemotron_onnx_path: str | None = None
    gguf_model_path: str | None = None

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
