from .vllm_client import VLLMBackend
from .llama_cpp_client import LlamaCppBackend
from ..config import settings

_backend_cache = None

def get_backend():
    global _backend_cache
    if _backend_cache is not None:
        return _backend_cache

    mode = settings.inference_mode
    if mode == "vl_fp8":
        _backend_cache = VLLMBackend(
            base_url=settings.vllm_base_url,
            model_id=settings.nemotron_vl_model_id,
        )
    elif mode == "text_4bit":
        _backend_cache = VLLMBackend(
            base_url=settings.vllm_base_url,
            model_id=settings.nemotron_text_model_id,
        )
    elif mode == "cpu_gguf":
        _backend_cache = LlamaCppBackend(
            gguf_path=settings.gguf_model_path,
            gpu_offload_fraction=settings.gpu_offload_fraction,
        )
    else:
        raise ValueError(f"Unknown INFERENCE_MODE {mode!r}")
    return _backend_cache
