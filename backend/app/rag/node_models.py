"""
Node-specific model configuration for LangGraph pipeline.

Allows per-node LLM model selection, quantization, and dtype configuration
via environment variables.

Supported nodes:
- planner
- retriever
- compressor
- critic
- refine_retrieve
- synthesizer
- citation_pruner

Environment variable patterns:
- {NODE_NAME}_MODEL: Override LLM model for specific node
- {NODE_NAME}_QUANTIZATION: fp8, bf16, or none
- {NODE_NAME}_DTYPE: float32, float16, bfloat16
"""

import os
import logging
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

# Supported node names
SUPPORTED_NODES = {
    "planner",
    "retriever",
    "compressor",
    "critic",
    "refine_retrieve",
    "synthesizer",
    "citation_pruner",
}


class NodeModelConfig:
    """Configuration for node-specific model settings."""
    
    def __init__(self, node_name: str):
        self.node_name = node_name
        if node_name not in SUPPORTED_NODES:
            raise ValueError(f"Unknown node: {node_name}. Supported: {SUPPORTED_NODES}")
        
        self._load_config()
    
    def _load_config(self):
        """Load node-specific configuration from environment variables."""
        node_upper = self.node_name.upper()
        
        # Model configuration
        self.model = os.getenv(f"{node_upper}_MODEL")
        self.quantization = os.getenv(f"{node_upper}_QUANTIZATION", "none")
        self.dtype = os.getenv(f"{node_upper}_DTYPE")
        
        # Log configuration
        if self.model or self.quantization != "none" or self.dtype:
            logger.info(
                f"Node '{self.node_name}' config: "
                f"model={self.model}, quantization={self.quantization}, dtype={self.dtype}"
            )
    
    def get_model(self, default: Optional[str] = None) -> Optional[str]:
        """Get model for this node, or default if not configured."""
        return self.model or default
    
    def get_quantization(self) -> str:
        """Get quantization setting (fp8, bf16, or none)."""
        return self.quantization
    
    def get_dtype(self) -> Optional[str]:
        """Get dtype setting."""
        return self.dtype
    
    def is_quantized(self) -> bool:
        """Check if this node uses quantization."""
        return self.quantization in ("fp8", "bf16")
    
    def to_dict(self) -> Dict[str, Any]:
        """Export configuration as dictionary."""
        return {
            "node_name": self.node_name,
            "model": self.model,
            "quantization": self.quantization,
            "dtype": self.dtype,
        }


def get_node_config(node_name: str) -> NodeModelConfig:
    """Get configuration for a specific node."""
    return NodeModelConfig(node_name)


def get_all_node_configs() -> Dict[str, NodeModelConfig]:
    """Get configurations for all nodes."""
    return {node: get_node_config(node) for node in SUPPORTED_NODES}


def validate_quantization(quantization: str) -> bool:
    """Validate quantization value."""
    return quantization in ("none", "fp8", "bf16")


def validate_dtype(dtype: str) -> bool:
    """Validate dtype value."""
    valid_dtypes = ("float32", "float16", "bfloat16", "torch.float32", "torch.float16", "torch.bfloat16")
    return dtype in valid_dtypes


def get_node_model_config_info() -> str:
    """Get human-readable info about node model configuration."""
    configs = get_all_node_configs()
    lines = ["Node-Specific Model Configuration:"]
    
    for node_name, config in configs.items():
        if config.model or config.quantization != "none" or config.dtype:
            lines.append(f"  {node_name}:")
            if config.model:
                lines.append(f"    model: {config.model}")
            if config.quantization != "none":
                lines.append(f"    quantization: {config.quantization}")
            if config.dtype:
                lines.append(f"    dtype: {config.dtype}")
    
    if len(lines) == 1:
        lines.append("  (None configured - using global defaults)")
    
    return "\n".join(lines)


def log_node_config_info():
    """Log all node model configurations."""
    logger.info(get_node_model_config_info())

