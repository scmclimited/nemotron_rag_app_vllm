"""
Embedding providers for CLIP and Nemotron embeddings.
Supports toggling between models via environment variable.
"""
import logging
import numpy as np
import torch
from abc import ABC, abstractmethod
from typing import List, Optional
from transformers import CLIPProcessor, CLIPModel, AutoTokenizer, AutoModel

from ..config import settings

logger = logging.getLogger(__name__)

# Global model caches
_clip_model = None
_clip_processor = None
_nemotron_model = None
_nemotron_tokenizer = None


class EmbeddingProvider(ABC):
    """Abstract base class for embedding providers."""
    
    @abstractmethod
    def embed_text(self, text: str) -> np.ndarray:
        """Embed a single text string."""
        pass
    
    @abstractmethod
    def embed_batch(self, texts: List[str]) -> np.ndarray:
        """Embed a batch of texts."""
        pass
    
    @abstractmethod
    def get_embedding_dim(self) -> int:
        """Get the dimension of embeddings."""
        pass
    
    @abstractmethod
    def get_model_name(self) -> str:
        """Get the name of the embedding model."""
        pass


class CLIPEmbeddingProvider(EmbeddingProvider):
    """CLIP embedding provider using openai/clip-vit-large-patch14."""
    
    def __init__(self):
        """Initialize CLIP model and processor."""
        self._load_models()
    
    @staticmethod
    def _load_models():
        """Load CLIP model and processor globally."""
        global _clip_model, _clip_processor
        
        if _clip_model is None:
            try:
                model_name = settings.clip_model
                logger.info(f"Loading CLIP model: {model_name}")
                
                _clip_model = CLIPModel.from_pretrained(model_name)
                _clip_processor = CLIPProcessor.from_pretrained(model_name)
                
                _clip_model.eval()
                if torch.cuda.is_available():
                    _clip_model = _clip_model.to("cuda")
                
                # Validate model
                test_text = "test"
                inputs = _clip_processor(
                    text=[test_text],
                    return_tensors="pt",
                    padding=True,
                    truncation=True
                )
                if torch.cuda.is_available():
                    inputs = {k: v.to("cuda") for k, v in inputs.items()}
                
                with torch.no_grad():
                    text_features = _clip_model.get_text_features(**inputs)
                    test_embedding = text_features[0].cpu().numpy()
                
                assert test_embedding is not None and len(test_embedding) > 0
                logger.info(f"✓ CLIP model loaded successfully ({settings.clip_embedding_dim} dims)")
                
            except Exception as e:
                logger.error(f"Failed to load CLIP model: {e}")
                _clip_model = None
                _clip_processor = None
                raise
    
    def embed_text(self, text: str) -> np.ndarray:
        """Embed a single text using CLIP."""
        if _clip_model is None:
            raise RuntimeError("CLIP model not loaded")
        
        try:
            inputs = _clip_processor(
                text=[text],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=77  # CLIP token limit
            )
            if torch.cuda.is_available():
                inputs = {k: v.to("cuda") for k, v in inputs.items()}
            
            with torch.no_grad():
                text_features = _clip_model.get_text_features(**inputs)
                embedding = text_features[0].cpu().numpy()
            
            # Normalize
            embedding = embedding / (np.linalg.norm(embedding) + 1e-8)
            return embedding.astype(np.float32)
        
        except Exception as e:
            logger.error(f"Error embedding text with CLIP: {e}")
            raise
    
    def embed_batch(self, texts: List[str]) -> np.ndarray:
        """Embed a batch of texts using CLIP."""
        if _clip_model is None:
            raise RuntimeError("CLIP model not loaded")
        
        try:
            inputs = _clip_processor(
                text=texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=77
            )
            if torch.cuda.is_available():
                inputs = {k: v.to("cuda") for k, v in inputs.items()}
            
            with torch.no_grad():
                text_features = _clip_model.get_text_features(**inputs)
                embeddings = text_features.cpu().numpy()
            
            # Normalize each embedding
            embeddings = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8)
            return embeddings.astype(np.float32)
        
        except Exception as e:
            logger.error(f"Error embedding batch with CLIP: {e}")
            raise
    
    def get_embedding_dim(self) -> int:
        """Get CLIP embedding dimension."""
        return settings.clip_embedding_dim
    
    def get_model_name(self) -> str:
        """Get CLIP model name."""
        return settings.clip_model


class NemotronEmbeddingProvider(EmbeddingProvider):
    """Nemotron embedding provider using dedicated embedding model."""
    
    def __init__(self):
        """Initialize Nemotron embedding model and tokenizer."""
        self._load_models()
    
    @staticmethod
    def _load_models():
        """Load Nemotron model and tokenizer globally."""
        global _nemotron_model, _nemotron_tokenizer
        
        if _nemotron_model is None:
            try:
                model_name = settings.nemotron_embedding_model
                logger.info(f"Loading Nemotron embedding model: {model_name}")
                
                _nemotron_tokenizer = AutoTokenizer.from_pretrained(model_name)
                _nemotron_model = AutoModel.from_pretrained(
                    model_name,
                    trust_remote_code=True
                )
                
                _nemotron_model.eval()
                if torch.cuda.is_available():
                    _nemotron_model = _nemotron_model.to("cuda")
                
                # Validate model
                test_text = "test"
                inputs = _nemotron_tokenizer(
                    test_text,
                    return_tensors="pt",
                    padding=True,
                    truncation=True
                )
                if torch.cuda.is_available():
                    inputs = {k: v.to("cuda") for k, v in inputs.items()}
                
                with torch.no_grad():
                    outputs = _nemotron_model(**inputs)
                    test_embedding = outputs.last_hidden_state[:, 0, :][0].cpu().numpy()
                
                assert test_embedding is not None and len(test_embedding) > 0
                logger.info(f"✓ Nemotron model loaded successfully ({settings.nemotron_embedding_dim} dims)")
                
            except Exception as e:
                logger.error(f"Failed to load Nemotron model: {e}")
                _nemotron_model = None
                _nemotron_tokenizer = None
                raise
    
    def embed_text(self, text: str) -> np.ndarray:
        """Embed a single text using Nemotron."""
        if _nemotron_model is None:
            raise RuntimeError("Nemotron model not loaded")
        
        try:
            inputs = _nemotron_tokenizer(
                text,
                return_tensors="pt",
                padding=True,
                truncation=True
            )
            if torch.cuda.is_available():
                inputs = {k: v.to("cuda") for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = _nemotron_model(**inputs)
                # Use [CLS] token representation (first token)
                embedding = outputs.last_hidden_state[:, 0, :][0].cpu().numpy()
            
            # Normalize
            embedding = embedding / (np.linalg.norm(embedding) + 1e-8)
            return embedding.astype(np.float32)
        
        except Exception as e:
            logger.error(f"Error embedding text with Nemotron: {e}")
            raise
    
    def embed_batch(self, texts: List[str]) -> np.ndarray:
        """Embed a batch of texts using Nemotron."""
        if _nemotron_model is None:
            raise RuntimeError("Nemotron model not loaded")
        
        try:
            inputs = _nemotron_tokenizer(
                texts,
                return_tensors="pt",
                padding=True,
                truncation=True
            )
            if torch.cuda.is_available():
                inputs = {k: v.to("cuda") for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = _nemotron_model(**inputs)
                # Use [CLS] token representation
                embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            
            # Normalize
            embeddings = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8)
            return embeddings.astype(np.float32)
        
        except Exception as e:
            logger.error(f"Error embedding batch with Nemotron: {e}")
            raise
    
    def get_embedding_dim(self) -> int:
        """Get Nemotron embedding dimension."""
        return settings.nemotron_embedding_dim
    
    def get_model_name(self) -> str:
        """Get Nemotron model name."""
        return settings.nemotron_embedding_model


def get_embedding_provider() -> EmbeddingProvider:
    """
    Get the active embedding provider based on config.
    
    Returns:
        EmbeddingProvider instance (CLIP or Nemotron)
    """
    embedding_model = settings.embedding_model.lower()
    
    if embedding_model == "clip":
        logger.info("Using CLIP embedding provider")
        return CLIPEmbeddingProvider()
    elif embedding_model == "nemotron":
        logger.info("Using Nemotron embedding provider")
        return NemotronEmbeddingProvider()
    else:
        raise ValueError(
            f"Unknown embedding model: {embedding_model}. "
            f"Must be 'clip' or 'nemotron'. Set EMBEDDING_MODEL in .env"
        )

