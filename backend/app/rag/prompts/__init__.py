"""
Prompt templates for RAG graph nodes.
"""
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

PROMPTS_DIR = Path(__file__).parent


def load_prompt(name: str) -> str:
    """Load a prompt template by name."""
    prompt_file = PROMPTS_DIR / f"{name}.txt"
    if not prompt_file.exists():
        raise FileNotFoundError(f"Prompt template not found: {prompt_file}")
    return prompt_file.read_text()


def format_template(name: str, **kwargs) -> str:
    """Load and format a prompt template with variables."""
    template = load_prompt(name)
    return template.format(**kwargs)

