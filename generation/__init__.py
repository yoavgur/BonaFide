"""Reproducible LLM generation pipeline.

Usage:
    python -m generation --model Qwen/Qwen3-4B-Thinking-2507 --input questions.csv
"""

from generation.config import GenerationConfig
from generation.model_registry import ModelProfile, get_model_profile
from generation.thinking import split_thinking

__all__ = [
    "GenerationConfig",
    "ModelProfile",
    "get_model_profile",
    "split_thinking",
]
