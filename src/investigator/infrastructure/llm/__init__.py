"""
LLM Infrastructure

Multi-server Ollama pool with VRAM-aware scheduling and resource management.
"""

# vram_calculator contains utility functions, not classes
from investigator.infrastructure.llm import vram_calculator
from investigator.infrastructure.llm.ollama import OllamaClient
from investigator.infrastructure.llm.pool import ResourceAwareOllamaPool
from investigator.infrastructure.llm.semaphore import DynamicLLMSemaphore

__all__ = [
    "OllamaClient",
    "ResourceAwareOllamaPool",
    "vram_calculator",
    "DynamicLLMSemaphore",
]
