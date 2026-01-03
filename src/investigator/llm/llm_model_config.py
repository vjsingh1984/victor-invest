#!/usr/bin/env python3
"""
LLM Model Configuration and Capability Management
Handles dynamic model context and parameter configuration
"""

from typing import Dict, Optional, Any
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """Configuration for a specific LLM model"""
    name: str
    context_window: int
    default_num_predict: int
    max_num_predict: int
    supports_system_prompt: bool = True
    supports_json_mode: bool = False
    temperature_range: tuple = (0.0, 1.0)
    description: str = ""


# Known model configurations with accurate context windows
MODEL_CONFIGS = {
    # Llama models - PREMIUM (70B+)
    "llama3.3:70b": ModelConfig(
        name="llama3.3:70b",
        context_window=131072,  # 128K context
        default_num_predict=4096,
        max_num_predict=16384,
        supports_system_prompt=True,
        supports_json_mode=True,
        description="Llama 3.3 70B - BEST ACCURACY (70.6B params, 128K context)"
    ),
    "llama-3.3-70b-instruct-q4_k_m-128K-custom": ModelConfig(
        name="llama-3.3-70b-instruct-q4_k_m-128K-custom",
        context_window=131072,  # 128K context
        default_num_predict=4096,
        max_num_predict=16384,
        supports_system_prompt=True,
        supports_json_mode=True,
        description="Llama 3.3 70B Custom - High accuracy (70.6B params)"
    ),
    "llama3.1:8b-instruct-q8_0": ModelConfig(
        name="llama3.1:8b-instruct-q8_0",
        context_window=131072,  # 128K context
        default_num_predict=2048,
        max_num_predict=8192,
        supports_system_prompt=True,
        supports_json_mode=True,
        description="Llama 3.1 8B with 128K context window"
    ),
    "llama3.1:8b": ModelConfig(
        name="llama3.1:8b",
        context_window=131072,  # 128K context
        default_num_predict=2048,
        max_num_predict=8192,
        supports_system_prompt=True,
        supports_json_mode=True,
        description="Llama 3.1 8B base model"
    ),
    "llama-3.3-70b-instruct-q4_k_m": ModelConfig(
        name="llama-3.3-70b-instruct-q4_k_m",
        context_window=131072,  # 128K context
        default_num_predict=4096,
        max_num_predict=16384,
        supports_system_prompt=True,
        supports_json_mode=True,
        description="Llama 3.3 70B with extended context"
    ),
    
    # Phi models
    "phi4-reasoning": ModelConfig(
        name="phi4-reasoning",
        context_window=16384,  # 16K context
        default_num_predict=2048,
        max_num_predict=4096,
        supports_system_prompt=True,
        supports_json_mode=True,
        description="Microsoft Phi-4 reasoning model"
    ),
    "phi4-reasoning:plus": ModelConfig(
        name="phi4-reasoning:plus",
        context_window=32768,  # 32K context
        default_num_predict=3072,
        max_num_predict=6144,
        supports_system_prompt=True,
        supports_json_mode=True,
        description="Microsoft Phi-4 reasoning model with extended context"
    ),
    "phi3:14b-medium-4k-instruct-q4_1": ModelConfig(
        name="phi3:14b-medium-4k-instruct-q4_1",
        context_window=4096,  # 4K context
        default_num_predict=1024,
        max_num_predict=2048,
        supports_system_prompt=True,
        description="Microsoft Phi-3 14B model"
    ),
    
    # DeepSeek models - REASONING SPECIALISTS
    "deepseek-r1:32b": ModelConfig(
        name="deepseek-r1:32b",
        context_window=131072,  # 128K context
        default_num_predict=6144,
        max_num_predict=16384,
        supports_system_prompt=True,
        supports_json_mode=True,
        description="DeepSeek R1 32B - REASONING EXPERT (32.8B params, 128K context, thinking capability)"
    ),

    # Qwen models (reasoning models with thinking tags)
    "qwen3:30b": ModelConfig(
        name="qwen3:30b",
        context_window=262144,  # 262K context - MASSIVE MoE!
        default_num_predict=4096,
        max_num_predict=16384,
        supports_system_prompt=True,
        supports_json_mode=True,
        description="Qwen3 30B MoE - HUGE CONTEXT (30.5B params, 262K context, thinking + tools)"
    ),
    "qwen3:32b": ModelConfig(
        name="qwen3:32b",
        context_window=40960,  # 40K context - VERIFIED from ollama show
        default_num_predict=4096,
        max_num_predict=8192,
        supports_system_prompt=True,
        supports_json_mode=True,
        description="Qwen3 32B with 40K context - excellent for long documents"
    ),
    "qwen3-coder:30b": ModelConfig(
        name="qwen3-coder:30b",
        context_window=262144,  # 262K context - MASSIVE!
        default_num_predict=4096,
        max_num_predict=16384,
        supports_system_prompt=True,
        supports_json_mode=True,
        description="Qwen3 Coder 30B with 262K context - code specialist"
    ),
    "qwen3:30b-a3b": ModelConfig(
        name="qwen3:30b-a3b",
        context_window=40960,  # 40K context
        default_num_predict=4096,
        max_num_predict=8192,
        supports_system_prompt=True,
        supports_json_mode=True,
        description="Qwen3 30B variant with 40K context"
    ),
    "qwen3-30b-40k-financial:latest": ModelConfig(
        name="qwen3-30b-40k-financial:latest",
        context_window=40960,  # 40K context
        default_num_predict=4096,
        max_num_predict=8192,
        supports_system_prompt=True,
        supports_json_mode=True,
        description="Qwen3 30B financial model with thinking"
    ),
    "qwen2.5:32b-instruct-q4_K_M": ModelConfig(
        name="qwen2.5:32b-instruct-q4_K_M",
        context_window=32768,  # 32K context
        default_num_predict=3072,
        max_num_predict=6144,
        supports_system_prompt=True,
        supports_json_mode=True,
        description="Qwen 2.5 32B instruction model"
    ),
    
    # Mixtral models
    "mixtral:8x7b-instruct-v0.1-q4_K_M": ModelConfig(
        name="mixtral:8x7b-instruct-v0.1-q4_K_M",
        context_window=32768,  # 32K context
        default_num_predict=2048,
        max_num_predict=4096,
        supports_system_prompt=True,
        description="Mixtral 8x7B MoE model"
    ),
    "mistral:v0.3": ModelConfig(
        name="mistral:v0.3",
        context_window=8192,  # 8K context
        default_num_predict=1024,
        max_num_predict=2048,
        supports_system_prompt=True,
        description="Mistral v0.3 base model"
    ),
    
    # Default fallback configuration
    "default": ModelConfig(
        name="default",
        context_window=4096,
        default_num_predict=1024,
        max_num_predict=2048,
        supports_system_prompt=True,
        description="Default fallback configuration"
    )
}


class ModelConfigManager:
    """Manages model configurations and capabilities"""

    def __init__(self, config=None):
        self.config = config  # Store config for model_specs access
        self.configs = MODEL_CONFIGS
        self.logger = logger

    def get_model_config(self, model_name: str) -> ModelConfig:
        """Get configuration for a specific model"""
        # PRIORITY 1: Try to get from config.json model_specs (single source of truth)
        if self.config and hasattr(self.config, 'ollama'):
            ollama_config = self.config.ollama
            if hasattr(ollama_config, 'model_specs'):
                model_specs = ollama_config.model_specs
                # Try exact match first
                if model_name in model_specs:
                    spec = model_specs[model_name]
                    # Convert to ModelConfig dataclass
                    self.logger.info(f"Using config.json spec for {model_name}")
                    return ModelConfig(
                        name=model_name,
                        context_window=getattr(spec, 'context_window', 4096),
                        default_num_predict=getattr(spec, 'default_num_predict', 2048),
                        max_num_predict=getattr(spec, 'max_num_predict', 4096),
                        supports_system_prompt=True,
                        supports_json_mode=True,
                        description=f"{model_name} from config.json"
                    )
                # Try partial match
                for spec_model, spec in model_specs.items():
                    if spec_model in model_name or model_name in spec_model:
                        self.logger.info(f"Using config.json spec for {spec_model} (matched {model_name})")
                        return ModelConfig(
                            name=spec_model,
                            context_window=getattr(spec, 'context_window', 4096),
                            default_num_predict=getattr(spec, 'default_num_predict', 2048),
                            max_num_predict=getattr(spec, 'max_num_predict', 4096),
                            supports_system_prompt=True,
                            supports_json_mode=True,
                            description=f"{spec_model} from config.json"
                        )

        # PRIORITY 2: Fallback to hardcoded MODEL_CONFIGS for backward compatibility
        if model_name in self.configs:
            return self.configs[model_name]

        # Try to match partial names
        for known_model, config in self.configs.items():
            if known_model in model_name or model_name in known_model:
                self.logger.info(f"Using fallback config for {known_model} for model {model_name}")
                return config

        # PRIORITY 3: Return default config
        self.logger.warning(f"Unknown model {model_name}, using default configuration")
        return self.configs["default"]
    
    def get_optimal_context_size(self, model_name: str, prompt_length: int, 
                                desired_output: int, task_type: str) -> Dict[str, int]:
        """Calculate optimal context parameters for a request"""
        config = self.get_model_config(model_name)
        
        # Estimate tokens (rough: 1 token ≈ 4 chars)
        prompt_tokens = prompt_length // 4
        
        # Get desired output tokens based on task type
        if task_type == "synthesis":
            # Synthesis needs more output for comprehensive analysis
            output_tokens = min(desired_output or 6144, config.max_num_predict)
        elif task_type == "sec":
            # SEC analysis needs moderate output
            output_tokens = min(desired_output or 4096, config.max_num_predict)
        elif task_type == "ta":
            # Technical analysis needs more tokens for comprehensive JSON output
            output_tokens = min(desired_output or 4096, config.max_num_predict)
        else:
            output_tokens = min(desired_output or config.default_num_predict, config.max_num_predict)
        
        # Calculate total needed context
        total_needed = prompt_tokens + output_tokens + 512  # 512 buffer
        
        # Ensure we don't exceed model's context window
        if total_needed > config.context_window:
            # Scale down proportionally
            scale_factor = config.context_window / total_needed * 0.9  # 90% to be safe
            output_tokens = int(output_tokens * scale_factor)
            total_needed = prompt_tokens + output_tokens + 512
        
        # Use the smaller of model's context window and what we need
        optimal_context = min(config.context_window, total_needed)
        
        self.logger.info(
            f"Model: {model_name}, Task: {task_type}, "
            f"Prompt tokens: {prompt_tokens}, Output tokens: {output_tokens}, "
            f"Total context: {optimal_context}/{config.context_window}"
        )
        
        return {
            "num_ctx": optimal_context,
            "num_predict": output_tokens,
            "prompt_tokens": prompt_tokens,
            "max_context": config.context_window
        }
    
    def get_recommended_models(self, task_type: str, context_needed: int) -> list:
        """Get recommended models for a specific task"""
        suitable_models = []
        
        for model_name, config in self.configs.items():
            if model_name == "default":
                continue
                
            if config.context_window >= context_needed:
                suitable_models.append({
                    "model": model_name,
                    "context_window": config.context_window,
                    "max_output": config.max_num_predict,
                    "description": config.description
                })
        
        # Sort by context window size
        suitable_models.sort(key=lambda x: x["context_window"], reverse=True)
        
        return suitable_models


# Singleton instance
_model_config_manager = None


def get_model_config_manager(config=None) -> ModelConfigManager:
    """Get singleton instance of ModelConfigManager

    Args:
        config: Optional config object with model_specs. If provided on first call,
                will be used to initialize the manager. Subsequent calls ignore this parameter.
    """
    global _model_config_manager
    if _model_config_manager is None:
        _model_config_manager = ModelConfigManager(config=config)
    return _model_config_manager