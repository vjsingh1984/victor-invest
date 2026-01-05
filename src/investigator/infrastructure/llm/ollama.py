"""
Ollama REST API Client
Async client for interacting with Ollama's REST API
"""

import asyncio
import json
import logging
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import AsyncIterator, Dict, List, Optional, Set, Union

import aiohttp


# Exception Classes (Fix: LLM client ignores HTTP failures)
class OllamaError(Exception):
    """Base exception for Ollama client errors"""

    def __init__(
        self,
        message: str,
        status_code: Optional[int] = None,
        endpoint: Optional[str] = None,
        model: Optional[str] = None,
    ):
        self.status_code = status_code
        self.endpoint = endpoint
        self.model = model

        # Build detailed error message
        parts = [message]
        if status_code:
            parts.append(f"status={status_code}")
        if endpoint:
            parts.append(f"endpoint={endpoint}")
        if model:
            parts.append(f"model={model}")

        super().__init__(f"{parts[0]} ({', '.join(parts[1:])})" if len(parts) > 1 else parts[0])


class OllamaHTTPError(OllamaError):
    """HTTP error from Ollama API (4xx/5xx responses)"""

    pass


class OllamaConnectionError(OllamaError):
    """Connection error to Ollama server"""

    pass


class OllamaTimeoutError(OllamaError):
    """Timeout error from Ollama server"""

    pass


class OllamaModel(Enum):
    """Available Ollama models with their characteristics (from actual installed models)"""

    # Premium reasoning models (70B+ parameters) - BEST FOR ACCURACY
    LLAMA3_3_70B_CUSTOM = "llama-3.3-70b-instruct-q4_k_m-128K-custom"  # 70.6B, 128K context
    LLAMA3_3_70B = "llama3.3:70b"  # 70.6B, 128K context - LATEST, MOST ACCURATE

    # Large reasoning models (30-40B parameters)
    DEEPSEEK_R1_32B = "deepseek-r1:32b"  # 32.8B, 128K context, THINKING capability
    QWEN3_32B = "qwen3:32b"  # 32.8B, 40K context, thinking + tools
    QWEN2_5_32B = "qwen2.5:32b-instruct-q4_K_M"  # 32.8B, 32K context
    QWEN3_30B_MoE = "qwen3:30b"  # 30.5B, 262K context, MoE, thinking + tools
    QWEN3_30B = "qwen3:30b-a3b"  # 30.5B, 40K context
    QWEN3_CODER_30B = "qwen3-coder:30b"  # 30.5B, 262K context, code specialist
    DEEPSEEK_CODER_33B = "deepseek-coder:33b-instruct"  # 33B, 16K context
    CODELLAMA_34B = "codellama:34b-python"  # 34B, 16K context

    # Medium models (8-27B parameters)
    GEMMA3_27B = "gemma3:27b"  # 27B params
    PHI4_REASONING = "phi4-reasoning:plus"  # 14.7B, 32K context
    PHI4_16K = "phi-4-q8_0-16K-custom"  # 14.7B, 16K context
    LLAMA3_1_8B = "llama3.1:8b-instruct-q8_0"  # 8B, 131K context
    MISTRAL_7B = "mistral:7b-instruct"  # 7B, 32K context (estimated)

    # Fast inference models (legacy)
    LLAMA3_2 = "llama3.2:latest"  # 2B params
    PHI3_MINI = "phi3:mini"  # Small, fast

    # Specialized models
    MIXTRAL_8X7B = "mixtral-8x7b-q4_k_m-32K"  # MoE, 32K context


@dataclass
class ModelConfig:
    """Configuration for model inference"""

    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 40
    num_predict: int = 4096
    stop: Optional[List[str]] = None
    seed: Optional[int] = None
    num_ctx: int = 16384  # CRITICAL: Increased from 4096 to 16384 for detailed prompts
    repeat_penalty: float = 1.1
    mirostat: int = 0
    mirostat_tau: float = 5.0
    mirostat_eta: float = 0.1


class OllamaClient:
    """
    Async client for Ollama REST API with advanced features
    """

    def __init__(self, base_url: str = "http://localhost:11434", timeout: int = 300, max_retries: int = 3):
        self.base_url = base_url
        self.timeout = aiohttp.ClientTimeout(total=timeout)
        self.max_retries = max_retries
        self.logger = logging.getLogger(self.__class__.__name__)
        self.verbose = False

        # Session management
        self._session: Optional[aiohttp.ClientSession] = None

        # Model management
        self.loaded_models: Set[str] = set()
        self.model_info: Dict[str, Dict] = {}

        # Performance tracking
        self.request_count = 0
        self.total_tokens = 0
        self.total_duration = 0

    @staticmethod
    def _estimate_tokens(text: Optional[str]) -> int:
        """Very rough heuristic to estimate token count from text length"""
        if not text:
            return 0
        # Average 4 characters per token with small buffer
        return max(1, len(text) // 4 + 1)

    async def __aenter__(self):
        """Async context manager entry"""
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.close()

    async def connect(self):
        """Initialize HTTP session"""
        if not self._session:
            connector = aiohttp.TCPConnector(limit=100, limit_per_host=30)
            self._session = aiohttp.ClientSession(connector=connector, timeout=self.timeout)

            # Check Ollama availability
            await self.health_check()

            # Load available models
            await self.list_models()

    async def close(self):
        """Close HTTP session"""
        if self._session:
            await self._session.close()
            self._session = None

    async def health_check(self) -> bool:
        """Check if Ollama service is healthy"""
        try:
            async with self._session.get(f"{self.base_url}/api/tags") as response:
                return response.status == 200
        except Exception as e:
            self.logger.error(f"Ollama health check failed: {e}")
            return False

    async def list_models(self) -> List[Dict]:
        """List available models"""
        try:
            async with self._session.get(f"{self.base_url}/api/tags") as response:
                if response.status == 200:
                    data = await response.json()
                    models = data.get("models", [])

                    # Cache model info
                    for model in models:
                        self.model_info[model["name"]] = model
                        self.loaded_models.add(model["name"])

                    return models
                return []
        except Exception as e:
            self.logger.error(f"Failed to list models: {e}")
            return []

    async def pull_model(self, model: str, stream: bool = True) -> AsyncIterator[Dict]:
        """Pull a model from Ollama registry"""
        # Ensure session is connected
        if not self._session:
            await self.connect()

        payload = {"name": model, "stream": stream}
        endpoint = f"{self.base_url}/api/pull"

        try:
            async with self._session.post(endpoint, json=payload) as response:
                # Fix: Check HTTP status before parsing JSON
                if response.status != 200:
                    error_text = await response.text()
                    raise OllamaHTTPError(
                        f"Failed to pull model '{model}': {error_text}",
                        status_code=response.status,
                        endpoint=endpoint,
                        model=model,
                    )

                if stream:
                    async for line in response.content:
                        if line:
                            yield json.loads(line)
                else:
                    yield await response.json()

        except OllamaHTTPError:
            raise  # Re-raise with context preserved

        except aiohttp.ClientError as e:
            raise OllamaConnectionError(
                f"Connection error pulling model '{model}': {e}", endpoint=endpoint, model=model
            )

        except asyncio.TimeoutError as e:
            raise OllamaTimeoutError(f"Timeout pulling model '{model}': {e}", endpoint=endpoint, model=model)

        except Exception as e:
            self.logger.error(f"Unexpected error pulling model {model}: {e}")
            raise OllamaError(f"Unexpected error pulling model: {e}", endpoint=endpoint, model=model)

    async def generate(
        self,
        model: str,
        prompt: str,
        system: Optional[str] = None,
        format: Optional[str] = None,
        images: Optional[List[str]] = None,
        config: Optional[ModelConfig] = None,
        stream: bool = False,
        prompt_name: Optional[str] = None,
        **extra_kwargs,
    ) -> Union[Dict, AsyncIterator[Dict]]:
        """
        Generate completion from model

        Args:
            model: Model name to use
            prompt: User prompt
            system: System prompt
            format: Response format (e.g., "json")
            images: Base64 encoded images
            config: Model configuration
            stream: Whether to stream response

        Returns:
            Response dict or async iterator for streaming
        """
        # Import dynamic semaphore
        from investigator.infrastructure.llm.semaphore import DynamicLLMContext

        # Determine task type from context (if available)
        task_type = "summary"  # default
        if hasattr(self, "_current_task_type"):
            task_type = self._current_task_type
        elif "analysis" in prompt.lower():
            if "technical" in prompt.lower():
                task_type = "technical"
            elif "fundamental" in prompt.lower():
                task_type = "fundamental"
            elif "sec" in prompt.lower() or "filing" in prompt.lower():
                task_type = "sec"
            elif "synthesis" in prompt.lower() or "recommendation" in prompt.lower():
                task_type = "synthesis"

        # Check if this might be using cached data
        is_cached = "cache" in prompt.lower() or (hasattr(self, "_use_cached_data") and self._use_cached_data)

        task_id = f"{model}_{format or 'text'}_{len(prompt)}"

        # Use dynamic VRAM-aware resource management
        prompt_tokens = self._estimate_tokens(prompt)
        prompt_tokens += self._estimate_tokens(system) if system else 0

        context_limit = config.num_ctx if config and getattr(config, "num_ctx", None) else 4096
        context_limit = max(1024, context_limit)

        prompt_tokens = min(prompt_tokens, context_limit)
        response_tokens = config.num_predict if config and getattr(config, "num_predict", None) else 1024
        response_tokens = max(1, min(response_tokens, max(1, context_limit - prompt_tokens)))

        async with DynamicLLMContext(
            model,
            task_type,
            is_cached,
            task_id,
            prompt_tokens=prompt_tokens,
            response_tokens=response_tokens,
            context_tokens=context_limit,
        ):
            if not config:
                config = ModelConfig()

            # Ensure session is initialized
            if not self._session:
                await self.connect()

            # Check if model is available (skip pulling for now)
            if model not in self.loaded_models:
                self.logger.info(f"Model {model} not in cache, attempting to use directly")
                self.loaded_models.add(model)

            payload = {
                "model": model,
                "prompt": prompt,
                "stream": stream,
                "options": {
                    "temperature": config.temperature,
                    "top_p": config.top_p,
                    "top_k": config.top_k,
                    "num_predict": config.num_predict,
                    "stop": config.stop,
                    "seed": config.seed,
                    "num_ctx": config.num_ctx,
                    "repeat_penalty": config.repeat_penalty,
                    "mirostat": config.mirostat,
                    "mirostat_tau": config.mirostat_tau,
                    "mirostat_eta": config.mirostat_eta,
                },
            }

            if system:
                payload["system"] = system

            if format:
                payload["format"] = format

            if images:
                payload["images"] = images

            # Override options with any explicitly supplied kwargs (e.g., temperature)
            option_keys = {
                "temperature",
                "top_p",
                "top_k",
                "num_predict",
                "stop",
                "seed",
                "num_ctx",
                "repeat_penalty",
                "mirostat",
                "mirostat_tau",
                "mirostat_eta",
            }
            for key in list(extra_kwargs.keys()):
                if key in option_keys:
                    payload["options"][key] = extra_kwargs.pop(key)

            # If callers provided keep_alive or other top-level fields, merge them
            top_level_keys = {"keep_alive"}
            for key in list(extra_kwargs.keys()):
                if key in top_level_keys:
                    payload[key] = extra_kwargs.pop(key)

            # Warn about any unused kwargs to aid debugging
            if extra_kwargs:
                self.logger.debug(
                    "Ignoring unsupported generate() kwargs for model %s: %s", model, ", ".join(extra_kwargs.keys())
                )

            # Retry logic with intelligent error handling
            for attempt in range(self.max_retries):
                try:
                    return await self._make_generation_request(payload, stream, prompt_name)

                except OllamaHTTPError as e:
                    # Only retry on 5xx errors (server errors) or 429 (rate limit)
                    is_retryable = (e.status_code and e.status_code >= 500) or e.status_code == 429

                    if not is_retryable or attempt == self.max_retries - 1:
                        # Don't retry 4xx errors (client errors like 404, 400)
                        self.logger.error(f"HTTP {e.status_code} error (non-retryable): {e}")
                        raise

                    # Retryable error - log and retry
                    backoff = 2**attempt
                    self.logger.warning(
                        f"HTTP {e.status_code} error on attempt {attempt + 1}/{self.max_retries}, "
                        f"retrying in {backoff}s: {e}"
                    )
                    await asyncio.sleep(backoff)

                except (OllamaConnectionError, OllamaTimeoutError) as e:
                    # Connection/timeout errors are retryable
                    if attempt == self.max_retries - 1:
                        raise

                    backoff = 2**attempt
                    self.logger.warning(
                        f"Connection/timeout error on attempt {attempt + 1}/{self.max_retries}, "
                        f"retrying in {backoff}s: {e}"
                    )
                    await asyncio.sleep(backoff)

                except Exception as e:
                    # Unknown errors - retry but log clearly
                    if attempt == self.max_retries - 1:
                        self.logger.error(f"Unexpected error after {self.max_retries} attempts: {e}")
                        raise

                    self.logger.warning(f"Unexpected error on attempt {attempt + 1}, retrying: {e}")
                    await asyncio.sleep(2**attempt)

    async def _make_generation_request(
        self,
        payload: Dict,
        stream: bool,
        prompt_name: Optional[str] = None,
    ) -> Union[Dict, AsyncIterator]:
        """Make generation request with proper handling"""
        # Ensure session is connected
        if not self._session:
            await self.connect()

        start_time = datetime.now()

        # DEBUG: Log the prompt being sent to Ollama
        model = payload.get("model", "unknown")
        prompt_length = len(payload.get("prompt", ""))
        prompt_preview = payload.get("prompt", "")[:300]
        self.logger.info(
            "📤 Sending to Ollama API | server=%s | model=%s | prompt_name=%s | prompt_chars=%d | format=%s | stream=%s",
            self.base_url,
            model,
            prompt_name or "N/A",
            prompt_length,
            payload.get("format", "text"),
            stream,
        )
        self.logger.debug("  - Prompt preview: %s...", prompt_preview)
        if getattr(self, "verbose", False):
            self.logger.debug("  - Full prompt:\n%s", payload.get("prompt", ""))

        endpoint = f"{self.base_url}/api/generate"

        async with self._session.post(endpoint, json=payload) as response:
            # Fix: Check HTTP status before parsing JSON
            if response.status != 200:
                error_text = await response.text()
                raise OllamaHTTPError(
                    f"Generation failed for model '{model}': {error_text}",
                    status_code=response.status,
                    endpoint=endpoint,
                    model=model,
                )

            if stream:
                return self._stream_response(response, start_time, prompt_name, model)
            else:
                result = await response.json()

                # Track metrics
                self._update_metrics(result, start_time)

                # CRITICAL FIX: For reasoning models (qwen3, deepseek-r1), they return JSON in "thinking" field
                # The response field may be empty. We preserve BOTH fields and let downstream processors handle extraction.
                # DO NOT parse JSON here - let agents use llm_response_processor.py to extract JSON from markdown blocks

                response_text = result.get("response", "")
                thinking_text = result.get("thinking", "")

                # Preserve the raw response structure - DO NOT parse or convert to dict
                # The LLM response processor will handle JSON extraction from ```json code blocks

                # Determine model type and which field has the actual content
                is_reasoning_model = len(thinking_text) > 0 and len(response_text) == 0
                primary_field = "thinking" if is_reasoning_model else "response"
                primary_length = len(thinking_text) if is_reasoning_model else len(response_text)

                duration = (datetime.now() - start_time).total_seconds()
                self.logger.info(
                    "📥 Received from Ollama API | server=%s | model=%s | prompt_name=%s | duration=%.2fs | primary_field=%s | primary_chars=%d",
                    self.base_url,
                    model,
                    prompt_name or "N/A",
                    duration,
                    primary_field,
                    primary_length,
                )

                # DEBUG: Log what we're preserving with clear indication of model type
                self.logger.info(
                    "🔍 Ollama response structure | model_type=%s | response_chars=%d | thinking_chars=%d",
                    "Reasoning (thinking-based)" if is_reasoning_model else "Standard (response-based)",
                    len(response_text),
                    len(thinking_text),
                )

                # Show preview of the primary content
                if is_reasoning_model and thinking_text:
                    self.logger.debug("  - Content preview: %s...", thinking_text[:200])
                elif response_text:
                    self.logger.debug("  - Content preview: %s...", response_text[:200])

                # Store both in the result for downstream processing
                # Agents can use llm_response_processor.py to extract JSON from either field
                result["_raw_response"] = response_text
                result["_raw_thinking"] = thinking_text

                # For backward compatibility, keep 'response' as the primary field
                # If response is empty but thinking has content, use thinking as fallback
                if is_reasoning_model:
                    self.logger.info("🧠 Using 'thinking' field as primary response (reasoning model)")
                    result["response"] = thinking_text
                else:
                    result["response"] = response_text

                # DEBUG: Log exactly what we're returning
                self.logger.debug(
                    "🔍 Final Ollama API response structure | has_response=%s | has_model=%s | has_created_at=%s | response_type=%s | response_empty=%s | response_preview=%s...",
                    "response" in result,
                    "model" in result,
                    "created_at" in result,
                    type(result.get("response")),
                    not result.get("response"),
                    str(result.get("response"))[:200] if result.get("response") else "",
                )

                # Return FULL result with all Ollama metadata (model, created_at, total_duration, etc.)
                return result

    async def _stream_response(
        self,
        response: aiohttp.ClientResponse,
        start_time: datetime,
        prompt_name: Optional[str] = None,
        model: Optional[str] = None,
    ) -> AsyncIterator[Dict]:
        """Stream response from Ollama"""
        full_response = ""

        async for line in response.content:
            if line:
                chunk = json.loads(line)
                full_response += chunk.get("response", "")

                if chunk.get("done"):
                    # Final chunk with metrics
                    self._update_metrics(chunk, start_time)
                    duration = (datetime.now() - start_time).total_seconds()
                    self.logger.info(
                        "📥 Received streamed Ollama response | server=%s | model=%s | prompt_name=%s | duration=%.2fs",
                        self.base_url,
                        model or "unknown",
                        prompt_name or "N/A",
                        duration,
                    )

                    # Parse JSON if needed
                    if "format" in chunk and chunk["format"] == "json":
                        try:
                            chunk["full_response"] = json.loads(full_response)
                        except json.JSONDecodeError:
                            chunk["full_response"] = full_response

                yield chunk

    async def embeddings(self, model: str, prompt: str) -> List[float]:
        """Generate embeddings for text"""
        payload = {"model": model, "prompt": prompt}

        try:
            async with self._session.post(f"{self.base_url}/api/embeddings", json=payload) as response:
                result = await response.json()
                return result.get("embedding", [])
        except Exception as e:
            self.logger.error(f"Failed to generate embeddings: {e}")
            raise

    async def chat(
        self,
        model: str,
        messages: List[Dict],
        format: Optional[str] = None,
        config: Optional[ModelConfig] = None,
        stream: bool = False,
    ) -> Union[Dict, AsyncIterator[Dict]]:
        """
        Chat completion with conversation context

        Args:
            model: Model to use
            messages: List of message dicts with 'role' and 'content'
            format: Response format
            config: Model configuration
            stream: Whether to stream response
        """
        if not config:
            config = ModelConfig()

        payload = {
            "model": model,
            "messages": messages,
            "stream": stream,
            "options": {
                "temperature": config.temperature,
                "top_p": config.top_p,
                "top_k": config.top_k,
                "num_predict": config.num_predict,
            },
        }

        if format:
            payload["format"] = format

        try:
            async with self._session.post(f"{self.base_url}/api/chat", json=payload) as response:
                if stream:
                    return self._stream_chat_response(response)
                else:
                    result = await response.json()
                    return result.get("message", {}).get("content", "")
        except Exception as e:
            self.logger.error(f"Chat completion failed: {e}")
            raise

    async def _stream_chat_response(self, response: aiohttp.ClientResponse) -> AsyncIterator[Dict]:
        """Stream chat response"""
        async for line in response.content:
            if line:
                chunk = json.loads(line)
                yield chunk

    def _update_metrics(self, result: Dict, start_time: datetime):
        """Update performance metrics"""
        self.request_count += 1

        # Token count
        if "eval_count" in result:
            self.total_tokens += result["eval_count"]
        if "prompt_eval_count" in result:
            self.total_tokens += result["prompt_eval_count"]

        # Duration
        duration = (datetime.now() - start_time).total_seconds()
        self.total_duration += duration

    async def get_metrics(self) -> Dict:
        """Get client performance metrics"""
        avg_duration = self.total_duration / self.request_count if self.request_count > 0 else 0

        return {
            "request_count": self.request_count,
            "total_tokens": self.total_tokens,
            "total_duration": self.total_duration,
            "average_duration": avg_duration,
            "tokens_per_second": self.total_tokens / self.total_duration if self.total_duration > 0 else 0,
            "loaded_models": list(self.loaded_models),
        }

    async def unload_model(self, model: str):
        """Unload a model from memory"""
        payload = {"name": model}

        try:
            async with self._session.post(f"{self.base_url}/api/unload", json=payload) as response:
                if response.status == 200:
                    self.loaded_models.discard(model)
                    return True
                return False
        except Exception as e:
            self.logger.error(f"Failed to unload model {model}: {e}")
            return False

    async def model_info(self, model: str) -> Dict:
        """Get detailed model information"""
        payload = {"name": model}

        try:
            async with self._session.post(f"{self.base_url}/api/show", json=payload) as response:
                if response.status == 200:
                    return await response.json()
                return {}
        except Exception as e:
            self.logger.error(f"Failed to get model info for {model}: {e}")
            return {}

    async def copy_model(self, source: str, destination: str):
        """Copy/duplicate a model"""
        payload = {"source": source, "destination": destination}

        try:
            async with self._session.post(f"{self.base_url}/api/copy", json=payload) as response:
                return response.status == 200
        except Exception as e:
            self.logger.error(f"Failed to copy model {source} to {destination}: {e}")
            return False

    async def delete_model(self, model: str):
        """Delete a model"""
        payload = {"name": model}

        try:
            async with self._session.delete(f"{self.base_url}/api/delete", json=payload) as response:
                if response.status == 200:
                    self.loaded_models.discard(model)
                    self.model_info.pop(model, None)
                    return True
                return False
        except Exception as e:
            self.logger.error(f"Failed to delete model {model}: {e}")
            return False


class OllamaModelSelector:
    """
    Intelligent model selection based on task requirements
    """

    def __init__(self, client: OllamaClient, config=None):
        self.client = client
        self.config = config  # Store config for model_specs access

        # Model capabilities matrix (from actual ollama show output)
        # NOTE: This is now fallback - config.json model_specs takes priority
        self.model_capabilities = {
            # Premium models (70B+) - BEST FOR ACCURACY
            OllamaModel.LLAMA3_3_70B: {
                "reasoning": 0.99,  # HIGHEST - Latest llama3.3:70b
                "analysis": 0.99,
                "speed": 0.20,
                "context": 131072,  # 128K actual
                "languages": ["en", "es", "fr", "de", "it", "pt"],
                "params": "70.6B",
                "specialized": "comprehensive_analysis",
                "thinking": True,  # Native reasoning
            },
            OllamaModel.LLAMA3_3_70B_CUSTOM: {
                "reasoning": 0.98,
                "analysis": 0.98,
                "speed": 0.20,
                "context": 131072,  # 128K actual
                "languages": ["en", "es", "fr", "de", "it", "pt"],
                "params": "70.6B",
                "specialized": "comprehensive_analysis",
            },
            # Large reasoning models (30-40B)
            OllamaModel.DEEPSEEK_R1_32B: {
                "reasoning": 0.98,  # EXCELLENT reasoning with explicit thinking
                "analysis": 0.96,
                "speed": 0.35,
                "context": 131072,  # 128K actual
                "languages": ["en", "zh"],
                "params": "32.8B",
                "specialized": "reasoning",
                "thinking": True,  # Explicit thinking capability
            },
            OllamaModel.QWEN3_30B_MoE: {
                "reasoning": 0.97,  # MoE architecture
                "analysis": 0.97,
                "speed": 0.45,  # MoE is faster
                "context": 262144,  # 262K actual - MASSIVE!
                "languages": ["en", "zh", "ja", "ko", "es", "fr", "de"],
                "params": "30.5B",
                "specialized": "long_documents",
                "thinking": True,  # Has thinking capability
            },
            OllamaModel.QWEN3_32B: {
                "reasoning": 0.96,
                "analysis": 0.96,
                "speed": 0.40,
                "context": 40960,  # 40K actual
                "languages": ["en", "zh", "ja", "ko", "es", "fr", "de"],
                "params": "32.8B",
                "specialized": "long_documents",
            },
            OllamaModel.QWEN2_5_32B: {
                "reasoning": 0.95,
                "analysis": 0.95,
                "speed": 0.42,
                "context": 32768,  # 32K actual
                "languages": ["en", "zh", "ja", "ko"],
                "params": "32.8B",
            },
            OllamaModel.QWEN3_30B: {
                "reasoning": 0.95,
                "analysis": 0.95,
                "speed": 0.42,
                "context": 40960,  # 40K actual
                "languages": ["en", "zh", "ja", "ko", "es", "fr"],
                "params": "30.5B",
            },
            OllamaModel.QWEN3_CODER_30B: {
                "reasoning": 0.94,
                "analysis": 0.92,
                "speed": 0.40,
                "context": 262144,  # 262K actual - MASSIVE!
                "languages": ["en", "zh", "ja", "ko"],
                "params": "30.5B",
                "specialized": "code",
            },
            OllamaModel.DEEPSEEK_CODER_33B: {
                "reasoning": 0.93,
                "analysis": 0.90,
                "speed": 0.42,
                "context": 16384,  # 16K actual
                "languages": ["en", "zh"],
                "params": "33B",
                "specialized": "code",
            },
            OllamaModel.CODELLAMA_34B: {
                "reasoning": 0.88,
                "analysis": 0.85,
                "speed": 0.40,
                "context": 16384,  # 16K actual
                "languages": ["en"],
                "params": "34B",
                "specialized": "python",
            },
            # Medium models (8-27B)
            OllamaModel.GEMMA3_27B: {
                "reasoning": 0.90,
                "analysis": 0.88,
                "speed": 0.50,
                "context": 8192,  # Estimated
                "languages": ["en"],
                "params": "27B",
            },
            OllamaModel.PHI4_REASONING: {
                "reasoning": 0.92,
                "analysis": 0.90,
                "speed": 0.60,
                "context": 32768,  # 32K actual
                "languages": ["en"],
                "params": "14.7B",
                "specialized": "reasoning",
            },
            OllamaModel.PHI4_16K: {
                "reasoning": 0.91,
                "analysis": 0.89,
                "speed": 0.65,
                "context": 16384,  # 16K actual
                "languages": ["en"],
                "params": "14.7B",
            },
            OllamaModel.LLAMA3_1_8B: {
                "reasoning": 0.88,
                "analysis": 0.87,
                "speed": 0.75,
                "context": 131072,  # 128K actual
                "languages": ["en", "es", "fr", "de"],
                "params": "8.0B",
            },
            OllamaModel.MISTRAL_7B: {
                "reasoning": 0.85,
                "analysis": 0.83,
                "speed": 0.80,
                "context": 32768,  # Estimated
                "languages": ["en", "fr", "es", "de", "it"],
                "params": "7B",
            },
            # Fast inference models
            OllamaModel.LLAMA3_2: {
                "reasoning": 0.70,
                "analysis": 0.68,
                "speed": 0.95,
                "context": 131072,  # Large context despite small size
                "languages": ["en"],
                "params": "2.0B",
            },
            OllamaModel.PHI3_MINI: {
                "reasoning": 0.65,
                "analysis": 0.63,
                "speed": 0.98,
                "context": 4096,
                "languages": ["en"],
                "params": "3.8B",
            },
            # Specialized models
            OllamaModel.MIXTRAL_8X7B: {
                "reasoning": 0.94,
                "analysis": 0.92,
                "speed": 0.45,
                "context": 32768,  # 32K actual
                "languages": ["en", "fr", "es", "de", "it"],
                "params": "46.7B",
                "specialized": "mixture_of_experts",
            },
        }

    def select_model(self, task_type: str, requirements: Dict) -> str:
        """
        Select optimal model based on task requirements (updated for actual installed models)

        Args:
            task_type: Type of task (analysis, reasoning, extraction, code, etc.)
            requirements: Dict with requirements like speed, accuracy, context_length

        Returns:
            Optimal model name
        """
        priority = requirements.get("priority", "balanced")
        context_needed = requirements.get("context_length", 4096)

        # Speed-focused selection
        if priority == "speed":
            if context_needed > 100000:
                return OllamaModel.LLAMA3_2.value  # Fast + 128K context
            else:
                return OllamaModel.PHI3_MINI.value  # Fastest

        # Code-focused selection
        elif task_type == "code":
            if context_needed > 100000:
                return OllamaModel.QWEN3_CODER_30B.value  # 262K context!
            elif context_needed > 16000:
                return OllamaModel.DEEPSEEK_CODER_33B.value  # 16K context
            else:
                return OllamaModel.CODELLAMA_34B.value  # Python specialist

        # Maximum accuracy (premium models)
        elif priority == "accuracy":
            if context_needed > 100000:
                return OllamaModel.LLAMA3_3_70B.value  # Best + 128K
            elif context_needed > 40000:
                return OllamaModel.QWEN3_CODER_30B.value  # 262K massive context
            elif context_needed > 32000:
                return OllamaModel.QWEN3_32B.value  # 40K context
            else:
                return OllamaModel.QWEN3_32B.value  # Best general purpose

        # Balanced approach (default)
        else:
            if context_needed > 100000:
                # Need 100K+ context
                return OllamaModel.LLAMA3_1_8B.value  # 128K, fast
            elif context_needed > 40000:
                # Need 40K-100K context
                return OllamaModel.QWEN3_CODER_30B.value  # 262K, great for long docs
            elif context_needed > 32000:
                # Need 32K-40K context
                return OllamaModel.QWEN3_32B.value  # 40K, excellent quality
            elif context_needed > 16000:
                # Need 16K-32K context
                return OllamaModel.PHI4_REASONING.value  # 32K, good reasoning
            elif context_needed > 8000:
                # Need 8K-16K context
                return OllamaModel.PHI4_16K.value  # 16K, fast
            else:
                # Standard context (<8K)
                return OllamaModel.MISTRAL_7B.value  # Fast + good quality

    async def benchmark_models(self, prompt: str, models: Optional[List[str]] = None) -> Dict:
        """Benchmark multiple models on the same prompt"""
        if not models:
            models = [m.value for m in OllamaModel]

        results = {}

        for model in models:
            if model not in await self.client.loaded_models:
                continue

            start_time = datetime.now()

            try:
                response = await self.client.generate(
                    model=model, prompt=prompt, config=ModelConfig(temperature=0.7, num_predict=500)
                )

                duration = (datetime.now() - start_time).total_seconds()

                results[model] = {"duration": duration, "response_length": len(str(response)), "success": True}
            except Exception as e:
                results[model] = {"error": str(e), "success": False}

        return results
