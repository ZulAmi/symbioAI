#!/usr/bin/env python3
"""
Advanced LLM Integration Engine for Symbio AI

Production-grade integration with state-of-the-art language models including
OpenAI GPT, Anthropic Claude, Google Gemini, and HuggingFace transformers.
This module provides unified access to multiple LLM providers with advanced
caching, batching, and optimization strategies.
"""

import asyncio
import logging
import json
import time
from typing import Dict, Any, List, Optional, Union, Callable, AsyncGenerator
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from enum import Enum
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
import hashlib
import pickle
from pathlib import Path

# Core dependencies
import aiohttp
import torch
import transformers
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM,
    BitsAndBytesConfig, GenerationConfig, TextStreamer
)
from sentence_transformers import SentenceTransformer
import tiktoken

# Quantization and optimization
try:
    import bitsandbytes as bnb
    QUANTIZATION_AVAILABLE = True
except ImportError:
    QUANTIZATION_AVAILABLE = False

try:
    from optimum.bettertransformer import BetterTransformer
    OPTIMIZATION_AVAILABLE = True
except ImportError:
    OPTIMIZATION_AVAILABLE = False


class LLMProvider(Enum):
    """Supported LLM providers."""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    HUGGINGFACE = "huggingface"
    LOCAL = "local"
    CUSTOM = "custom"


class ModelSize(Enum):
    """Model size categories for optimization."""
    TINY = "tiny"      # < 1B parameters
    SMALL = "small"    # 1B - 7B parameters
    MEDIUM = "medium"  # 7B - 30B parameters
    LARGE = "large"    # 30B - 70B parameters
    XLARGE = "xlarge"  # > 70B parameters


@dataclass
class LLMConfig:
    """Configuration for LLM models."""
    provider: LLMProvider
    model_name: str
    model_size: ModelSize
    max_tokens: int = 4096
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    repetition_penalty: float = 1.1
    do_sample: bool = True
    use_cache: bool = True
    quantize: bool = True
    optimize: bool = True
    streaming: bool = False
    batch_size: int = 1
    api_key: Optional[str] = None
    api_base: Optional[str] = None
    timeout: int = 60
    retry_attempts: int = 3
    custom_params: Dict[str, Any] = field(default_factory=dict)


@dataclass 
class LLMResponse:
    """Response from LLM inference."""
    text: str
    tokens_used: int
    latency_ms: float
    model_id: str
    provider: LLMProvider
    cached: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseLLMProvider(ABC):
    """Abstract base class for LLM providers."""
    
    def __init__(self, config: LLMConfig):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self._cache: Dict[str, LLMResponse] = {}
        self._stats = {
            "total_requests": 0,
            "cache_hits": 0,
            "total_tokens": 0,
            "total_latency": 0.0
        }
    
    @abstractmethod
    async def generate(self, prompt: str, **kwargs) -> LLMResponse:
        """Generate text from prompt."""
        pass
    
    @abstractmethod
    async def generate_stream(self, prompt: str, **kwargs) -> AsyncGenerator[str, None]:
        """Generate text with streaming."""
        pass
    
    @abstractmethod
    async def embed(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for texts."""
        pass
    
    def _get_cache_key(self, prompt: str, **kwargs) -> str:
        """Generate cache key for request."""
        cache_data = {"prompt": prompt, **kwargs}
        cache_str = json.dumps(cache_data, sort_keys=True)
        return hashlib.md5(cache_str.encode()).hexdigest()
    
    def _update_stats(self, response: LLMResponse):
        """Update usage statistics."""
        self._stats["total_requests"] += 1
        self._stats["total_tokens"] += response.tokens_used
        self._stats["total_latency"] += response.latency_ms
        if response.cached:
            self._stats["cache_hits"] += 1
    
    def get_stats(self) -> Dict[str, Any]:
        """Get provider statistics."""
        stats = self._stats.copy()
        if stats["total_requests"] > 0:
            stats["avg_latency"] = stats["total_latency"] / stats["total_requests"]
            stats["cache_hit_rate"] = stats["cache_hits"] / stats["total_requests"]
        return stats


class OpenAIProvider(BaseLLMProvider):
    """OpenAI GPT provider implementation."""
    
    def __init__(self, config: LLMConfig):
        super().__init__(config)
        self.client = None
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
    
    async def _init_client(self):
        """Initialize OpenAI client."""
        if self.client is None:
            import openai
            self.client = openai.AsyncOpenAI(
                api_key=self.config.api_key,
                base_url=self.config.api_base,
                timeout=self.config.timeout
            )
    
    async def generate(self, prompt: str, **kwargs) -> LLMResponse:
        """Generate text using OpenAI API."""
        start_time = time.time()
        
        # Check cache
        cache_key = self._get_cache_key(prompt, **kwargs)
        if self.config.use_cache and cache_key in self._cache:
            response = self._cache[cache_key]
            response.cached = True
            self._update_stats(response)
            return response
        
        await self._init_client()
        
        try:
            # Prepare request
            messages = [{"role": "user", "content": prompt}]
            
            response = await self.client.chat.completions.create(
                model=self.config.model_name,
                messages=messages,
                max_tokens=kwargs.get("max_tokens", self.config.max_tokens),
                temperature=kwargs.get("temperature", self.config.temperature),
                top_p=kwargs.get("top_p", self.config.top_p),
                stream=False
            )
            
            # Process response
            text = response.choices[0].message.content
            tokens_used = response.usage.total_tokens
            latency_ms = (time.time() - start_time) * 1000
            
            llm_response = LLMResponse(
                text=text,
                tokens_used=tokens_used,
                latency_ms=latency_ms,
                model_id=self.config.model_name,
                provider=LLMProvider.OPENAI,
                metadata={"finish_reason": response.choices[0].finish_reason}
            )
            
            # Cache response
            if self.config.use_cache:
                self._cache[cache_key] = llm_response
            
            self._update_stats(llm_response)
            return llm_response
            
        except Exception as e:
            self.logger.error(f"OpenAI API error: {e}")
            raise
    
    async def generate_stream(self, prompt: str, **kwargs) -> AsyncGenerator[str, None]:
        """Generate streaming text using OpenAI API."""
        await self._init_client()
        
        messages = [{"role": "user", "content": prompt}]
        
        async with self.client.chat.completions.create(
            model=self.config.model_name,
            messages=messages,
            max_tokens=kwargs.get("max_tokens", self.config.max_tokens),
            temperature=kwargs.get("temperature", self.config.temperature),
            stream=True
        ) as stream:
            async for chunk in stream:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
    
    async def embed(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings using OpenAI API."""
        await self._init_client()
        
        response = await self.client.embeddings.create(
            model="text-embedding-ada-002",
            input=texts
        )
        
        return [data.embedding for data in response.data]


class AnthropicProvider(BaseLLMProvider):
    """Anthropic Claude provider implementation."""
    
    def __init__(self, config: LLMConfig):
        super().__init__(config)
        self.client = None
    
    async def _init_client(self):
        """Initialize Anthropic client."""
        if self.client is None:
            import anthropic
            self.client = anthropic.AsyncAnthropic(
                api_key=self.config.api_key,
                timeout=self.config.timeout
            )
    
    async def generate(self, prompt: str, **kwargs) -> LLMResponse:
        """Generate text using Anthropic API."""
        start_time = time.time()
        
        # Check cache
        cache_key = self._get_cache_key(prompt, **kwargs)
        if self.config.use_cache and cache_key in self._cache:
            response = self._cache[cache_key]
            response.cached = True
            self._update_stats(response)
            return response
        
        await self._init_client()
        
        try:
            response = await self.client.messages.create(
                model=self.config.model_name,
                max_tokens=kwargs.get("max_tokens", self.config.max_tokens),
                temperature=kwargs.get("temperature", self.config.temperature),
                messages=[{"role": "user", "content": prompt}]
            )
            
            text = response.content[0].text
            tokens_used = response.usage.input_tokens + response.usage.output_tokens
            latency_ms = (time.time() - start_time) * 1000
            
            llm_response = LLMResponse(
                text=text,
                tokens_used=tokens_used,
                latency_ms=latency_ms,
                model_id=self.config.model_name,
                provider=LLMProvider.ANTHROPIC,
                metadata={"stop_reason": response.stop_reason}
            )
            
            if self.config.use_cache:
                self._cache[cache_key] = llm_response
            
            self._update_stats(llm_response)
            return llm_response
            
        except Exception as e:
            self.logger.error(f"Anthropic API error: {e}")
            raise
    
    async def generate_stream(self, prompt: str, **kwargs) -> AsyncGenerator[str, None]:
        """Generate streaming text using Anthropic API."""
        await self._init_client()
        
        async with self.client.messages.stream(
            model=self.config.model_name,
            max_tokens=kwargs.get("max_tokens", self.config.max_tokens),
            messages=[{"role": "user", "content": prompt}]
        ) as stream:
            async for text in stream.text_stream:
                yield text
    
    async def embed(self, texts: List[str]) -> List[List[float]]:
        """Anthropic doesn't provide embeddings - use sentence transformers."""
        model = SentenceTransformer('all-MiniLM-L6-v2')
        embeddings = model.encode(texts)
        return embeddings.tolist()


class HuggingFaceProvider(BaseLLMProvider):
    """HuggingFace transformers provider implementation."""
    
    def __init__(self, config: LLMConfig):
        super().__init__(config)
        self.model = None
        self.tokenizer = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.embedding_model = None
    
    async def _init_model(self):
        """Initialize HuggingFace model."""
        if self.model is None:
            self.logger.info(f"Loading HuggingFace model: {self.config.model_name}")
            
            # Configure quantization if available
            quantization_config = None
            if self.config.quantize and QUANTIZATION_AVAILABLE and torch.cuda.is_available():
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4"
                )
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.model_name,
                trust_remote_code=True,
                padding_side="left"
            )
            
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load model
            model_kwargs = {
                "torch_dtype": torch.float16 if torch.cuda.is_available() else torch.float32,
                "low_cpu_mem_usage": True,
                "trust_remote_code": True
            }
            
            if quantization_config:
                model_kwargs["quantization_config"] = quantization_config
            else:
                model_kwargs["device_map"] = "auto"
            
            try:
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.config.model_name,
                    **model_kwargs
                )
            except:
                # Fallback for seq2seq models
                self.model = AutoModelForSeq2SeqLM.from_pretrained(
                    self.config.model_name,
                    **model_kwargs
                )
            
            # Optimize model if available
            if self.config.optimize and OPTIMIZATION_AVAILABLE:
                try:
                    self.model = BetterTransformer.transform(self.model)
                    self.logger.info("Applied BetterTransformer optimization")
                except:
                    self.logger.warning("BetterTransformer optimization failed")
            
            self.model.eval()
            self.logger.info(f"Model loaded successfully on {self.device}")

    async def apply_lora_adapter(self, adapter_dir: str) -> bool:
        """
        Apply a LoRA adapter to the loaded model at runtime.
        Returns True on success, False otherwise.
        """
        try:
            import importlib
            peft = importlib.import_module("peft")
            PeftModel = getattr(peft, "PeftModel")
        except Exception:
            self.logger.error("peft is not installed; cannot apply LoRA adapter")
            return False

        await self._init_model()
        try:
            self.model = PeftModel.from_pretrained(self.model, adapter_dir)
            self.model.to(self.device)
            self.model.eval()
            self.logger.info(f"Applied LoRA adapter from {adapter_dir}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to apply LoRA adapter: {e}")
            return False
    
    async def generate(self, prompt: str, **kwargs) -> LLMResponse:
        """Generate text using HuggingFace model."""
        start_time = time.time()
        
        # Check cache
        cache_key = self._get_cache_key(prompt, **kwargs)
        if self.config.use_cache and cache_key in self._cache:
            response = self._cache[cache_key]
            response.cached = True
            self._update_stats(response)
            return response
        
        await self._init_model()
        
        try:
            # Tokenize input
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.config.max_tokens
            ).to(self.device)
            
            # Generation configuration
            generation_config = GenerationConfig(
                max_new_tokens=kwargs.get("max_tokens", 1024),
                temperature=kwargs.get("temperature", self.config.temperature),
                top_p=kwargs.get("top_p", self.config.top_p),
                top_k=kwargs.get("top_k", self.config.top_k),
                repetition_penalty=kwargs.get("repetition_penalty", self.config.repetition_penalty),
                do_sample=self.config.do_sample,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                use_cache=True
            )
            
            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    generation_config=generation_config
                )
            
            # Decode response
            generated_tokens = outputs[0][inputs.input_ids.shape[1]:]
            text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
            
            tokens_used = len(outputs[0])
            latency_ms = (time.time() - start_time) * 1000
            
            llm_response = LLMResponse(
                text=text.strip(),
                tokens_used=tokens_used,
                latency_ms=latency_ms,
                model_id=self.config.model_name,
                provider=LLMProvider.HUGGINGFACE,
                metadata={"device": str(self.device)}
            )
            
            if self.config.use_cache:
                self._cache[cache_key] = llm_response
            
            self._update_stats(llm_response)
            return llm_response
            
        except Exception as e:
            self.logger.error(f"HuggingFace generation error: {e}")
            raise
    
    async def generate_stream(self, prompt: str, **kwargs) -> AsyncGenerator[str, None]:
        """Generate streaming text using HuggingFace model."""
        await self._init_model()
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        streamer = TextStreamer(
            self.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True
        )
        
        generation_config = GenerationConfig(
            max_new_tokens=kwargs.get("max_tokens", 1024),
            temperature=kwargs.get("temperature", self.config.temperature),
            do_sample=self.config.do_sample,
            streamer=streamer
        )
        
        # Note: This is a simplified streaming implementation
        # In production, you'd want proper async streaming
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                generation_config=generation_config
            )
        
        generated_tokens = outputs[0][inputs.input_ids.shape[1]:]
        text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        
        # Simulate streaming by yielding chunks
        words = text.split()
        for word in words:
            yield word + " "
            await asyncio.sleep(0.01)  # Small delay for streaming effect
    
    async def embed(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings using sentence transformers."""
        if self.embedding_model is None:
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        embeddings = self.embedding_model.encode(texts)
        return embeddings.tolist()


class LLMManager:
    """
    Advanced LLM management system for Symbio AI.
    
    Provides unified access to multiple LLM providers with intelligent
    routing, load balancing, and optimization strategies.
    """
    
    def __init__(self):
        self.providers: Dict[str, BaseLLMProvider] = {}
        self.default_provider: Optional[str] = None
        self.logger = logging.getLogger(__name__)
        self._executor = ThreadPoolExecutor(max_workers=4)
        
        # Routing strategies
        self.routing_strategies = {
            "round_robin": self._route_round_robin,
            "least_latency": self._route_least_latency,
            "best_quality": self._route_best_quality,
            "cost_optimized": self._route_cost_optimized
        }
        self.current_strategy = "least_latency"
        self._round_robin_index = 0
    
    def register_provider(self, name: str, provider: BaseLLMProvider, default: bool = False):
        """Register an LLM provider."""
        self.providers[name] = provider
        if default or self.default_provider is None:
            self.default_provider = name
        self.logger.info(f"Registered LLM provider: {name}")
    
    def add_openai_provider(self, name: str, model_name: str, api_key: str, **kwargs):
        """Add OpenAI provider."""
        config = LLMConfig(
            provider=LLMProvider.OPENAI,
            model_name=model_name,
            model_size=self._infer_model_size(model_name),
            api_key=api_key,
            **kwargs
        )
        provider = OpenAIProvider(config)
        self.register_provider(name, provider)
    
    def add_anthropic_provider(self, name: str, model_name: str, api_key: str, **kwargs):
        """Add Anthropic provider."""
        config = LLMConfig(
            provider=LLMProvider.ANTHROPIC,
            model_name=model_name,
            model_size=self._infer_model_size(model_name),
            api_key=api_key,
            **kwargs
        )
        provider = AnthropicProvider(config)
        self.register_provider(name, provider)
    
    def add_huggingface_provider(self, name: str, model_name: str, **kwargs):
        """Add HuggingFace provider."""
        config = LLMConfig(
            provider=LLMProvider.HUGGINGFACE,
            model_name=model_name,
            model_size=self._infer_model_size(model_name),
            **kwargs
        )
        provider = HuggingFaceProvider(config)
        self.register_provider(name, provider)
    
    async def generate(self, prompt: str, provider_name: Optional[str] = None, **kwargs) -> LLMResponse:
        """Generate text using specified or automatically selected provider."""
        if provider_name is None:
            provider_name = self._route_request(prompt, **kwargs)
        
        if provider_name not in self.providers:
            raise ValueError(f"Provider {provider_name} not found")
        
        provider = self.providers[provider_name]
        return await provider.generate(prompt, **kwargs)
    
    async def generate_batch(self, prompts: List[str], **kwargs) -> List[LLMResponse]:
        """Generate text for multiple prompts in parallel."""
        tasks = []
        for prompt in prompts:
            provider_name = self._route_request(prompt, **kwargs)
            task = self.generate(prompt, provider_name, **kwargs)
            tasks.append(task)
        
        return await asyncio.gather(*tasks)
    
    async def generate_stream(self, prompt: str, provider_name: Optional[str] = None, **kwargs) -> AsyncGenerator[str, None]:
        """Generate streaming text."""
        if provider_name is None:
            provider_name = self._route_request(prompt, **kwargs)
        
        provider = self.providers[provider_name]
        async for chunk in provider.generate_stream(prompt, **kwargs):
            yield chunk
    
    async def embed(self, texts: List[str], provider_name: Optional[str] = None) -> List[List[float]]:
        """Generate embeddings."""
        if provider_name is None:
            provider_name = self.default_provider
        
        provider = self.providers[provider_name]
        return await provider.embed(texts)
    
    def _route_request(self, prompt: str, **kwargs) -> str:
        """Route request to appropriate provider."""
        strategy = self.routing_strategies.get(self.current_strategy)
        if strategy:
            return strategy(prompt, **kwargs)
        return self.default_provider
    
    def _route_round_robin(self, prompt: str, **kwargs) -> str:
        """Round-robin routing strategy."""
        providers = list(self.providers.keys())
        provider = providers[self._round_robin_index % len(providers)]
        self._round_robin_index += 1
        return provider
    
    def _route_least_latency(self, prompt: str, **kwargs) -> str:
        """Route to provider with lowest average latency."""
        best_provider = self.default_provider
        best_latency = float('inf')
        
        for name, provider in self.providers.items():
            stats = provider.get_stats()
            avg_latency = stats.get("avg_latency", float('inf'))
            if avg_latency < best_latency:
                best_latency = avg_latency
                best_provider = name
        
        return best_provider
    
    def _route_best_quality(self, prompt: str, **kwargs) -> str:
        """Route to highest quality provider based on model size."""
        # Prioritize larger models for complex tasks
        if len(prompt) > 1000 or kwargs.get("temperature", 0.7) < 0.3:
            # Complex/precise task - use larger model
            for name, provider in self.providers.items():
                if provider.config.model_size in [ModelSize.LARGE, ModelSize.XLARGE]:
                    return name
        
        return self.default_provider
    
    def _route_cost_optimized(self, prompt: str, **kwargs) -> str:
        """Route to most cost-effective provider."""
        # Prefer local/HuggingFace models for cost optimization
        for name, provider in self.providers.items():
            if provider.config.provider in [LLMProvider.HUGGINGFACE, LLMProvider.LOCAL]:
                return name
        
        return self.default_provider
    
    def _infer_model_size(self, model_name: str) -> ModelSize:
        """Infer model size from model name."""
        model_name_lower = model_name.lower()
        
        if any(x in model_name_lower for x in ["70b", "65b", "175b"]):
            return ModelSize.XLARGE
        elif any(x in model_name_lower for x in ["30b", "33b", "40b"]):
            return ModelSize.LARGE
        elif any(x in model_name_lower for x in ["7b", "13b", "20b"]):
            return ModelSize.MEDIUM
        elif any(x in model_name_lower for x in ["1b", "3b", "6b"]):
            return ModelSize.SMALL
        else:
            return ModelSize.TINY
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Get comprehensive system statistics."""
        stats = {
            "total_providers": len(self.providers),
            "default_provider": self.default_provider,
            "routing_strategy": self.current_strategy,
            "providers": {}
        }
        
        for name, provider in self.providers.items():
            stats["providers"][name] = {
                "config": {
                    "provider": provider.config.provider.value,
                    "model_name": provider.config.model_name,
                    "model_size": provider.config.model_size.value
                },
                "stats": provider.get_stats()
            }
        
        return stats
    
    async def benchmark_providers(self, test_prompts: List[str]) -> Dict[str, Any]:
        """Benchmark all providers with test prompts."""
        results = {}
        
        for name, provider in self.providers.items():
            self.logger.info(f"Benchmarking provider: {name}")
            provider_results = {
                "responses": [],
                "avg_latency": 0,
                "total_tokens": 0,
                "errors": 0
            }
            
            for prompt in test_prompts:
                try:
                    response = await provider.generate(prompt, max_tokens=100)
                    provider_results["responses"].append({
                        "prompt": prompt[:50] + "...",
                        "response": response.text[:100] + "...",
                        "latency_ms": response.latency_ms,
                        "tokens": response.tokens_used
                    })
                    provider_results["total_tokens"] += response.tokens_used
                except Exception as e:
                    provider_results["errors"] += 1
                    self.logger.error(f"Benchmark error for {name}: {e}")
            
            if provider_results["responses"]:
                latencies = [r["latency_ms"] for r in provider_results["responses"]]
                provider_results["avg_latency"] = sum(latencies) / len(latencies)
            
            results[name] = provider_results
        
        return results
    
    def set_routing_strategy(self, strategy: str):
        """Set the routing strategy."""
        if strategy in self.routing_strategies:
            self.current_strategy = strategy
            self.logger.info(f"Routing strategy set to: {strategy}")
        else:
            raise ValueError(f"Unknown routing strategy: {strategy}")


# Convenience functions for easy setup
def create_production_llm_manager() -> LLMManager:
    """Create a production-ready LLM manager with multiple providers."""
    manager = LLMManager()
    
    # Add providers based on available API keys
    import os
    
    if os.getenv("OPENAI_API_KEY"):
        manager.add_openai_provider(
            "gpt-4",
            "gpt-4-turbo-preview",
            os.getenv("OPENAI_API_KEY"),
            default=True
        )
        manager.add_openai_provider(
            "gpt-3.5",
            "gpt-3.5-turbo",
            os.getenv("OPENAI_API_KEY")
        )
    
    if os.getenv("ANTHROPIC_API_KEY"):
        manager.add_anthropic_provider(
            "claude-3",
            "claude-3-opus-20240229",
            os.getenv("ANTHROPIC_API_KEY")
        )
    
    # Add open-source models
    try:
        manager.add_huggingface_provider(
            "llama-2-7b",
            "meta-llama/Llama-2-7b-chat-hf",
            quantize=True,
            optimize=True
        )
    except:
        pass  # Model might not be available
    
    try:
        manager.add_huggingface_provider(
            "mistral-7b",
            "mistralai/Mistral-7B-Instruct-v0.1",
            quantize=True,
            optimize=True
        )
    except:
        pass  # Model might not be available
    
    return manager


async def test_llm_integration():
    """Test the LLM integration system."""
    print("🚀 Testing Advanced LLM Integration")
    print("=" * 50)
    
    manager = create_production_llm_manager()
    
    if not manager.providers:
        print("❌ No LLM providers configured. Please set API keys or install models.")
        return
    
    print(f"✅ Initialized with {len(manager.providers)} providers")
    
    # Test basic generation
    test_prompt = "Explain the concept of evolutionary algorithms in AI in one paragraph."
    
    try:
        response = await manager.generate(test_prompt, max_tokens=150)
        print(f"\n📝 Generated Response:")
        print(f"Provider: {response.provider.value}")
        print(f"Model: {response.model_id}")
        print(f"Tokens: {response.tokens_used}")
        print(f"Latency: {response.latency_ms:.2f}ms")
        print(f"Cached: {response.cached}")
        print(f"Text: {response.text}")
        
    except Exception as e:
        print(f"❌ Generation failed: {e}")
    
    # Test batch generation
    batch_prompts = [
        "What is machine learning?",
        "Explain neural networks.",
        "Define artificial intelligence."
    ]
    
    try:
        batch_responses = await manager.generate_batch(batch_prompts, max_tokens=50)
        print(f"\n📊 Batch Generation Results:")
        for i, response in enumerate(batch_responses):
            print(f"  {i+1}. {response.text[:100]}... ({response.latency_ms:.2f}ms)")
    except Exception as e:
        print(f"❌ Batch generation failed: {e}")
    
    # Show system stats
    stats = manager.get_system_stats()
    print(f"\n📈 System Statistics:")
    print(f"Total Providers: {stats['total_providers']}")
    print(f"Default Provider: {stats['default_provider']}")
    print(f"Routing Strategy: {stats['routing_strategy']}")
    
    print("\n✅ LLM Integration Test Complete!")


if __name__ == "__main__":
    asyncio.run(test_llm_integration())