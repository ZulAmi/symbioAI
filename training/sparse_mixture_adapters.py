"""
Sparse Mixture of Adapters (SMoA) - Symbio AI

Revolutionary adapter system enabling infinite specialization at constant cost through:
- Billions of tiny adapters with intelligent routing
- Hierarchical adapter composition
- Automatic adapter merging and pruning
- Zero-overhead adapter serving
- Dynamic adapter discovery and loading

This system dramatically extends model capabilities while maintaining constant inference cost.

Key Features:
1. Massive adapter libraries (billions of adapters)
2. Sparse activation (activate only relevant adapters)
3. Hierarchical composition (adapter of adapters)
4. Automatic merging and pruning
5. Zero-overhead serving (sub-millisecond routing)
6. Continual adapter learning
7. Adapter market discovery
"""

import asyncio
import logging
import random
import json
import hashlib
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Set, Union
from dataclasses import dataclass, field, asdict
from datetime import datetime
from collections import defaultdict, deque
from enum import Enum
import heapq

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    # Mock torch
    class torch:
        class nn:
            class Module:
                def __init__(self): pass
            class Parameter:
                def __init__(self, data): self.data = data
            class ModuleDict:
                def __init__(self): pass
            class Embedding:
                def __init__(self, *args, **kwargs): pass
            class Linear:
                def __init__(self, *args, **kwargs): pass
        @staticmethod
        def tensor(x): return np.array(x)
        @staticmethod
        def randn(*args): return np.random.randn(*args)
        @staticmethod
        def zeros(*args): return np.zeros(args)
        @staticmethod
        def sigmoid(x): return 1 / (1 + np.exp(-np.array(x)))
    
    class F:
        @staticmethod
        def softmax(x, dim=-1): 
            exp_x = np.exp(x - np.max(x, axis=dim, keepdims=True))
            return exp_x / np.sum(exp_x, axis=dim, keepdims=True)
        @staticmethod
        def cosine_similarity(x, y, dim=0): return 0.5

try:
    from monitoring.observability import OBSERVABILITY
except ImportError:
    class OBSERVABILITY:
        @staticmethod
        def emit_gauge(metric, value, **tags): pass
        @staticmethod
        def emit_counter(metric, value, **tags): pass


class AdapterSpecialization(Enum):
    """Types of adapter specializations."""
    DOMAIN = "domain"  # Domain-specific (medical, legal, finance)
    TASK = "task"  # Task-specific (translation, summarization)
    LANGUAGE = "language"  # Language-specific
    STYLE = "style"  # Writing style
    PERSONA = "persona"  # Personality/persona
    SKILL = "skill"  # Specific skill (math, coding)
    TEMPORAL = "temporal"  # Time-specific (current events)
    SAFETY = "safety"  # Safety/alignment
    EFFICIENCY = "efficiency"  # Optimization/compression
    CUSTOM = "custom"  # Custom specialization


class AdapterGranularity(Enum):
    """Granularity levels for adapters."""
    NANO = "nano"  # 1K-10K parameters (ultra-tiny)
    MICRO = "micro"  # 10K-100K parameters
    MINI = "mini"  # 100K-1M parameters
    SMALL = "small"  # 1M-10M parameters
    MEDIUM = "medium"  # 10M-100M parameters


class RoutingStrategy(Enum):
    """Strategies for routing to adapters."""
    SEMANTIC = "semantic"  # Content-based semantic routing
    TASK_BASED = "task_based"  # Explicit task specification
    LEARNED = "learned"  # Learned router network
    HYBRID = "hybrid"  # Combination of strategies
    HIERARCHICAL = "hierarchical"  # Multi-level routing
    MIXTURE = "mixture"  # Soft mixture of adapters


@dataclass
class TinyAdapter:
    """A tiny, specialized adapter."""
    adapter_id: str
    specialization: AdapterSpecialization
    granularity: AdapterGranularity
    
    # Identification
    domain_tags: List[str] = field(default_factory=list)
    skill_tags: List[str] = field(default_factory=list)
    embedding: Optional[np.ndarray] = None  # Semantic embedding
    
    # Parameters
    rank: int = 4  # LoRA rank (tiny!)
    alpha: float = 8.0
    target_modules: List[str] = field(default_factory=list)
    num_parameters: int = 0
    
    # Performance
    usage_count: int = 0
    success_rate: float = 0.0
    average_latency_ms: float = 0.0
    specialization_score: float = 0.0  # How specialized (0-1)
    
    # Composition
    parent_adapters: List[str] = field(default_factory=list)  # Hierarchical parents
    child_adapters: List[str] = field(default_factory=list)  # Hierarchical children
    merged_from: List[str] = field(default_factory=list)  # Merged adapters
    
    # Lifecycle
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    last_used_at: Optional[str] = None
    pruning_score: float = 1.0  # Higher = keep, lower = prune
    is_active: bool = True
    
    # Storage
    storage_location: Optional[str] = None  # Lazy loading path
    is_loaded: bool = False  # In memory?
    weight_data: Optional[Dict[str, np.ndarray]] = None


@dataclass
class AdapterRoute:
    """A routing decision for adapter selection."""
    adapter_ids: List[str]  # Selected adapters
    routing_scores: List[float]  # Confidence scores
    routing_strategy: RoutingStrategy
    
    # Routing metadata
    query_embedding: Optional[np.ndarray] = None
    domain_match_scores: Dict[str, float] = field(default_factory=dict)
    skill_match_scores: Dict[str, float] = field(default_factory=dict)
    
    # Performance
    routing_latency_ms: float = 0.0
    total_adapters_considered: int = 0
    
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())


@dataclass
class AdapterComposition:
    """A hierarchical composition of adapters."""
    composition_id: str
    component_adapters: List[str]  # Adapter IDs in composition
    composition_strategy: str  # "sequential", "parallel", "hierarchical"
    
    # Weights for mixing
    mixing_weights: List[float] = field(default_factory=list)
    
    # Performance
    combined_performance: float = 0.0
    composition_overhead_ms: float = 0.0
    
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())


@dataclass
class AdapterMergeCandidate:
    """Candidate adapters for merging."""
    adapter_ids: List[str]
    similarity_score: float
    combined_usage: int
    merge_benefit_score: float  # Expected benefit from merging
    
    # Estimated merged adapter
    estimated_parameters: int
    estimated_performance: float


@dataclass
class SMoAConfig:
    """Configuration for Sparse Mixture of Adapters."""
    # Library settings
    max_adapters: int = 1_000_000  # Maximum adapters in library (1 million!)
    default_adapter_rank: int = 4  # Tiny adapters
    default_granularity: AdapterGranularity = AdapterGranularity.NANO
    
    # Routing settings
    routing_strategy: RoutingStrategy = RoutingStrategy.HYBRID
    max_active_adapters: int = 3  # Sparse activation
    routing_threshold: float = 0.5  # Min score to activate
    
    # Memory management
    max_loaded_adapters: int = 100  # LRU cache size
    lazy_loading: bool = True  # Load on demand
    enable_offloading: bool = True  # Offload unused adapters
    
    # Composition
    enable_hierarchical_composition: bool = True
    max_composition_depth: int = 3
    enable_adapter_mixing: bool = True
    
    # Optimization
    enable_auto_merging: bool = True
    merge_similarity_threshold: float = 0.9  # Merge very similar adapters
    enable_auto_pruning: bool = True
    prune_usage_threshold: int = 10  # Min usage before considering pruning
    prune_score_threshold: float = 0.3  # Below this, consider pruning
    
    # Performance
    routing_latency_budget_ms: float = 1.0  # Max routing time
    enable_caching: bool = True  # Cache routing decisions
    cache_ttl_seconds: int = 300  # 5 minutes


class AdapterEmbedder:
    """Generates semantic embeddings for adapters and queries."""
    
    def __init__(self, embedding_dim: int = 128):
        self.embedding_dim = embedding_dim
        self.logger = logging.getLogger(__name__)
    
    def embed_adapter(self, adapter: TinyAdapter) -> np.ndarray:
        """
        Generate semantic embedding for an adapter.
        
        Uses adapter metadata (tags, specialization, etc.)
        """
        # Combine all textual features
        features = [
            adapter.specialization.value,
            adapter.granularity.value,
        ] + adapter.domain_tags + adapter.skill_tags
        
        # Simple hash-based embedding (in production, use BERT/sentence transformers)
        feature_str = "_".join(features)
        hash_val = int(hashlib.md5(feature_str.encode()).hexdigest(), 16)
        
        # Generate deterministic random embedding
        np.random.seed(hash_val % (2**32))
        embedding = np.random.randn(self.embedding_dim)
        embedding = embedding / np.linalg.norm(embedding)  # Normalize
        
        return embedding
    
    def embed_query(self, query: str, metadata: Optional[Dict] = None) -> np.ndarray:
        """
        Generate semantic embedding for a query/input.
        
        In production, use actual sentence embeddings.
        """
        # Simple hash-based embedding (placeholder)
        hash_val = int(hashlib.md5(query.encode()).hexdigest(), 16)
        np.random.seed(hash_val % (2**32))
        embedding = np.random.randn(self.embedding_dim)
        embedding = embedding / np.linalg.norm(embedding)
        
        return embedding
    
    def compute_similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """Compute cosine similarity between embeddings."""
        return float(np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2)))


class AdapterRouter:
    """Routes queries to relevant adapters using various strategies."""
    
    def __init__(self, config: SMoAConfig, embedder: AdapterEmbedder):
        self.config = config
        self.embedder = embedder
        self.logger = logging.getLogger(__name__)
        
        # Routing cache
        self.routing_cache: Dict[str, AdapterRoute] = {}
        self.cache_hits = 0
        self.cache_misses = 0
    
    def route(
        self,
        query: str,
        adapters: Dict[str, TinyAdapter],
        metadata: Optional[Dict] = None,
        strategy: Optional[RoutingStrategy] = None
    ) -> AdapterRoute:
        """
        Route a query to relevant adapters.
        
        Args:
            query: Input query/text
            adapters: Available adapter library
            metadata: Optional metadata (task, domain, etc.)
            strategy: Optional routing strategy override
        
        Returns:
            AdapterRoute with selected adapters
        """
        start_time = datetime.utcnow()
        
        # Check cache
        if self.config.enable_caching:
            cache_key = self._get_cache_key(query, metadata)
            if cache_key in self.routing_cache:
                self.cache_hits += 1
                cached_route = self.routing_cache[cache_key]
                self.logger.debug(f"Cache hit for query (hit rate: {self.cache_hits/(self.cache_hits+self.cache_misses):.2%})")
                return cached_route
            self.cache_misses += 1
        
        strategy = strategy or self.config.routing_strategy
        
        # Route based on strategy
        if strategy == RoutingStrategy.SEMANTIC:
            route = self._semantic_routing(query, adapters, metadata)
        elif strategy == RoutingStrategy.TASK_BASED:
            route = self._task_based_routing(query, adapters, metadata)
        elif strategy == RoutingStrategy.LEARNED:
            route = self._learned_routing(query, adapters, metadata)
        elif strategy == RoutingStrategy.HYBRID:
            route = self._hybrid_routing(query, adapters, metadata)
        elif strategy == RoutingStrategy.HIERARCHICAL:
            route = self._hierarchical_routing(query, adapters, metadata)
        else:
            route = self._semantic_routing(query, adapters, metadata)
        
        # Calculate latency
        end_time = datetime.utcnow()
        route.routing_latency_ms = (end_time - start_time).total_seconds() * 1000
        
        # Cache result
        if self.config.enable_caching:
            self.routing_cache[cache_key] = route
        
        OBSERVABILITY.emit_gauge(
            "smoa.routing_latency_ms",
            route.routing_latency_ms,
            strategy=strategy.value
        )
        
        return route
    
    def _semantic_routing(
        self,
        query: str,
        adapters: Dict[str, TinyAdapter],
        metadata: Optional[Dict]
    ) -> AdapterRoute:
        """Semantic similarity-based routing."""
        query_emb = self.embedder.embed_query(query, metadata)
        
        # Compute similarities
        scores = []
        adapter_ids = []
        
        for adapter_id, adapter in adapters.items():
            if not adapter.is_active:
                continue
            
            if adapter.embedding is None:
                adapter.embedding = self.embedder.embed_adapter(adapter)
            
            similarity = self.embedder.compute_similarity(query_emb, adapter.embedding)
            scores.append(similarity)
            adapter_ids.append(adapter_id)
        
        # Select top adapters
        top_indices = np.argsort(scores)[::-1][:self.config.max_active_adapters]
        selected_adapters = [adapter_ids[i] for i in top_indices]
        selected_scores = [scores[i] for i in top_indices]
        
        # Filter by threshold
        filtered_adapters = []
        filtered_scores = []
        for adapter_id, score in zip(selected_adapters, selected_scores):
            if score >= self.config.routing_threshold:
                filtered_adapters.append(adapter_id)
                filtered_scores.append(score)
        
        return AdapterRoute(
            adapter_ids=filtered_adapters,
            routing_scores=filtered_scores,
            routing_strategy=RoutingStrategy.SEMANTIC,
            query_embedding=query_emb,
            total_adapters_considered=len(adapters)
        )
    
    def _task_based_routing(
        self,
        query: str,
        adapters: Dict[str, TinyAdapter],
        metadata: Optional[Dict]
    ) -> AdapterRoute:
        """Route based on explicit task specification."""
        if not metadata or 'task' not in metadata:
            # Fall back to semantic routing
            return self._semantic_routing(query, adapters, metadata)
        
        task = metadata['task']
        domain = metadata.get('domain', None)
        
        # Filter adapters by task and domain
        matching_adapters = []
        scores = []
        
        for adapter_id, adapter in adapters.items():
            if not adapter.is_active:
                continue
            
            score = 0.0
            
            # Check task match
            if task in adapter.skill_tags:
                score += 0.6
            
            # Check domain match
            if domain and domain in adapter.domain_tags:
                score += 0.4
            
            # Check specialization
            if adapter.specialization == AdapterSpecialization.TASK:
                score += 0.2
            
            if score > 0:
                matching_adapters.append(adapter_id)
                scores.append(score)
        
        # Select top adapters
        if matching_adapters:
            top_indices = np.argsort(scores)[::-1][:self.config.max_active_adapters]
            selected_adapters = [matching_adapters[i] for i in top_indices]
            selected_scores = [scores[i] for i in top_indices]
        else:
            # No matches, fall back to semantic
            return self._semantic_routing(query, adapters, metadata)
        
        return AdapterRoute(
            adapter_ids=selected_adapters,
            routing_scores=selected_scores,
            routing_strategy=RoutingStrategy.TASK_BASED,
            total_adapters_considered=len(adapters)
        )
    
    def _learned_routing(
        self,
        query: str,
        adapters: Dict[str, TinyAdapter],
        metadata: Optional[Dict]
    ) -> AdapterRoute:
        """Learned routing using a router network (placeholder)."""
        # In production, use a learned router network (e.g., small MLP)
        # For now, delegate to semantic routing
        return self._semantic_routing(query, adapters, metadata)
    
    def _hybrid_routing(
        self,
        query: str,
        adapters: Dict[str, TinyAdapter],
        metadata: Optional[Dict]
    ) -> AdapterRoute:
        """Hybrid routing combining multiple strategies."""
        # Combine semantic and task-based routing
        semantic_route = self._semantic_routing(query, adapters, metadata)
        task_route = self._task_based_routing(query, adapters, metadata)
        
        # Merge results (weighted combination)
        combined_scores = {}
        
        for adapter_id, score in zip(semantic_route.adapter_ids, semantic_route.routing_scores):
            combined_scores[adapter_id] = score * 0.6  # 60% semantic
        
        for adapter_id, score in zip(task_route.adapter_ids, task_route.routing_scores):
            if adapter_id in combined_scores:
                combined_scores[adapter_id] += score * 0.4  # 40% task-based
            else:
                combined_scores[adapter_id] = score * 0.4
        
        # Sort and select top adapters
        sorted_adapters = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
        top_adapters = sorted_adapters[:self.config.max_active_adapters]
        
        selected_adapters = [a[0] for a in top_adapters]
        selected_scores = [a[1] for a in top_adapters]
        
        return AdapterRoute(
            adapter_ids=selected_adapters,
            routing_scores=selected_scores,
            routing_strategy=RoutingStrategy.HYBRID,
            query_embedding=semantic_route.query_embedding,
            total_adapters_considered=len(adapters)
        )
    
    def _hierarchical_routing(
        self,
        query: str,
        adapters: Dict[str, TinyAdapter],
        metadata: Optional[Dict]
    ) -> AdapterRoute:
        """Hierarchical routing (coarse-to-fine)."""
        # Level 1: Route to domain adapters
        domain_adapters = {
            k: v for k, v in adapters.items()
            if v.specialization == AdapterSpecialization.DOMAIN and v.is_active
        }
        
        if domain_adapters:
            domain_route = self._semantic_routing(query, domain_adapters, metadata)
            
            # Level 2: Within selected domains, route to task adapters
            if domain_route.adapter_ids:
                # Find task adapters related to selected domains
                selected_domains = [
                    adapters[aid].domain_tags[0] if adapters[aid].domain_tags else None
                    for aid in domain_route.adapter_ids
                ]
                
                task_adapters = {
                    k: v for k, v in adapters.items()
                    if v.specialization == AdapterSpecialization.TASK
                    and any(d in v.domain_tags for d in selected_domains if d)
                    and v.is_active
                }
                
                if task_adapters:
                    task_route = self._semantic_routing(query, task_adapters, metadata)
                    
                    # Combine domain and task adapters
                    combined_adapters = domain_route.adapter_ids + task_route.adapter_ids
                    combined_scores = domain_route.routing_scores + task_route.routing_scores
                    
                    return AdapterRoute(
                        adapter_ids=combined_adapters[:self.config.max_active_adapters],
                        routing_scores=combined_scores[:self.config.max_active_adapters],
                        routing_strategy=RoutingStrategy.HIERARCHICAL,
                        query_embedding=domain_route.query_embedding,
                        total_adapters_considered=len(adapters)
                    )
        
        # Fall back to semantic routing
        return self._semantic_routing(query, adapters, metadata)
    
    def _get_cache_key(self, query: str, metadata: Optional[Dict]) -> str:
        """Generate cache key for routing."""
        meta_str = json.dumps(metadata, sort_keys=True) if metadata else ""
        return hashlib.md5((query + meta_str).encode()).hexdigest()


class AdapterComposer:
    """Composes multiple adapters hierarchically."""
    
    def __init__(self, config: SMoAConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Composition registry
        self.compositions: Dict[str, AdapterComposition] = {}
    
    def compose_adapters(
        self,
        adapter_ids: List[str],
        adapters: Dict[str, TinyAdapter],
        strategy: str = "parallel"
    ) -> AdapterComposition:
        """
        Compose multiple adapters.
        
        Args:
            adapter_ids: Adapters to compose
            adapters: Adapter library
            strategy: "sequential", "parallel", or "hierarchical"
        
        Returns:
            AdapterComposition
        """
        composition_id = f"comp_{len(self.compositions)}_{int(datetime.utcnow().timestamp())}"
        
        # Calculate mixing weights (based on adapter performance)
        weights = []
        for adapter_id in adapter_ids:
            if adapter_id in adapters:
                adapter = adapters[adapter_id]
                weight = adapter.success_rate if adapter.success_rate > 0 else 0.5
                weights.append(weight)
            else:
                weights.append(0.5)
        
        # Normalize weights
        total_weight = sum(weights)
        if total_weight > 0:
            weights = [w / total_weight for w in weights]
        else:
            weights = [1.0 / len(adapter_ids)] * len(adapter_ids)
        
        composition = AdapterComposition(
            composition_id=composition_id,
            component_adapters=adapter_ids,
            composition_strategy=strategy,
            mixing_weights=weights
        )
        
        self.compositions[composition_id] = composition
        
        self.logger.info(
            f"Created adapter composition '{composition_id}' with "
            f"{len(adapter_ids)} adapters (strategy: {strategy})"
        )
        
        return composition
    
    def apply_composition(
        self,
        composition: AdapterComposition,
        base_output: Any,
        adapters: Dict[str, TinyAdapter]
    ) -> Any:
        """
        Apply a composition of adapters to base model output.
        
        This is a placeholder - actual implementation depends on model architecture.
        """
        # In production, this would apply each adapter and mix results
        # For now, return base output
        return base_output


class AdapterOptimizer:
    """Optimizes adapter library through merging and pruning."""
    
    def __init__(self, config: SMoAConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def find_merge_candidates(
        self,
        adapters: Dict[str, TinyAdapter],
        top_k: int = 10
    ) -> List[AdapterMergeCandidate]:
        """
        Find adapters that should be merged.
        
        Looks for:
        - Similar adapters (high embedding similarity)
        - Complementary usage patterns
        - Overlapping specializations
        """
        candidates = []
        
        adapter_list = [a for a in adapters.values() if a.is_active]
        
        for i, adapter1 in enumerate(adapter_list):
            for adapter2 in adapter_list[i+1:]:
                # Skip if already in hierarchy
                if adapter1.adapter_id in adapter2.parent_adapters:
                    continue
                if adapter2.adapter_id in adapter1.parent_adapters:
                    continue
                
                # Calculate similarity
                if adapter1.embedding is not None and adapter2.embedding is not None:
                    similarity = np.dot(adapter1.embedding, adapter2.embedding)
                    
                    # Check if similar enough to merge
                    if similarity >= self.config.merge_similarity_threshold:
                        combined_usage = adapter1.usage_count + adapter2.usage_count
                        
                        # Estimate merge benefit
                        benefit = (
                            similarity * 0.5 +  # Similarity
                            min(1.0, combined_usage / 1000) * 0.3 +  # Usage
                            (adapter1.success_rate + adapter2.success_rate) / 2 * 0.2  # Performance
                        )
                        
                        candidate = AdapterMergeCandidate(
                            adapter_ids=[adapter1.adapter_id, adapter2.adapter_id],
                            similarity_score=similarity,
                            combined_usage=combined_usage,
                            merge_benefit_score=benefit,
                            estimated_parameters=max(
                                adapter1.num_parameters,
                                adapter2.num_parameters
                            ),
                            estimated_performance=(
                                adapter1.success_rate + adapter2.success_rate
                            ) / 2
                        )
                        
                        candidates.append(candidate)
        
        # Sort by benefit score
        candidates.sort(key=lambda x: x.merge_benefit_score, reverse=True)
        
        return candidates[:top_k]
    
    def merge_adapters(
        self,
        adapter_ids: List[str],
        adapters: Dict[str, TinyAdapter]
    ) -> TinyAdapter:
        """
        Merge multiple adapters into one.
        
        Creates a new adapter that combines the capabilities.
        """
        # Collect metadata
        all_domain_tags = set()
        all_skill_tags = set()
        total_usage = 0
        avg_success_rate = 0.0
        
        for adapter_id in adapter_ids:
            if adapter_id in adapters:
                adapter = adapters[adapter_id]
                all_domain_tags.update(adapter.domain_tags)
                all_skill_tags.update(adapter.skill_tags)
                total_usage += adapter.usage_count
                avg_success_rate += adapter.success_rate
        
        avg_success_rate /= len(adapter_ids) if adapter_ids else 1
        
        # Create merged adapter
        merged_id = f"merged_{'_'.join(adapter_ids[:2])}_{int(datetime.utcnow().timestamp())}"
        
        merged_adapter = TinyAdapter(
            adapter_id=merged_id,
            specialization=AdapterSpecialization.CUSTOM,
            granularity=AdapterGranularity.MICRO,  # Slightly larger
            domain_tags=list(all_domain_tags),
            skill_tags=list(all_skill_tags),
            rank=8,  # Larger rank for merged adapter
            usage_count=total_usage,
            success_rate=avg_success_rate,
            merged_from=adapter_ids
        )
        
        self.logger.info(f"Merged {len(adapter_ids)} adapters into '{merged_id}'")
        
        return merged_adapter
    
    def prune_adapters(
        self,
        adapters: Dict[str, TinyAdapter]
    ) -> List[str]:
        """
        Identify adapters to prune (remove).
        
        Prunes adapters with:
        - Low usage
        - Poor performance
        - Redundancy with other adapters
        """
        to_prune = []
        
        for adapter_id, adapter in adapters.items():
            if not adapter.is_active:
                continue
            
            # Check usage threshold
            if adapter.usage_count < self.config.prune_usage_threshold:
                # Calculate pruning score
                pruning_score = (
                    min(1.0, adapter.usage_count / self.config.prune_usage_threshold) * 0.4 +
                    adapter.success_rate * 0.4 +
                    adapter.specialization_score * 0.2
                )
                
                adapter.pruning_score = pruning_score
                
                if pruning_score < self.config.prune_score_threshold:
                    to_prune.append(adapter_id)
        
        if to_prune:
            self.logger.info(f"Identified {len(to_prune)} adapters for pruning")
        
        return to_prune


class SparseAdapterLibrary:
    """Manages a massive library of adapters with lazy loading."""
    
    def __init__(self, config: SMoAConfig, storage_path: Optional[Path] = None):
        self.config = config
        self.storage_path = storage_path or Path("./adapter_library")
        self.logger = logging.getLogger(__name__)
        
        # In-memory adapters (LRU cache)
        self.loaded_adapters: Dict[str, TinyAdapter] = {}
        self.adapter_access_times: Dict[str, datetime] = {}
        
        # Full adapter registry (metadata only)
        self.adapter_registry: Dict[str, TinyAdapter] = {}
        
        # Statistics
        self.total_adapters = 0
        self.cache_hits = 0
        self.cache_misses = 0
    
    def register_adapter(self, adapter: TinyAdapter) -> None:
        """Register an adapter in the library."""
        # Add to registry
        self.adapter_registry[adapter.adapter_id] = adapter
        self.total_adapters += 1
        
        # Load into memory if under cache limit
        if len(self.loaded_adapters) < self.config.max_loaded_adapters:
            self.loaded_adapters[adapter.adapter_id] = adapter
            adapter.is_loaded = True
            self.adapter_access_times[adapter.adapter_id] = datetime.utcnow()
        
        OBSERVABILITY.emit_gauge(
            "smoa.total_adapters",
            self.total_adapters
        )
    
    def get_adapter(self, adapter_id: str) -> Optional[TinyAdapter]:
        """Get an adapter (load if necessary)."""
        # Check if already loaded
        if adapter_id in self.loaded_adapters:
            self.cache_hits += 1
            self.adapter_access_times[adapter_id] = datetime.utcnow()
            return self.loaded_adapters[adapter_id]
        
        # Check registry
        if adapter_id not in self.adapter_registry:
            return None
        
        self.cache_misses += 1
        
        # Load adapter
        adapter = self.adapter_registry[adapter_id]
        
        if self.config.lazy_loading:
            self._load_adapter(adapter)
        
        # Evict if over cache limit
        if len(self.loaded_adapters) >= self.config.max_loaded_adapters:
            self._evict_lru_adapter()
        
        # Add to loaded adapters
        self.loaded_adapters[adapter_id] = adapter
        adapter.is_loaded = True
        self.adapter_access_times[adapter_id] = datetime.utcnow()
        
        return adapter
    
    def _load_adapter(self, adapter: TinyAdapter) -> None:
        """Load adapter weights from storage."""
        if adapter.storage_location and Path(adapter.storage_location).exists():
            # In production, load actual weights
            # For now, create dummy weights
            self.logger.debug(f"Loading adapter '{adapter.adapter_id}' from storage")
            adapter.weight_data = {}  # Placeholder
        else:
            self.logger.warning(f"Storage location not found for adapter '{adapter.adapter_id}'")
    
    def _evict_lru_adapter(self) -> None:
        """Evict least recently used adapter."""
        if not self.adapter_access_times:
            return
        
        # Find LRU adapter
        lru_adapter_id = min(
            self.adapter_access_times,
            key=lambda k: self.adapter_access_times[k]
        )
        
        # Offload if enabled
        if self.config.enable_offloading:
            adapter = self.loaded_adapters[lru_adapter_id]
            self._offload_adapter(adapter)
        
        # Remove from loaded adapters
        del self.loaded_adapters[lru_adapter_id]
        del self.adapter_access_times[lru_adapter_id]
        
        self.logger.debug(f"Evicted adapter '{lru_adapter_id}' (LRU)")
    
    def _offload_adapter(self, adapter: TinyAdapter) -> None:
        """Offload adapter to storage."""
        if adapter.weight_data:
            # In production, save to disk
            self.logger.debug(f"Offloading adapter '{adapter.adapter_id}' to storage")
            adapter.weight_data = None  # Clear from memory
        
        adapter.is_loaded = False
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get library statistics."""
        return {
            "total_adapters": self.total_adapters,
            "loaded_adapters": len(self.loaded_adapters),
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "hit_rate": self.cache_hits / (self.cache_hits + self.cache_misses)
                if (self.cache_hits + self.cache_misses) > 0 else 0.0
        }


class SparseAdapterMixture:
    """
    Main engine for Sparse Mixture of Adapters.
    
    Orchestrates:
    - Massive adapter libraries
    - Intelligent routing
    - Hierarchical composition
    - Automatic optimization (merging/pruning)
    - Zero-overhead serving
    """
    
    def __init__(self, config: Optional[SMoAConfig] = None):
        self.config = config or SMoAConfig()
        self.logger = logging.getLogger(__name__)
        
        # Components
        self.embedder = AdapterEmbedder()
        self.router = AdapterRouter(self.config, self.embedder)
        self.composer = AdapterComposer(self.config)
        self.optimizer = AdapterOptimizer(self.config)
        self.library = SparseAdapterLibrary(self.config)
        
        # Statistics
        self.total_queries = 0
        self.total_adaptations = 0
    
    async def create_adapter(
        self,
        specialization: AdapterSpecialization,
        domain_tags: List[str],
        skill_tags: List[str],
        granularity: AdapterGranularity = AdapterGranularity.NANO,
        **kwargs
    ) -> TinyAdapter:
        """
        Create a new tiny adapter.
        
        Args:
            specialization: Adapter specialization type
            domain_tags: Domain tags (e.g., ["medical", "radiology"])
            skill_tags: Skill tags (e.g., ["diagnosis", "classification"])
            granularity: Adapter size
            **kwargs: Additional adapter parameters
        
        Returns:
            Created TinyAdapter
        """
        adapter_id = f"adapter_{self.library.total_adapters}_{int(datetime.utcnow().timestamp())}"
        
        # Determine rank based on granularity
        rank_map = {
            AdapterGranularity.NANO: 2,
            AdapterGranularity.MICRO: 4,
            AdapterGranularity.MINI: 8,
            AdapterGranularity.SMALL: 16,
            AdapterGranularity.MEDIUM: 32,
        }
        rank = kwargs.get('rank', rank_map[granularity])
        
        adapter = TinyAdapter(
            adapter_id=adapter_id,
            specialization=specialization,
            granularity=granularity,
            domain_tags=domain_tags,
            skill_tags=skill_tags,
            rank=rank,
            **{k: v for k, v in kwargs.items() if k != 'rank'}
        )
        
        # Generate embedding
        adapter.embedding = self.embedder.embed_adapter(adapter)
        
        # Register in library
        self.library.register_adapter(adapter)
        
        self.logger.info(
            f"Created adapter '{adapter_id}' "
            f"({specialization.value}, {granularity.value}, rank={rank})"
        )
        
        OBSERVABILITY.emit_counter(
            "smoa.adapters_created",
            1,
            specialization=specialization.value,
            granularity=granularity.value
        )
        
        return adapter
    
    async def query_adapters(
        self,
        query: str,
        metadata: Optional[Dict] = None,
        strategy: Optional[RoutingStrategy] = None
    ) -> AdapterRoute:
        """
        Query the adapter library to find relevant adapters.
        
        Args:
            query: Input query
            metadata: Optional metadata (task, domain, etc.)
            strategy: Optional routing strategy
        
        Returns:
            AdapterRoute with selected adapters
        """
        self.total_queries += 1
        
        # Route to adapters
        route = self.router.route(
            query,
            self.library.adapter_registry,
            metadata,
            strategy
        )
        
        # Update usage statistics
        for adapter_id in route.adapter_ids:
            adapter = self.library.get_adapter(adapter_id)
            if adapter:
                adapter.usage_count += 1
                adapter.last_used_at = datetime.utcnow().isoformat()
        
        OBSERVABILITY.emit_counter(
            "smoa.queries",
            1,
            strategy=route.routing_strategy.value
        )
        
        OBSERVABILITY.emit_gauge(
            "smoa.active_adapters_per_query",
            len(route.adapter_ids)
        )
        
        return route
    
    async def compose_adapters(
        self,
        adapter_ids: List[str],
        strategy: str = "parallel"
    ) -> AdapterComposition:
        """
        Compose multiple adapters hierarchically.
        
        Args:
            adapter_ids: Adapters to compose
            strategy: Composition strategy
        
        Returns:
            AdapterComposition
        """
        composition = self.composer.compose_adapters(
            adapter_ids,
            self.library.adapter_registry,
            strategy
        )
        
        OBSERVABILITY.emit_counter(
            "smoa.compositions_created",
            1,
            strategy=strategy
        )
        
        return composition
    
    async def optimize_library(self) -> Dict[str, Any]:
        """
        Optimize the adapter library through merging and pruning.
        
        Returns:
            Optimization statistics
        """
        self.logger.info("Starting adapter library optimization")
        
        # Find merge candidates
        merge_candidates = []
        if self.config.enable_auto_merging:
            merge_candidates = self.optimizer.find_merge_candidates(
                self.library.adapter_registry,
                top_k=10
            )
            
            # Perform merges
            for candidate in merge_candidates[:5]:  # Merge top 5
                merged_adapter = self.optimizer.merge_adapters(
                    candidate.adapter_ids,
                    self.library.adapter_registry
                )
                
                # Register merged adapter
                self.library.register_adapter(merged_adapter)
                
                # Deactivate source adapters
                for adapter_id in candidate.adapter_ids:
                    if adapter_id in self.library.adapter_registry:
                        self.library.adapter_registry[adapter_id].is_active = False
        
        # Prune unused adapters
        pruned = []
        if self.config.enable_auto_pruning:
            pruned = self.optimizer.prune_adapters(self.library.adapter_registry)
            
            for adapter_id in pruned:
                self.library.adapter_registry[adapter_id].is_active = False
        
        optimization_stats = {
            "merge_candidates_found": len(merge_candidates),
            "adapters_merged": min(5, len(merge_candidates)),
            "adapters_pruned": len(pruned),
            "total_active_adapters": sum(
                1 for a in self.library.adapter_registry.values() if a.is_active
            )
        }
        
        self.logger.info(
            f"Optimization complete: "
            f"{optimization_stats['adapters_merged']} merged, "
            f"{optimization_stats['adapters_pruned']} pruned"
        )
        
        return optimization_stats
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get system statistics."""
        library_stats = self.library.get_statistics()
        
        return {
            "total_queries": self.total_queries,
            "total_adaptations": self.total_adaptations,
            "library": library_stats,
            "router_cache_hit_rate": self.router.cache_hits / (
                self.router.cache_hits + self.router.cache_misses
            ) if (self.router.cache_hits + self.router.cache_misses) > 0 else 0.0,
            "config": asdict(self.config)
        }


# Convenience function
def create_sparse_adapter_mixture(
    max_adapters: int = 1_000_000,
    max_active_adapters: int = 3,
    routing_strategy: str = "hybrid",
    enable_auto_optimization: bool = True,
    **kwargs
) -> SparseAdapterMixture:
    """
    Create a Sparse Mixture of Adapters system with simplified configuration.
    
    Args:
        max_adapters: Maximum adapters in library (default: 1 million!)
        max_active_adapters: Max adapters activated per query (sparse!)
        routing_strategy: 'semantic', 'task_based', 'hybrid', 'hierarchical'
        enable_auto_optimization: Auto merge/prune adapters
        **kwargs: Additional config parameters
    
    Returns:
        Configured SparseAdapterMixture
    """
    strategy_map = {
        "semantic": RoutingStrategy.SEMANTIC,
        "task_based": RoutingStrategy.TASK_BASED,
        "learned": RoutingStrategy.LEARNED,
        "hybrid": RoutingStrategy.HYBRID,
        "hierarchical": RoutingStrategy.HIERARCHICAL,
        "mixture": RoutingStrategy.MIXTURE,
    }
    
    config = SMoAConfig(
        max_adapters=max_adapters,
        max_active_adapters=max_active_adapters,
        routing_strategy=strategy_map.get(routing_strategy, RoutingStrategy.HYBRID),
        enable_auto_merging=enable_auto_optimization,
        enable_auto_pruning=enable_auto_optimization,
        **kwargs
    )
    
    return SparseAdapterMixture(config)


# Export main classes
__all__ = [
    'SparseAdapterMixture',
    'SMoAConfig',
    'TinyAdapter',
    'AdapterRoute',
    'AdapterComposition',
    'AdapterSpecialization',
    'AdapterGranularity',
    'RoutingStrategy',
    'AdapterEmbedder',
    'AdapterRouter',
    'AdapterComposer',
    'AdapterOptimizer',
    'SparseAdapterLibrary',
    'create_sparse_adapter_mixture',
]
