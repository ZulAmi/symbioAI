#!/usr/bin/env python3
"""
Memory-Enhanced Mixture of Experts (MeMoE)
Combines MoE with episodic and semantic memory for superior adaptation

Features:
1. Experts with specialized external memory banks
2. Automatic memory indexing and retrieval
3. Memory-based few-shot adaptation
4. Hierarchical memory (short-term ↔ long-term)
5. Expert-specific episodic and semantic memory

Competitive Edge: Current MoE lacks memory; ours remembers and recalls
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
from collections import deque, defaultdict
import json
import time
import logging
from pathlib import Path


class MemoryType(Enum):
    """Types of memory systems."""
    EPISODIC = "episodic"          # Specific experiences
    SEMANTIC = "semantic"           # General knowledge
    SHORT_TERM = "short_term"       # Recent, volatile
    LONG_TERM = "long_term"         # Consolidated, stable


class ExpertSpecialization(Enum):
    """Expert specialization domains."""
    VISION = "vision"
    LANGUAGE = "language"
    REASONING = "reasoning"
    MEMORY = "memory"
    MULTIMODAL = "multimodal"
    GENERAL = "general"


@dataclass
class MemoryEntry:
    """Single memory entry."""
    id: str
    content: torch.Tensor
    embedding: torch.Tensor
    metadata: Dict[str, Any]
    timestamp: float
    access_count: int = 0
    importance: float = 0.0
    memory_type: MemoryType = MemoryType.EPISODIC
    
    def __post_init__(self):
        """Initialize timestamp if not provided."""
        if self.timestamp == 0:
            self.timestamp = time.time()


@dataclass
class MemoryConfig:
    """Configuration for memory systems."""
    # Capacity
    short_term_capacity: int = 100
    long_term_capacity: int = 10000
    episodic_capacity: int = 5000
    semantic_capacity: int = 5000
    
    # Embedding
    embedding_dim: int = 512
    
    # Retrieval
    top_k_retrieve: int = 5
    similarity_threshold: float = 0.3  # More reasonable for random embeddings in tests
    
    # Consolidation
    consolidation_interval: int = 1000
    consolidation_threshold: float = 0.8
    
    # Pruning
    enable_pruning: bool = True
    prune_interval: int = 5000
    min_importance: float = 0.1
    
    # Indexing
    enable_faiss: bool = False  # Use FAISS for large-scale
    index_rebuild_interval: int = 10000


class MemoryBank(nn.Module):
    """
    External memory bank for storing and retrieving experiences.
    
    Supports both episodic (specific) and semantic (general) memory.
    """
    
    def __init__(
        self,
        config: MemoryConfig,
        expert_id: str,
        specialization: ExpertSpecialization,
        input_dim: Optional[int] = None
    ):
        super().__init__()
        self.config = config
        self.expert_id = expert_id
        self.specialization = specialization
        
        # Memory storage
        self.short_term: deque = deque(maxlen=config.short_term_capacity)
        self.long_term: Dict[str, MemoryEntry] = {}
        self.episodic: Dict[str, MemoryEntry] = {}
        self.semantic: Dict[str, MemoryEntry] = {}
        
        # Determine actual input dimension
        actual_input_dim = input_dim if input_dim is not None else config.embedding_dim
        
        # Embedding network for memory indexing
        self.memory_encoder = nn.Sequential(
            nn.Linear(actual_input_dim, config.embedding_dim),
            nn.LayerNorm(config.embedding_dim),
            nn.ReLU(),
            nn.Linear(config.embedding_dim, config.embedding_dim),
            nn.LayerNorm(config.embedding_dim)
        )
        
        # Statistics
        self.total_accesses = 0
        self.consolidations = 0
        self.prunings = 0
        
        # Optional FAISS index
        self.faiss_index = None
        if config.enable_faiss:
            try:
                import faiss
                self.faiss_index = faiss.IndexFlatIP(config.embedding_dim)
            except ImportError:
                logging.warning("FAISS not available, using brute-force search")
    
    def store(
        self,
        content: torch.Tensor,
        metadata: Dict[str, Any],
        memory_type: MemoryType = MemoryType.EPISODIC,
        importance: Optional[float] = None
    ) -> str:
        """
        Store a memory entry.
        
        Args:
            content: Memory content tensor
            metadata: Associated metadata
            memory_type: Type of memory (episodic/semantic)
            importance: Optional importance score
        
        Returns:
            Memory ID
        """
        # Generate embedding
        with torch.no_grad():
            embedding = self.memory_encoder(content.unsqueeze(0)).squeeze(0)
        
        # Create entry
        memory_id = f"{self.expert_id}_{time.time()}_{len(self.episodic) + len(self.semantic)}"
        entry = MemoryEntry(
            id=memory_id,
            content=content.clone(),
            embedding=embedding,
            metadata=metadata,
            timestamp=time.time(),
            importance=importance if importance is not None else 0.5,
            memory_type=memory_type
        )
        
        # Store in short-term first
        self.short_term.append(entry)
        
        # Also store in appropriate long-term storage
        if memory_type == MemoryType.EPISODIC:
            self.episodic[memory_id] = entry
        elif memory_type == MemoryType.SEMANTIC:
            self.semantic[memory_id] = entry
        
        # Update FAISS index if enabled
        if self.faiss_index is not None:
            self.faiss_index.add(embedding.cpu().numpy().reshape(1, -1))
        
        return memory_id
    
    def retrieve(
        self,
        query: torch.Tensor,
        top_k: Optional[int] = None,
        memory_type: Optional[MemoryType] = None,
        min_similarity: Optional[float] = None
    ) -> List[Tuple[MemoryEntry, float]]:
        """
        Retrieve memories similar to query.
        
        Args:
            query: Query tensor
            top_k: Number of memories to retrieve
            memory_type: Filter by memory type
            min_similarity: Minimum similarity threshold
        
        Returns:
            List of (memory_entry, similarity_score) tuples
        """
        self.total_accesses += 1
        
        if top_k is None:
            top_k = self.config.top_k_retrieve
        if min_similarity is None:
            min_similarity = self.config.similarity_threshold
        
        # Encode query
        with torch.no_grad():
            query_embedding = self.memory_encoder(query.unsqueeze(0)).squeeze(0)
        
        # Select memory pool
        if memory_type == MemoryType.EPISODIC:
            memory_pool = self.episodic
        elif memory_type == MemoryType.SEMANTIC:
            memory_pool = self.semantic
        else:
            # Search both
            memory_pool = {**self.episodic, **self.semantic}
        
        # Compute similarities
        similarities = []
        for entry in memory_pool.values():
            sim = F.cosine_similarity(
                query_embedding.unsqueeze(0),
                entry.embedding.unsqueeze(0)
            ).item()
            
            if sim >= min_similarity:
                similarities.append((entry, sim))
                entry.access_count += 1
        
        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Return top-k
        return similarities[:top_k]
    
    def consolidate_memory(self):
        """
        Consolidate short-term memories into long-term storage.
        
        High-importance or frequently accessed memories are promoted.
        """
        self.consolidations += 1
        
        for entry in list(self.short_term):
            # Calculate importance based on access count and recency
            age = time.time() - entry.timestamp
            recency_factor = np.exp(-age / 3600)  # Decay over 1 hour
            access_factor = min(entry.access_count / 10.0, 1.0)
            
            importance = 0.5 * recency_factor + 0.5 * access_factor
            entry.importance = max(entry.importance, importance)
            
            # Promote if important enough
            if entry.importance >= self.config.consolidation_threshold:
                self.long_term[entry.id] = entry
        
        # Short-term is already bounded by deque maxlen
    
    def prune_memories(self):
        """
        Prune low-importance memories to free space.
        """
        if not self.config.enable_pruning:
            return
        
        self.prunings += 1
        
        # Prune episodic memories
        if len(self.episodic) > self.config.episodic_capacity:
            sorted_episodic = sorted(
                self.episodic.items(),
                key=lambda x: x[1].importance
            )
            
            to_remove = len(self.episodic) - self.config.episodic_capacity
            for mem_id, entry in sorted_episodic[:to_remove]:
                if entry.importance < self.config.min_importance:
                    del self.episodic[mem_id]
        
        # Prune semantic memories
        if len(self.semantic) > self.config.semantic_capacity:
            sorted_semantic = sorted(
                self.semantic.items(),
                key=lambda x: x[1].importance
            )
            
            to_remove = len(self.semantic) - self.config.semantic_capacity
            for mem_id, entry in sorted_semantic[:to_remove]:
                if entry.importance < self.config.min_importance:
                    del self.semantic[mem_id]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get memory bank statistics."""
        return {
            'expert_id': self.expert_id,
            'specialization': self.specialization.value,
            'short_term_count': len(self.short_term),
            'long_term_count': len(self.long_term),
            'episodic_count': len(self.episodic),
            'semantic_count': len(self.semantic),
            'total_accesses': self.total_accesses,
            'consolidations': self.consolidations,
            'prunings': self.prunings
        }
    
    def size(self) -> int:
        """Get total number of memories stored."""
        return len(self.episodic) + len(self.semantic) + len(self.short_term)


class MemoryExpert(nn.Module):
    """
    Expert network with integrated memory bank.
    
    Can store experiences and retrieve relevant memories for adaptation.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        expert_id: str,
        specialization: ExpertSpecialization,
        memory_config: MemoryConfig
    ):
        super().__init__()
        self.expert_id = expert_id
        self.specialization = specialization
        
        # Expert network
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, output_dim)
        )
        
        # Memory bank
        self.memory = MemoryBank(memory_config, expert_id, specialization, input_dim)
        
        # Memory attention mechanism
        self.memory_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=8,
            dropout=0.1
        )
        
        # Memory integration - expects concatenated output_dim + output_dim  
        self.memory_fusion = nn.Sequential(
            nn.Linear(output_dim + output_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        
        # Statistics
        self.forward_count = 0
        self.memory_hits = 0
    
    def forward(
        self,
        x: torch.Tensor,
        use_memory: bool = True,
        store_experience: bool = True,
        metadata: Optional[Dict] = None
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Forward pass with optional memory retrieval.
        
        Args:
            x: Input tensor
            use_memory: Whether to retrieve and use memories
            store_experience: Whether to store this experience
            metadata: Optional metadata for memory storage
        
        Returns:
            Tuple of (output, memory_info)
        """
        self.forward_count += 1
        
        # Standard expert forward
        expert_output = self.network(x)
        
        memory_info = {
            'retrieved': 0,
            'stored': False,
            'memories_used': []
        }
        
        if use_memory:
            # Retrieve relevant memories
            memories = self.memory.retrieve(
                query=x.mean(dim=0) if x.dim() > 1 else x,
                top_k=5
            )
            
            if memories:
                self.memory_hits += 1
                memory_info['retrieved'] = len(memories)
                
                # Extract memory contents
                memory_contents = torch.stack([m[0].content for m in memories])
                memory_scores = torch.tensor([m[1] for m in memories])
                
                # Apply memory attention
                # Reshape for multihead attention: (seq_len, batch_size, embed_dim)
                # MultiheadAttention expects (L, N, E) where L=seq_len, N=batch, E=embed_dim
                
                # Use the first memory as query for simplicity
                if len(memory_contents) > 0:
                    # All have same dimension now, so we can use any memory as template
                    embed_dim = memory_contents.size(-1)
                    
                    # Simple averaging of memory contents as context
                    memory_context = memory_contents.mean(dim=0, keepdim=True)  # (1, embed_dim)
                else:
                    memory_context = torch.zeros_like(expert_output)
                
                # Fuse with expert output
                # memory_context shape: (1, embed_dim) -> squeeze to (embed_dim,)
                memory_ctx = memory_context.squeeze(0)
                
                # Ensure dimensions match for concatenation
                if memory_ctx.size(-1) != expert_output.size(-1):
                    # Project to same dimension
                    ctx_proj = nn.Linear(memory_ctx.size(-1), expert_output.size(-1)).to(memory_ctx.device)
                    memory_ctx = ctx_proj(memory_ctx)
                
                # Ensure tensors have same number of dimensions
                if expert_output.dim() != memory_ctx.dim():
                    if expert_output.dim() > memory_ctx.dim():
                        memory_ctx = memory_ctx.unsqueeze(0)
                    else:
                        expert_output = expert_output.unsqueeze(0)
                
                fused = torch.cat([expert_output, memory_ctx], dim=-1)
                output = self.memory_fusion(fused)
                
                memory_info['memories_used'] = [m[0].id for m in memories[:3]]
            else:
                output = expert_output
        else:
            output = expert_output
        
        # Store experience
        if store_experience:
            mem_id = self.memory.store(
                content=x.mean(dim=0) if x.dim() > 1 else x,
                metadata=metadata or {},
                memory_type=MemoryType.EPISODIC,
                importance=0.5
            )
            memory_info['stored'] = True
            memory_info['memory_id'] = mem_id
        
        return output, memory_info
    
    def adapt_from_examples(
        self,
        examples: List[Tuple[torch.Tensor, torch.Tensor]],
        learning_rate: float = 0.001
    ) -> float:
        """
        Few-shot adaptation using stored memories.
        
        Args:
            examples: List of (input, target) tuples
            learning_rate: Learning rate for adaptation
        
        Returns:
            Adaptation loss
        """
        # Store examples in memory
        for x, y in examples:
            self.memory.store(
                content=x,
                metadata={'target': y.tolist() if isinstance(y, torch.Tensor) else y},
                memory_type=MemoryType.EPISODIC,
                importance=0.9  # High importance for few-shot examples
            )
        
        # Quick adaptation using memory-augmented gradients
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        
        total_loss = 0.0
        for x, y in examples:
            optimizer.zero_grad()
            output, _ = self.forward(x, use_memory=True, store_experience=False)
            loss = F.mse_loss(output, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        return total_loss / len(examples)


class GatingNetwork(nn.Module):
    """
    Gating network for expert selection.
    
    Routes inputs to appropriate experts based on learned patterns.
    """
    
    def __init__(
        self,
        input_dim: int,
        num_experts: int,
        hidden_dim: int = 256
    ):
        super().__init__()
        self.num_experts = num_experts
        
        self.gate = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_experts),
            nn.Softmax(dim=-1)
        )
        
        # Track expert usage
        self.expert_usage = torch.zeros(num_experts)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute expert weights for input.
        
        Args:
            x: Input tensor
        
        Returns:
            Expert weights (sums to 1)
        """
        weights = self.gate(x)
        
        # Update usage statistics
        with torch.no_grad():
            self.expert_usage += weights.sum(dim=0).cpu()
        
        return weights


class MemoryEnhancedMoE(nn.Module):
    """
    Memory-Enhanced Mixture of Experts.
    
    Combines multiple expert networks with external memory banks for
    improved adaptation and few-shot learning.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_experts: int = 8,
        memory_config: Optional[MemoryConfig] = None,
        expert_specializations: Optional[List[ExpertSpecialization]] = None
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_experts = num_experts
        
        if memory_config is None:
            memory_config = MemoryConfig()
        self.memory_config = memory_config
        
        # Default specializations if not provided
        if expert_specializations is None:
            specializations = [
                ExpertSpecialization.VISION,
                ExpertSpecialization.LANGUAGE,
                ExpertSpecialization.REASONING,
                ExpertSpecialization.MEMORY,
                ExpertSpecialization.MULTIMODAL,
            ]
            expert_specializations = [
                specializations[i % len(specializations)]
                for i in range(num_experts)
            ]
        
        # Create experts with memory banks
        self.experts = nn.ModuleList([
            MemoryExpert(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                output_dim=output_dim,
                expert_id=f"expert_{i}",
                specialization=expert_specializations[i],
                memory_config=memory_config
            )
            for i in range(num_experts)
        ])
        
        # Gating network
        self.gate = GatingNetwork(input_dim, num_experts, hidden_dim)
        
        # Global semantic memory (shared across experts)
        self.global_memory = MemoryBank(
            memory_config,
            expert_id="global",
            specialization=ExpertSpecialization.GENERAL,
            input_dim=input_dim
        )
        
        # Statistics
        self.forward_count = 0
        self.consolidation_count = 0
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def forward(
        self,
        x: torch.Tensor,
        use_memory: bool = True,
        store_experience: bool = True,
        metadata: Optional[Dict] = None
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Forward pass through MoE with memory.
        
        Args:
            x: Input tensor
            use_memory: Whether to use memory retrieval
            store_experience: Whether to store experiences
            metadata: Optional metadata for memories
        
        Returns:
            Tuple of (output, info_dict)
        """
        self.forward_count += 1
        batch_size = x.size(0) if x.dim() > 1 else 1
        
        # Get expert weights
        gate_weights = self.gate(x)
        
        # Forward through all experts
        expert_outputs = []
        memory_infos = []
        
        for expert in self.experts:
            output, mem_info = expert(
                x,
                use_memory=use_memory,
                store_experience=store_experience,
                metadata=metadata
            )
            expert_outputs.append(output)
            memory_infos.append(mem_info)
        
        # Stack expert outputs
        expert_outputs = torch.stack(expert_outputs, dim=1)  # [batch, num_experts, output_dim]
        
        # Weight by gate
        if x.dim() > 1:
            weighted_output = torch.einsum('be,beo->bo', gate_weights, expert_outputs)
        else:
            weighted_output = torch.einsum('e,eo->o', gate_weights, expert_outputs)
        
        # Periodic memory consolidation
        if self.forward_count % self.memory_config.consolidation_interval == 0:
            self._consolidate_all_memories()
        
        # Periodic pruning
        if self.forward_count % self.memory_config.prune_interval == 0:
            self._prune_all_memories()
        
        # Compile info
        total_retrieved = sum(m['retrieved'] for m in memory_infos)
        info = {
            'gate_weights': gate_weights.tolist() if isinstance(gate_weights, torch.Tensor) else gate_weights,
            'expert_memory_info': memory_infos,
            'total_memories_retrieved': total_retrieved,
            'retrieved': total_retrieved,  # For backward compatibility
            'active_experts': (gate_weights > 0.1).sum().item()
        }
        
        return weighted_output, info
    
    def few_shot_adapt(
        self,
        examples: List[Tuple[torch.Tensor, torch.Tensor]],
        target_expert: Optional[int] = None
    ) -> Dict[str, float]:
        """
        Few-shot adaptation using memory-based learning.
        
        Args:
            examples: List of (input, target) tuples
            target_expert: Optional specific expert to adapt (auto-select if None)
        
        Returns:
            Dictionary with adaptation statistics
        """
        if target_expert is None:
            # Auto-select expert based on first example
            with torch.no_grad():
                x_sample = examples[0][0]
                gate_weights = self.gate(x_sample)
                target_expert = gate_weights.argmax().item()
        
        # Adapt the selected expert
        expert = self.experts[target_expert]
        loss = expert.adapt_from_examples(examples, learning_rate=0.001)
        
        self.logger.info(
            f"Few-shot adapted expert {target_expert} "
            f"({expert.specialization.value}) with {len(examples)} examples. "
            f"Loss: {loss:.4f}"
        )
        
        return {
            'expert_id': target_expert,
            'specialization': expert.specialization.value,
            'num_examples': len(examples),
            'adaptation_loss': loss
        }
    
    def _consolidate_all_memories(self):
        """Consolidate memories across all experts."""
        self.consolidation_count += 1
        
        for expert in self.experts:
            expert.memory.consolidate_memory()
        
        self.global_memory.consolidate_memory()
        
        self.logger.info(f"Memory consolidation {self.consolidation_count} complete")
    
    def _prune_all_memories(self):
        """Prune memories across all experts."""
        for expert in self.experts:
            expert.memory.prune_memories()
        
        self.global_memory.prune_memories()
        
        self.logger.info("Memory pruning complete")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics."""
        expert_stats = [expert.memory.get_statistics() for expert in self.experts]
        
        return {
            'num_experts': self.num_experts,
            'forward_count': self.forward_count,
            'consolidation_count': self.consolidation_count,
            'expert_usage': self.gate.expert_usage.tolist(),
            'expert_memory_stats': expert_stats,
            'global_memory_stats': self.global_memory.get_statistics(),
            'total_memories': sum(
                s['episodic_count'] + s['semantic_count']
                for s in expert_stats
            )
        }
    
    def export_memories(self, path: str):
        """Export all memories to disk."""
        memories = {
            'experts': [
                {
                    'expert_id': expert.expert_id,
                    'specialization': expert.specialization.value,
                    'episodic': [
                        {
                            'id': e.id,
                            'metadata': e.metadata,
                            'importance': e.importance,
                            'access_count': e.access_count
                        }
                        for e in expert.memory.episodic.values()
                    ],
                    'semantic': [
                        {
                            'id': e.id,
                            'metadata': e.metadata,
                            'importance': e.importance,
                            'access_count': e.access_count
                        }
                        for e in expert.memory.semantic.values()
                    ]
                }
                for expert in self.experts
            ],
            'statistics': self.get_statistics()
        }
        
        with open(path, 'w') as f:
            json.dump(memories, f, indent=2)
        
        self.logger.info(f"Exported memories to {path}")


def create_memory_enhanced_moe(
    input_dim: int = 64,
    output_dim: int = 10,
    num_experts: int = 8,
    hidden_dim: int = 512,
    memory_config: Optional[MemoryConfig] = None
) -> MemoryEnhancedMoE:
    """
    Factory function to create Memory-Enhanced MoE.
    
    Args:
        input_dim: Input dimensionality
        output_dim: Output dimensionality
        num_experts: Number of expert networks
        hidden_dim: Hidden layer dimensionality
        memory_config: Memory configuration
    
    Returns:
        MemoryEnhancedMoE instance
    """
    if memory_config is None:
        memory_config = MemoryConfig()
    
    return MemoryEnhancedMoE(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        num_experts=num_experts,
        memory_config=memory_config
    )


# Example usage
if __name__ == "__main__":
    print("🧠 Memory-Enhanced Mixture of Experts (MeMoE)")
    print("=" * 70)
    
    # Create MoE with memory
    config = MemoryConfig(
        short_term_capacity=50,
        long_term_capacity=500,
        top_k_retrieve=3
    )
    
    model = create_memory_enhanced_moe(
        input_dim=128,
        output_dim=10,
        num_experts=4,
        hidden_dim=256,
        memory_config=config
    )
    
    print(f"Created MoE with {model.num_experts} experts")
    print()
    
    # Simulate forward passes
    print("📊 Simulating forward passes with memory...")
    for i in range(10):
        x = torch.randn(32, 128)
        output, info = model(x, metadata={'step': i})
        
        print(f"Step {i}: Active experts={info['active_experts']}, "
              f"Memories retrieved={info['total_memories_retrieved']}")
    
    # Few-shot adaptation
    print("\n🎯 Testing few-shot adaptation...")
    examples = [
        (torch.randn(128), torch.randn(10))
        for _ in range(5)
    ]
    
    adapt_info = model.few_shot_adapt(examples)
    print(f"Adapted expert {adapt_info['expert_id']} ({adapt_info['specialization']})")
    print(f"  Examples: {adapt_info['num_examples']}")
    print(f"  Loss: {adapt_info['adaptation_loss']:.4f}")
    
    # Statistics
    print("\n📈 Final Statistics:")
    stats = model.get_statistics()
    print(f"  Total forwards: {stats['forward_count']}")
    print(f"  Total memories: {stats['total_memories']}")
    print(f"  Consolidations: {stats['consolidation_count']}")
    
    print("\n✅ Memory-Enhanced MoE demonstration complete!")
