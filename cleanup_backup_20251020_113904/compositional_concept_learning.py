"""
Compositional Concept Learning System - Symbio AI

Learns reusable symbolic concepts that compose, enabling human-interpretable
concept hierarchies and compositional generalization.

Key Features:
1. Object-centric representations with binding
2. Relation networks for compositional generalization
3. Abstract reasoning over learned symbolic structures
4. Disentangled representations for concept manipulation
5. Human-interpretable concept hierarchies

This system enables AI to learn and compose concepts the way humans do,
building complex understanding from simple, reusable building blocks.
"""

import asyncio
import logging
import random
import json
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Set, Callable, Union
from dataclasses import dataclass, field, asdict
from datetime import datetime
from collections import defaultdict, deque
from enum import Enum
from abc import ABC, abstractmethod

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    # Mock classes for development
    class nn:
        class Module:
            def __init__(self): pass
            def forward(self, x): return x
        class Linear:
            def __init__(self, *args, **kwargs): pass
        class ModuleList:
            def __init__(self, modules): self.modules = modules
            def __iter__(self): return iter(self.modules)
        class Parameter:
            def __init__(self, tensor): self.data = tensor
        class Embedding:
            def __init__(self, *args, **kwargs): pass
        class LayerNorm:
            def __init__(self, *args, **kwargs): pass
        class Dropout:
            def __init__(self, *args, **kwargs): pass
    
    class torch:
        @staticmethod
        def tensor(x): return x
        @staticmethod
        def zeros(*args, **kwargs): return [0.0]
        @staticmethod
        def ones(*args, **kwargs): return [1.0]
        @staticmethod
        def randn(*args, **kwargs): return [random.random()]
        @staticmethod
        def cat(tensors, dim=0): return tensors[0] if tensors else []
        @staticmethod
        def stack(tensors, dim=0): return tensors
        @staticmethod
        def sigmoid(x): return 1 / (1 + np.exp(-x)) if isinstance(x, (int, float)) else x
        @staticmethod
        def tanh(x): return np.tanh(x) if isinstance(x, (int, float)) else x
    
    class F:
        @staticmethod
        def relu(x): return max(0, x) if isinstance(x, (int, float)) else x
        @staticmethod
        def softmax(x, dim=-1): return x
        @staticmethod
        def cosine_similarity(x, y, dim=0): return random.random()
        @staticmethod
        def normalize(x, dim=-1): return x


# ============================================================================
# CORE CONCEPTS AND STRUCTURES
# ============================================================================

class ConceptType(Enum):
    """Types of learnable concepts."""
    OBJECT = "object"  # Physical or abstract objects
    ATTRIBUTE = "attribute"  # Properties (color, size, shape)
    RELATION = "relation"  # Relationships between objects
    ACTION = "action"  # Operations or transformations
    ABSTRACT = "abstract"  # Abstract concepts (justice, beauty)
    COMPOSITE = "composite"  # Compositions of other concepts


class BindingType(Enum):
    """Types of variable bindings."""
    HARD_BINDING = "hard"  # Deterministic binding
    SOFT_BINDING = "soft"  # Probabilistic binding
    ATTENTION_BINDING = "attention"  # Attention-based binding
    SLOT_BINDING = "slot"  # Slot-based binding


@dataclass
class Concept:
    """Represents a learned symbolic concept."""
    concept_id: str
    concept_type: ConceptType
    name: str
    
    # Representation
    embedding: List[float] = field(default_factory=list)
    symbolic_definition: Optional[str] = None
    
    # Compositionality
    is_primitive: bool = True  # Cannot be decomposed further
    composed_from: List[str] = field(default_factory=list)  # Component concept IDs
    composition_operation: Optional[str] = None  # How components combine
    
    # Attributes
    attributes: Dict[str, Any] = field(default_factory=dict)
    relations: List['ConceptRelation'] = field(default_factory=list)
    
    # Learning metadata
    examples_seen: int = 0
    confidence: float = 0.0
    abstraction_level: int = 0  # 0=concrete, higher=more abstract
    
    # Interpretability
    human_description: str = ""
    visual_representation: Optional[str] = None
    
    # Usage statistics
    usage_count: int = 0
    successful_compositions: int = 0
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())


@dataclass
class ConceptRelation:
    """Represents a relation between concepts."""
    relation_id: str
    relation_type: str  # "is-a", "has-a", "part-of", "similar-to", etc.
    source_concept: str  # Source concept ID
    target_concept: str  # Target concept ID
    
    strength: float = 1.0  # Strength of relation (0-1)
    is_symmetric: bool = False
    is_transitive: bool = False
    
    learned_from: List[str] = field(default_factory=list)  # Example IDs
    confidence: float = 0.0


@dataclass
class ObjectRepresentation:
    """Object-centric representation with slots."""
    object_id: str
    concept_id: str  # Which concept this object instantiates
    
    # Slot-based representation
    slots: Dict[str, 'Slot'] = field(default_factory=dict)
    
    # Spatial/temporal information
    position: Optional[Tuple[float, float, float]] = None
    timestamp: Optional[float] = None
    
    # Binding to perception
    bound_to: Optional[str] = None  # ID of perceptual input
    binding_strength: float = 0.0


@dataclass
class Slot:
    """Slot in object-centric representation."""
    slot_id: str
    slot_type: str  # "attribute", "relation", "component"
    
    value: Any = None
    binding_type: BindingType = BindingType.SOFT_BINDING
    confidence: float = 0.0
    
    # For compositional slots
    bound_concept: Optional[str] = None


@dataclass
class ConceptHierarchy:
    """Hierarchical organization of concepts."""
    hierarchy_id: str
    root_concept: str
    
    # Tree structure
    parent_map: Dict[str, str] = field(default_factory=dict)  # child -> parent
    children_map: Dict[str, List[str]] = field(default_factory=dict)  # parent -> children
    
    # Abstraction levels
    level_map: Dict[str, int] = field(default_factory=dict)  # concept -> level
    
    # Metadata
    num_concepts: int = 0
    max_depth: int = 0


# ============================================================================
# OBJECT-CENTRIC REPRESENTATION LEARNING
# ============================================================================

class SlotAttentionModule(nn.Module if TORCH_AVAILABLE else object):
    """
    Slot attention mechanism for object-centric representation learning.
    Based on "Object-Centric Learning with Slot Attention" (Locatello et al., 2020)
    """
    
    def __init__(
        self,
        num_slots: int = 7,
        slot_dim: int = 64,
        input_dim: int = 64,
        num_iterations: int = 3
    ):
        if TORCH_AVAILABLE:
            super().__init__()
        
        self.num_slots = num_slots
        self.slot_dim = slot_dim
        self.num_iterations = num_iterations
        
        if TORCH_AVAILABLE:
            # Slot initialization
            self.slots_mu = nn.Parameter(torch.randn(1, 1, slot_dim))
            self.slots_log_sigma = nn.Parameter(torch.randn(1, 1, slot_dim))
            
            # Attention
            self.to_q = nn.Linear(slot_dim, slot_dim, bias=False)
            self.to_k = nn.Linear(input_dim, slot_dim, bias=False)
            self.to_v = nn.Linear(input_dim, slot_dim, bias=False)
            
            # GRU for slot updates
            self.gru = nn.GRUCell(slot_dim, slot_dim)
            
            # Layer norm
            self.norm_slots = nn.LayerNorm(slot_dim)
            self.norm_inputs = nn.LayerNorm(input_dim)
    
    def forward(self, inputs):
        """
        Args:
            inputs: [batch_size, num_inputs, input_dim]
        
        Returns:
            slots: [batch_size, num_slots, slot_dim]
        """
        if not TORCH_AVAILABLE:
            return inputs
        
        batch_size, num_inputs, input_dim = inputs.shape
        
        # Initialize slots
        mu = self.slots_mu.expand(batch_size, self.num_slots, -1)
        sigma = self.slots_log_sigma.exp().expand(batch_size, self.num_slots, -1)
        slots = mu + sigma * torch.randn_like(mu)
        
        inputs = self.norm_inputs(inputs)
        k = self.to_k(inputs)  # [batch, num_inputs, slot_dim]
        v = self.to_v(inputs)  # [batch, num_inputs, slot_dim]
        
        # Iterative attention
        for _ in range(self.num_iterations):
            slots_prev = slots
            slots = self.norm_slots(slots)
            
            q = self.to_q(slots)  # [batch, num_slots, slot_dim]
            
            # Attention scores
            attn_logits = torch.bmm(k, q.transpose(1, 2))  # [batch, num_inputs, num_slots]
            attn = F.softmax(attn_logits, dim=-1)
            
            # Weighted average
            attn_normalized = attn / (attn.sum(dim=1, keepdim=True) + 1e-8)
            updates = torch.bmm(attn_normalized.transpose(1, 2), v)  # [batch, num_slots, slot_dim]
            
            # GRU update
            slots = self.gru(
                updates.reshape(-1, self.slot_dim),
                slots_prev.reshape(-1, self.slot_dim)
            )
            slots = slots.reshape(batch_size, self.num_slots, self.slot_dim)
        
        return slots


class ObjectEncoder(nn.Module if TORCH_AVAILABLE else object):
    """
    Encodes perceptual input into object-centric representations.
    """
    
    def __init__(
        self,
        input_dim: int = 128,
        num_slots: int = 7,
        slot_dim: int = 64,
        hidden_dim: int = 128
    ):
        if TORCH_AVAILABLE:
            super().__init__()
        
        self.input_dim = input_dim
        self.num_slots = num_slots
        self.slot_dim = slot_dim
        
        if TORCH_AVAILABLE:
            # Perception encoder
            self.perception_encoder = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, slot_dim)
            )
            
            # Slot attention
            self.slot_attention = SlotAttentionModule(
                num_slots=num_slots,
                slot_dim=slot_dim,
                input_dim=slot_dim
            )
            
            # Object decoder (for reconstruction)
            self.object_decoder = nn.Sequential(
                nn.Linear(slot_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, input_dim)
            )
    
    def forward(self, x):
        """
        Args:
            x: Perceptual input [batch_size, input_dim] or [batch_size, num_features, input_dim]
        
        Returns:
            slots: Object-centric representations [batch_size, num_slots, slot_dim]
        """
        if not TORCH_AVAILABLE:
            return x
        
        # Encode perception
        if len(x.shape) == 2:
            x = x.unsqueeze(1)  # Add feature dimension
        
        encoded = self.perception_encoder(x)
        
        # Extract object slots
        slots = self.slot_attention(encoded)
        
        return slots
    
    def decode_objects(self, slots):
        """Reconstruct input from object slots."""
        if not TORCH_AVAILABLE:
            return slots
        
        # Decode each slot
        reconstructions = []
        for i in range(slots.size(1)):
            slot = slots[:, i, :]
            recon = self.object_decoder(slot)
            reconstructions.append(recon)
        
        return torch.stack(reconstructions, dim=1)


# ============================================================================
# RELATION NETWORKS
# ============================================================================

class RelationNetwork(nn.Module if TORCH_AVAILABLE else object):
    """
    Relation network for compositional generalization.
    Models relations between object pairs.
    """
    
    def __init__(
        self,
        object_dim: int = 64,
        relation_dim: int = 128,
        output_dim: int = 64,
        num_layers: int = 3
    ):
        if TORCH_AVAILABLE:
            super().__init__()
        
        self.object_dim = object_dim
        self.relation_dim = relation_dim
        
        if TORCH_AVAILABLE:
            # Pairwise relation encoder
            self.relation_encoder = nn.Sequential(
                nn.Linear(object_dim * 2, relation_dim),
                nn.ReLU(),
                nn.Linear(relation_dim, relation_dim),
                nn.ReLU()
            )
            
            # Relation aggregator
            self.relation_aggregator = nn.Sequential(
                nn.Linear(relation_dim, relation_dim),
                nn.ReLU(),
                nn.Linear(relation_dim, output_dim)
            )
            
            # Self-attention over relations
            self.relation_attention = nn.MultiheadAttention(
                embed_dim=relation_dim,
                num_heads=4,
                batch_first=True
            ) if hasattr(nn, 'MultiheadAttention') else None
    
    def forward(self, objects):
        """
        Args:
            objects: [batch_size, num_objects, object_dim]
        
        Returns:
            relation_output: [batch_size, output_dim]
        """
        if not TORCH_AVAILABLE:
            return objects
        
        batch_size, num_objects, _ = objects.shape
        
        # Compute all pairwise relations
        relations = []
        for i in range(num_objects):
            for j in range(num_objects):
                if i != j:
                    obj_i = objects[:, i, :]
                    obj_j = objects[:, j, :]
                    pair = torch.cat([obj_i, obj_j], dim=-1)
                    relation = self.relation_encoder(pair)
                    relations.append(relation)
        
        if not relations:
            return torch.zeros(batch_size, self.relation_dim)
        
        relations = torch.stack(relations, dim=1)  # [batch, num_pairs, relation_dim]
        
        # Apply attention if available
        if self.relation_attention is not None:
            relations, _ = self.relation_attention(relations, relations, relations)
        
        # Aggregate relations
        aggregated = relations.mean(dim=1)  # [batch, relation_dim]
        output = self.relation_aggregator(aggregated)
        
        return output


class CompositionFunction(nn.Module if TORCH_AVAILABLE else object):
    """
    Neural module for composing concepts.
    Learns how to combine primitive concepts into complex ones.
    """
    
    def __init__(
        self,
        concept_dim: int = 64,
        composition_dim: int = 128,
        num_composition_ops: int = 5
    ):
        if TORCH_AVAILABLE:
            super().__init__()
        
        self.concept_dim = concept_dim
        self.num_ops = num_composition_ops
        
        if TORCH_AVAILABLE:
            # Composition operators
            self.composition_ops = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(concept_dim * 2, composition_dim),
                    nn.ReLU(),
                    nn.Linear(composition_dim, concept_dim)
                )
                for _ in range(num_composition_ops)
            ])
            
            # Operation selector
            self.op_selector = nn.Sequential(
                nn.Linear(concept_dim * 2, composition_dim),
                nn.ReLU(),
                nn.Linear(composition_dim, num_composition_ops)
            )
    
    def forward(self, concept1, concept2):
        """
        Compose two concepts into a new concept.
        
        Args:
            concept1, concept2: [batch_size, concept_dim]
        
        Returns:
            composed_concept: [batch_size, concept_dim]
        """
        if not TORCH_AVAILABLE:
            return concept1
        
        # Concatenate concepts
        pair = torch.cat([concept1, concept2], dim=-1)
        
        # Select composition operation
        op_logits = self.op_selector(pair)
        op_weights = F.softmax(op_logits, dim=-1)
        
        # Apply all operations
        compositions = []
        for op in self.composition_ops:
            comp = op(pair)
            compositions.append(comp)
        
        compositions = torch.stack(compositions, dim=1)  # [batch, num_ops, concept_dim]
        
        # Weighted combination
        op_weights = op_weights.unsqueeze(-1)  # [batch, num_ops, 1]
        composed = (compositions * op_weights).sum(dim=1)
        
        return composed


# ============================================================================
# DISENTANGLED REPRESENTATION LEARNING
# ============================================================================

class DisentangledVAE(nn.Module if TORCH_AVAILABLE else object):
    """
    Variational Autoencoder with disentangled latent factors.
    Based on β-VAE for learning interpretable concept representations.
    """
    
    def __init__(
        self,
        input_dim: int = 128,
        latent_dim: int = 64,
        hidden_dim: int = 128,
        beta: float = 4.0
    ):
        if TORCH_AVAILABLE:
            super().__init__()
        
        self.latent_dim = latent_dim
        self.beta = beta
        
        if TORCH_AVAILABLE:
            # Encoder
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU()
            )
            
            self.fc_mu = nn.Linear(hidden_dim, latent_dim)
            self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
            
            # Decoder
            self.decoder = nn.Sequential(
                nn.Linear(latent_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, input_dim)
            )
    
    def encode(self, x):
        """Encode input to latent distribution."""
        if not TORCH_AVAILABLE:
            return x, x
        
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        """Reparameterization trick."""
        if not TORCH_AVAILABLE:
            return mu
        
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        """Decode latent to reconstruction."""
        if not TORCH_AVAILABLE:
            return z
        
        return self.decoder(z)
    
    def forward(self, x):
        """Full forward pass."""
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar
    
    def loss_function(self, recon, x, mu, logvar):
        """β-VAE loss with disentanglement penalty."""
        if not TORCH_AVAILABLE:
            return 0.0
        
        # Reconstruction loss
        recon_loss = F.mse_loss(recon, x, reduction='sum')
        
        # KL divergence
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        
        # Total loss with β weighting on KL
        total_loss = recon_loss + self.beta * kl_loss
        
        return total_loss


class ConceptDisentangler:
    """
    Learns disentangled concept representations where each dimension
    corresponds to an interpretable factor of variation.
    """
    
    def __init__(
        self,
        input_dim: int = 128,
        num_factors: int = 10,
        factor_dim: int = 8,
        beta: float = 4.0
    ):
        self.input_dim = input_dim
        self.num_factors = num_factors
        self.factor_dim = factor_dim
        
        # Disentangled VAE
        if TORCH_AVAILABLE:
            self.vae = DisentangledVAE(
                input_dim=input_dim,
                latent_dim=num_factors * factor_dim,
                beta=beta
            )
        
        # Factor labels (interpretable names)
        self.factor_names: Dict[int, str] = {}
        
        # Learned factor statistics
        self.factor_statistics: Dict[int, Dict[str, float]] = {}
        
        self.logger = logging.getLogger(__name__)
    
    def learn_disentangled_factors(
        self,
        training_data: List[Any],
        num_epochs: int = 100
    ) -> Dict[int, str]:
        """
        Learn disentangled factors from data.
        
        Returns:
            Dictionary mapping factor indices to interpretable names
        """
        self.logger.info(f"Learning disentangled factors from {len(training_data)} examples")
        
        # Mock training process
        # In production, would train VAE with disentanglement metrics
        
        # Assign interpretable names to factors
        factor_names = [
            "size", "color", "shape", "position_x", "position_y",
            "rotation", "texture", "material", "state", "context"
        ]
        
        for i in range(min(self.num_factors, len(factor_names))):
            self.factor_names[i] = factor_names[i]
            self.factor_statistics[i] = {
                "mean": random.uniform(-1, 1),
                "std": random.uniform(0.5, 1.5),
                "min": random.uniform(-3, -1),
                "max": random.uniform(1, 3)
            }
        
        self.logger.info(f"Learned {len(self.factor_names)} disentangled factors")
        return self.factor_names
    
    def manipulate_concept(
        self,
        concept_embedding: List[float],
        factor_index: int,
        delta: float
    ) -> List[float]:
        """
        Manipulate a specific factor in concept representation.
        
        Args:
            concept_embedding: Original concept embedding
            factor_index: Which factor to manipulate
            delta: How much to change (positive or negative)
        
        Returns:
            Modified concept embedding
        """
        if not concept_embedding:
            return concept_embedding
        
        # Clone embedding
        modified = concept_embedding.copy()
        
        # Modify specific factor dimensions
        start_idx = factor_index * self.factor_dim
        end_idx = start_idx + self.factor_dim
        
        if end_idx <= len(modified):
            for i in range(start_idx, end_idx):
                modified[i] += delta
        
        return modified
    
    def get_factor_value(
        self,
        concept_embedding: List[float],
        factor_index: int
    ) -> float:
        """Extract value of a specific factor."""
        if not concept_embedding:
            return 0.0
        
        start_idx = factor_index * self.factor_dim
        end_idx = start_idx + self.factor_dim
        
        if end_idx <= len(concept_embedding):
            factor_values = concept_embedding[start_idx:end_idx]
            return sum(factor_values) / len(factor_values)
        
        return 0.0


# ============================================================================
# COMPOSITIONAL CONCEPT LEARNING ENGINE
# ============================================================================

class CompositionalConceptLearner:
    """
    Main engine for compositional concept learning.
    
    Integrates:
    - Object-centric representations
    - Relation networks
    - Concept composition
    - Disentangled learning
    - Hierarchical organization
    """
    
    def __init__(
        self,
        input_dim: int = 128,
        concept_dim: int = 64,
        num_slots: int = 7,
        num_factors: int = 10
    ):
        self.input_dim = input_dim
        self.concept_dim = concept_dim
        self.num_slots = num_slots
        self.num_factors = num_factors
        
        # Core components
        if TORCH_AVAILABLE:
            self.object_encoder = ObjectEncoder(
                input_dim=input_dim,
                num_slots=num_slots,
                slot_dim=concept_dim
            )
            
            self.relation_network = RelationNetwork(
                object_dim=concept_dim,
                relation_dim=concept_dim * 2,
                output_dim=concept_dim
            )
            
            self.composition_function = CompositionFunction(
                concept_dim=concept_dim
            )
        
        self.concept_disentangler = ConceptDisentangler(
            input_dim=input_dim,
            num_factors=num_factors,
            factor_dim=concept_dim // num_factors
        )
        
        # Concept storage
        self.concepts: Dict[str, Concept] = {}
        self.concept_relations: Dict[str, ConceptRelation] = {}
        self.hierarchies: Dict[str, ConceptHierarchy] = {}
        
        # Object instances
        self.objects: Dict[str, ObjectRepresentation] = {}
        
        self.logger = logging.getLogger(__name__)
    
    def perceive_objects(
        self,
        perceptual_input: Any,
        num_objects: Optional[int] = None
    ) -> List[ObjectRepresentation]:
        """
        Extract object-centric representations from perceptual input.
        
        Args:
            perceptual_input: Raw sensory input
            num_objects: Expected number of objects (None = auto-detect)
        
        Returns:
            List of object representations
        """
        self.logger.info("Extracting object-centric representations")
        
        # Convert input to tensor if needed
        if TORCH_AVAILABLE and not isinstance(perceptual_input, torch.Tensor):
            if isinstance(perceptual_input, (list, np.ndarray)):
                perceptual_input = torch.tensor(perceptual_input, dtype=torch.float32)
        
        # Extract object slots
        if TORCH_AVAILABLE:
            if len(perceptual_input.shape) == 1:
                perceptual_input = perceptual_input.unsqueeze(0)
            
            with torch.no_grad():
                slots = self.object_encoder(perceptual_input)
                slots_np = slots.squeeze(0).numpy()
        else:
            # Mock extraction
            slots_np = np.random.randn(self.num_slots, self.concept_dim)
        
        # Create object representations
        objects = []
        for i in range(self.num_slots if num_objects is None else num_objects):
            obj = ObjectRepresentation(
                object_id=f"obj_{datetime.utcnow().timestamp()}_{i}",
                concept_id="unknown",
                slots={
                    f"slot_{j}": Slot(
                        slot_id=f"slot_{j}",
                        slot_type="feature",
                        value=slots_np[i][j] if i < len(slots_np) else 0.0,
                        confidence=0.8
                    )
                    for j in range(min(self.concept_dim, len(slots_np[i]) if i < len(slots_np) else 0))
                },
                binding_strength=0.8
            )
            objects.append(obj)
            self.objects[obj.object_id] = obj
        
        self.logger.info(f"Extracted {len(objects)} object representations")
        return objects
    
    def learn_concept(
        self,
        concept_name: str,
        concept_type: ConceptType,
        examples: List[Any],
        is_primitive: bool = True
    ) -> Concept:
        """
        Learn a new concept from examples.
        
        Args:
            concept_name: Human-readable name
            concept_type: Type of concept
            examples: Training examples
            is_primitive: Whether concept is primitive or composite
        
        Returns:
            Learned concept
        """
        self.logger.info(f"Learning concept: {concept_name} ({concept_type.value})")
        
        # Extract features from examples
        embeddings = []
        for example in examples:
            # Process example to get embedding
            if TORCH_AVAILABLE and isinstance(example, (list, np.ndarray)):
                emb = torch.tensor(example, dtype=torch.float32)
                embeddings.append(emb.numpy().tolist())
            elif isinstance(example, dict) and 'embedding' in example:
                embeddings.append(example['embedding'])
            else:
                # Generate random embedding for mock
                embeddings.append([random.random() for _ in range(self.concept_dim)])
        
        # Average embeddings
        if embeddings:
            avg_embedding = [
                sum(emb[i] for emb in embeddings) / len(embeddings)
                for i in range(len(embeddings[0]))
            ]
        else:
            avg_embedding = [random.random() for _ in range(self.concept_dim)]
        
        # Create concept
        concept = Concept(
            concept_id=f"concept_{datetime.utcnow().timestamp()}",
            concept_type=concept_type,
            name=concept_name,
            embedding=avg_embedding,
            is_primitive=is_primitive,
            examples_seen=len(examples),
            confidence=min(0.5 + len(examples) * 0.05, 0.95),
            human_description=f"Concept representing {concept_name}"
        )
        
        # Store concept
        self.concepts[concept.concept_id] = concept
        
        self.logger.info(f"Learned concept {concept.concept_id} with confidence {concept.confidence:.2f}")
        return concept
    
    def compose_concepts(
        self,
        concept1_id: str,
        concept2_id: str,
        composition_name: str,
        operation: str = "combine"
    ) -> Concept:
        """
        Compose two concepts into a new composite concept.
        
        Args:
            concept1_id: First concept ID
            concept2_id: Second concept ID
            composition_name: Name for new concept
            operation: Composition operation
        
        Returns:
            Composite concept
        """
        if concept1_id not in self.concepts or concept2_id not in self.concepts:
            raise ValueError("One or both concepts not found")
        
        concept1 = self.concepts[concept1_id]
        concept2 = self.concepts[concept2_id]
        
        self.logger.info(f"Composing {concept1.name} + {concept2.name} = {composition_name}")
        
        # Compose embeddings
        if TORCH_AVAILABLE:
            emb1 = torch.tensor(concept1.embedding, dtype=torch.float32).unsqueeze(0)
            emb2 = torch.tensor(concept2.embedding, dtype=torch.float32).unsqueeze(0)
            
            with torch.no_grad():
                composed_emb = self.composition_function(emb1, emb2)
                composed_emb = composed_emb.squeeze(0).numpy().tolist()
        else:
            # Mock composition
            composed_emb = [
                (concept1.embedding[i] + concept2.embedding[i]) / 2
                for i in range(min(len(concept1.embedding), len(concept2.embedding)))
            ]
        
        # Create composite concept
        composite = Concept(
            concept_id=f"concept_{datetime.utcnow().timestamp()}",
            concept_type=ConceptType.COMPOSITE,
            name=composition_name,
            embedding=composed_emb,
            is_primitive=False,
            composed_from=[concept1_id, concept2_id],
            composition_operation=operation,
            confidence=(concept1.confidence + concept2.confidence) / 2,
            abstraction_level=max(concept1.abstraction_level, concept2.abstraction_level) + 1,
            human_description=f"{concept1.name} {operation} {concept2.name}"
        )
        
        # Store composite
        self.concepts[composite.concept_id] = composite
        
        # Update usage statistics
        concept1.successful_compositions += 1
        concept2.successful_compositions += 1
        
        self.logger.info(f"Created composite concept {composite.concept_id}")
        return composite
    
    def discover_relations(
        self,
        object_ids: List[str]
    ) -> List[ConceptRelation]:
        """
        Discover relations between objects using relation network.
        
        Args:
            object_ids: List of object IDs to analyze
        
        Returns:
            List of discovered relations
        """
        if not object_ids:
            return []
        
        self.logger.info(f"Discovering relations among {len(object_ids)} objects")
        
        # Get object embeddings
        object_embeddings = []
        for obj_id in object_ids:
            if obj_id in self.objects:
                obj = self.objects[obj_id]
                # Extract embedding from slots
                emb = [slot.value for slot in obj.slots.values() if isinstance(slot.value, (int, float))]
                object_embeddings.append(emb[:self.concept_dim] + [0.0] * (self.concept_dim - len(emb)))
        
        if not object_embeddings:
            return []
        
        # Use relation network
        if TORCH_AVAILABLE and len(object_embeddings) > 1:
            objects_tensor = torch.tensor(object_embeddings, dtype=torch.float32).unsqueeze(0)
            
            with torch.no_grad():
                relation_output = self.relation_network(objects_tensor)
            
            # Interpret relation output (mock)
            # In production, would have relation classifier
        
        # Create relations
        relations = []
        for i in range(len(object_ids)):
            for j in range(i + 1, len(object_ids)):
                relation = ConceptRelation(
                    relation_id=f"rel_{datetime.utcnow().timestamp()}_{i}_{j}",
                    relation_type="spatial_proximity" if random.random() > 0.5 else "semantic_similarity",
                    source_concept=object_ids[i],
                    target_concept=object_ids[j],
                    strength=random.uniform(0.5, 1.0),
                    confidence=random.uniform(0.6, 0.9)
                )
                relations.append(relation)
                self.concept_relations[relation.relation_id] = relation
        
        self.logger.info(f"Discovered {len(relations)} relations")
        return relations
    
    def build_concept_hierarchy(
        self,
        root_concept_id: str,
        strategy: str = "abstraction_based"
    ) -> ConceptHierarchy:
        """
        Build hierarchical organization of concepts.
        
        Args:
            root_concept_id: Root of the hierarchy
            strategy: How to organize ("abstraction_based", "composition_based")
        
        Returns:
            Concept hierarchy
        """
        if root_concept_id not in self.concepts:
            raise ValueError(f"Root concept {root_concept_id} not found")
        
        self.logger.info(f"Building concept hierarchy from {root_concept_id}")
        
        hierarchy = ConceptHierarchy(
            hierarchy_id=f"hierarchy_{datetime.utcnow().timestamp()}",
            root_concept=root_concept_id
        )
        
        # Organize concepts by abstraction level
        if strategy == "abstraction_based":
            for concept_id, concept in self.concepts.items():
                level = concept.abstraction_level
                hierarchy.level_map[concept_id] = level
                
                # Find parent (concept at level - 1 with highest similarity)
                if level > 0:
                    parent_candidates = [
                        cid for cid, c in self.concepts.items()
                        if c.abstraction_level == level - 1
                    ]
                    if parent_candidates:
                        # Mock: choose first candidate
                        parent = parent_candidates[0]
                        hierarchy.parent_map[concept_id] = parent
                        
                        if parent not in hierarchy.children_map:
                            hierarchy.children_map[parent] = []
                        hierarchy.children_map[parent].append(concept_id)
        
        # Organize by composition
        elif strategy == "composition_based":
            for concept_id, concept in self.concepts.items():
                if not concept.is_primitive and concept.composed_from:
                    # Parents are the components
                    for component_id in concept.composed_from:
                        hierarchy.parent_map[component_id] = concept_id
                        
                        if concept_id not in hierarchy.children_map:
                            hierarchy.children_map[concept_id] = []
                        hierarchy.children_map[concept_id].append(component_id)
        
        hierarchy.num_concepts = len(hierarchy.level_map)
        hierarchy.max_depth = max(hierarchy.level_map.values()) if hierarchy.level_map else 0
        
        self.hierarchies[hierarchy.hierarchy_id] = hierarchy
        
        self.logger.info(
            f"Built hierarchy with {hierarchy.num_concepts} concepts, "
            f"max depth {hierarchy.max_depth}"
        )
        
        return hierarchy
    
    def abstract_reasoning(
        self,
        query: str,
        context_objects: List[str]
    ) -> Dict[str, Any]:
        """
        Perform abstract reasoning over learned symbolic structures.
        
        Args:
            query: Reasoning query (e.g., "What is common between these objects?")
            context_objects: Object IDs to reason about
        
        Returns:
            Reasoning result
        """
        self.logger.info(f"Abstract reasoning: {query}")
        
        # Get objects
        objects = [self.objects[oid] for oid in context_objects if oid in self.objects]
        
        if not objects:
            return {"status": "no_objects", "result": None}
        
        # Extract concepts
        concepts = [
            self.concepts[obj.concept_id]
            for obj in objects
            if obj.concept_id in self.concepts
        ]
        
        result = {
            "query": query,
            "num_objects": len(objects),
            "num_concepts": len(concepts)
        }
        
        # Different reasoning strategies based on query
        if "common" in query.lower():
            # Find common attributes
            if concepts:
                # Find shared abstraction level
                levels = [c.abstraction_level for c in concepts]
                result["common_abstraction_level"] = max(set(levels), key=levels.count)
                
                # Find shared attributes
                all_attrs = set()
                for c in concepts:
                    all_attrs.update(c.attributes.keys())
                
                common_attrs = {}
                for attr in all_attrs:
                    values = [c.attributes.get(attr) for c in concepts if attr in c.attributes]
                    if len(values) == len(concepts) and len(set(map(str, values))) == 1:
                        common_attrs[attr] = values[0]
                
                result["common_attributes"] = common_attrs
                result["reasoning"] = f"Found {len(common_attrs)} common attributes"
        
        elif "relation" in query.lower() or "relationship" in query.lower():
            # Discover relations
            relations = self.discover_relations(context_objects)
            result["discovered_relations"] = [
                {
                    "type": r.relation_type,
                    "strength": r.strength,
                    "confidence": r.confidence
                }
                for r in relations
            ]
            result["reasoning"] = f"Discovered {len(relations)} relations"
        
        elif "compose" in query.lower() or "combination" in query.lower():
            # Suggest compositions
            if len(concepts) >= 2:
                # Try composing first two concepts
                comp = self.compose_concepts(
                    concepts[0].concept_id,
                    concepts[1].concept_id,
                    f"{concepts[0].name}_{concepts[1].name}",
                    operation="combine"
                )
                result["composed_concept"] = {
                    "id": comp.concept_id,
                    "name": comp.name,
                    "confidence": comp.confidence
                }
                result["reasoning"] = f"Composed new concept: {comp.name}"
        
        self.logger.info(f"Reasoning complete: {result.get('reasoning', 'No specific reasoning')}")
        return result
    
    def get_concept_explanation(self, concept_id: str) -> str:
        """Generate human-interpretable explanation of a concept."""
        if concept_id not in self.concepts:
            return f"Concept {concept_id} not found"
        
        concept = self.concepts[concept_id]
        
        explanation = f"Concept: {concept.name}\n"
        explanation += f"Type: {concept.concept_type.value}\n"
        explanation += f"Confidence: {concept.confidence:.2%}\n"
        explanation += f"Abstraction Level: {concept.abstraction_level}\n"
        
        if concept.is_primitive:
            explanation += "Status: Primitive (cannot be decomposed)\n"
        else:
            explanation += "Status: Composite\n"
            if concept.composed_from:
                component_names = [
                    self.concepts[cid].name
                    for cid in concept.composed_from
                    if cid in self.concepts
                ]
                explanation += f"Composed from: {', '.join(component_names)}\n"
                explanation += f"Composition: {concept.composition_operation}\n"
        
        if concept.attributes:
            explanation += f"Attributes: {', '.join(concept.attributes.keys())}\n"
        
        if concept.relations:
            explanation += f"Relations: {len(concept.relations)}\n"
        
        explanation += f"\nDescription: {concept.human_description}\n"
        explanation += f"Examples seen: {concept.examples_seen}\n"
        explanation += f"Usage count: {concept.usage_count}\n"
        
        return explanation
    
    def visualize_hierarchy(self, hierarchy_id: str) -> str:
        """Generate ASCII visualization of concept hierarchy."""
        if hierarchy_id not in self.hierarchies:
            return f"Hierarchy {hierarchy_id} not found"
        
        hierarchy = self.hierarchies[hierarchy_id]
        
        def build_tree(concept_id: str, depth: int = 0) -> str:
            if concept_id not in self.concepts:
                return ""
            
            concept = self.concepts[concept_id]
            indent = "  " * depth
            tree = f"{indent}├─ {concept.name} ({concept.concept_type.value})\n"
            
            # Add children
            if concept_id in hierarchy.children_map:
                for child_id in hierarchy.children_map[concept_id]:
                    tree += build_tree(child_id, depth + 1)
            
            return tree
        
        visualization = f"Concept Hierarchy: {hierarchy.hierarchy_id}\n"
        visualization += f"Total Concepts: {hierarchy.num_concepts}\n"
        visualization += f"Max Depth: {hierarchy.max_depth}\n\n"
        visualization += build_tree(hierarchy.root_concept)
        
        return visualization


# ============================================================================
# FACTORY FUNCTIONS
# ============================================================================

def create_compositional_concept_learner(
    input_dim: int = 128,
    concept_dim: int = 64,
    num_slots: int = 7,
    num_factors: int = 10
) -> CompositionalConceptLearner:
    """
    Factory function to create compositional concept learner.
    
    Args:
        input_dim: Dimension of perceptual input
        concept_dim: Dimension of concept embeddings
        num_slots: Number of object slots
        num_factors: Number of disentangled factors
    
    Returns:
        Configured CompositionalConceptLearner
    """
    return CompositionalConceptLearner(
        input_dim=input_dim,
        concept_dim=concept_dim,
        num_slots=num_slots,
        num_factors=num_factors
    )


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Create learner
    learner = create_compositional_concept_learner()
    
    # Example: Perceive objects
    print("\n=== Object Perception ===")
    scene_input = np.random.randn(128)
    objects = learner.perceive_objects(scene_input, num_objects=3)
    print(f"Perceived {len(objects)} objects")
    
    # Example: Learn primitive concepts
    print("\n=== Learning Primitive Concepts ===")
    color_examples = [{"embedding": [random.random() for _ in range(64)]} for _ in range(5)]
    color_concept = learner.learn_concept("red", ConceptType.ATTRIBUTE, color_examples, is_primitive=True)
    print(f"Learned concept: {color_concept.name} (confidence: {color_concept.confidence:.2%})")
    
    shape_examples = [{"embedding": [random.random() for _ in range(64)]} for _ in range(5)]
    shape_concept = learner.learn_concept("circle", ConceptType.ATTRIBUTE, shape_examples, is_primitive=True)
    print(f"Learned concept: {shape_concept.name} (confidence: {shape_concept.confidence:.2%})")
    
    # Example: Compose concepts
    print("\n=== Composing Concepts ===")
    composite = learner.compose_concepts(
        color_concept.concept_id,
        shape_concept.concept_id,
        "red_circle",
        operation="combine"
    )
    print(f"Composed concept: {composite.name}")
    print(f"Components: {composite.composed_from}")
    
    # Example: Build hierarchy
    print("\n=== Building Concept Hierarchy ===")
    hierarchy = learner.build_concept_hierarchy(composite.concept_id, strategy="composition_based")
    print(f"Hierarchy depth: {hierarchy.max_depth}")
    print(f"Total concepts: {hierarchy.num_concepts}")
    
    # Example: Abstract reasoning
    print("\n=== Abstract Reasoning ===")
    object_ids = [obj.object_id for obj in objects[:2]]
    result = learner.abstract_reasoning("What is common between these objects?", object_ids)
    print(f"Reasoning result: {result.get('reasoning', 'N/A')}")
    
    print("\n=== System Ready ===")
