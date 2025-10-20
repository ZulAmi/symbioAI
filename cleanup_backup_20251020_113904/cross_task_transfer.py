"""
Cross-Task Transfer Learning Engine - Symbio AI

Automatically discovers and exploits knowledge transfer patterns across tasks.
Uses graph neural networks to model task relationships, generates automatic
curricula, performs meta-knowledge distillation, and enables zero-shot synthesis.

This system goes beyond traditional transfer learning to automatic discovery
of transferability patterns and knowledge synthesis across domains.
"""

import asyncio
import logging
import random
import json
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Callable, Set
from dataclasses import dataclass, field, asdict, is_dataclass
from datetime import datetime
from collections import defaultdict
from enum import Enum

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    # Mock torch for development
    class torch:
        class nn:
            class Module:
                pass
            class Linear:
                def __init__(self, *args, **kwargs): pass
            class GRU:
                def __init__(self, *args, **kwargs): pass
        class Tensor:
            pass
        @staticmethod
        def tensor(x): return x
        @staticmethod
        def zeros(*args, **kwargs): return [0.0]
        @staticmethod
        def randn(*args, **kwargs): return [random.random()]
    
    class F:
        @staticmethod
        def cosine_similarity(x, y, dim=0): return random.random()
        @staticmethod
        def relu(x): return max(0, x) if isinstance(x, (int, float)) else x
        @staticmethod
        def softmax(x, dim=-1): return x

from deployment.observability import OBSERVABILITY


class TaskRelationType(Enum):
    """Types of relationships between tasks."""
    IDENTICAL = "identical"  # Same task
    HIGHLY_SIMILAR = "highly_similar"  # Very related
    MODERATELY_SIMILAR = "moderately_similar"  # Somewhat related
    WEAKLY_SIMILAR = "weakly_similar"  # Distantly related
    COMPLEMENTARY = "complementary"  # Different but synergistic
    INDEPENDENT = "independent"  # No clear relationship
    ANTAGONISTIC = "antagonistic"  # Negatively related


class TransferDirection(Enum):
    """Direction of knowledge transfer."""
    UNIDIRECTIONAL = "unidirectional"  # A → B only
    BIDIRECTIONAL = "bidirectional"  # A ↔ B
    MULTI_SOURCE = "multi_source"  # Multiple → One
    MULTI_TARGET = "multi_target"  # One → Multiple
    UNIVERSAL = "universal"  # All ↔ All


class CurriculumStrategy(Enum):
    """Strategies for curriculum generation."""
    EASY_TO_HARD = "easy_to_hard"
    DIVERSE_SAMPLING = "diverse_sampling"
    UNCERTAINTY_DRIVEN = "uncertainty_driven"
    TRANSFER_POTENTIAL = "transfer_potential"
    ADAPTIVE_DIFFICULTY = "adaptive_difficulty"


@dataclass
class TaskDescriptor:
    """Comprehensive task description for transfer learning."""
    task_id: str
    task_name: str
    task_type: str  # classification, regression, generation, etc.
    domain: str  # vision, nlp, audio, etc.
    
    # Task characteristics
    input_dimensionality: int = 0
    output_dimensionality: int = 0
    sample_complexity: float = 0.0  # Estimated samples needed
    computational_complexity: float = 0.0  # Compute requirements
    
    # Semantic information
    task_description: str = ""
    required_skills: List[str] = field(default_factory=list)
    domain_knowledge: List[str] = field(default_factory=list)
    
    # Performance history
    best_performance: float = 0.0
    current_performance: float = 0.0
    training_history: List[Dict[str, Any]] = field(default_factory=list)
    
    # Transfer metadata
    source_tasks: List[str] = field(default_factory=list)  # Tasks this learned from
    target_tasks: List[str] = field(default_factory=list)  # Tasks that learned from this
    transfer_efficiency: Dict[str, float] = field(default_factory=dict)  # Task -> efficiency
    
    # Embedding
    task_embedding: Optional[List[float]] = None  # Learned representation
    
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())


@dataclass
class TransferEdge:
    """Represents transfer relationship between two tasks."""
    source_task: str
    target_task: str
    relation_type: TaskRelationType
    transfer_direction: TransferDirection
    
    # Transfer metrics
    transfer_coefficient: float = 0.0  # How well knowledge transfers (0-1)
    performance_gain: float = 0.0  # Performance improvement from transfer
    sample_efficiency_gain: float = 0.0  # Reduction in samples needed
    convergence_speed_gain: float = 0.0  # Faster training
    
    # Transfer mechanism
    shared_representations: List[str] = field(default_factory=list)
    transferred_layers: List[int] = field(default_factory=list)
    adaptation_required: float = 0.0  # How much fine-tuning needed
    
    # Statistics
    transfer_attempts: int = 0
    successful_transfers: int = 0
    last_transfer: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    
    # Learned insights
    optimal_transfer_strategy: str = ""
    failure_modes: List[str] = field(default_factory=list)


@dataclass
class Curriculum:
    """Automatically generated learning curriculum."""
    curriculum_id: str
    target_task: str
    strategy: CurriculumStrategy
    
    # Curriculum sequence
    task_sequence: List[str] = field(default_factory=list)  # Ordered tasks
    task_difficulties: Dict[str, float] = field(default_factory=dict)
    task_dependencies: Dict[str, List[str]] = field(default_factory=dict)
    
    # Performance tracking
    expected_performance: float = 0.0
    actual_performance: float = 0.0
    curriculum_efficiency: float = 0.0  # Vs. direct training
    
    # Adaptation
    adaptive_adjustments: int = 0
    skipped_tasks: List[str] = field(default_factory=list)
    repeated_tasks: List[str] = field(default_factory=list)
    
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    completed_at: Optional[str] = None


@dataclass
class MetaKnowledge:
    """Meta-knowledge distilled across domains."""
    knowledge_id: str
    knowledge_type: str  # "representation", "strategy", "prior", "invariance"
    
    # Source information
    source_tasks: List[str] = field(default_factory=list)
    source_domains: List[str] = field(default_factory=list)
    
    # Knowledge content
    knowledge_embedding: Optional[List[float]] = None
    applicable_conditions: List[str] = field(default_factory=list)
    incompatible_conditions: List[str] = field(default_factory=list)
    
    # Performance
    generalization_score: float = 0.0  # How well it generalizes
    applicability_count: int = 0  # How many tasks it helps
    average_improvement: float = 0.0  # Average performance gain
    
    # Evolution
    refinement_count: int = 0
    last_updated: str = field(default_factory=lambda: datetime.utcnow().isoformat())


class TaskRelationshipGraph(nn.Module):
    """
    Graph Neural Network for modeling task relationships and transfer patterns.
    Learns embeddings that capture task similarity and transferability.
    """
    
    def __init__(
        self,
        task_embedding_dim: int = 128,
        hidden_dim: int = 256,
        num_relation_types: int = 7,
        num_layers: int = 3
    ):
        super().__init__()
        
        self.embedding_dim = task_embedding_dim
        self.hidden_dim = hidden_dim
        
        # Task embedding layers
        self.task_encoder = nn.Linear(task_embedding_dim, hidden_dim)
        
        # Graph convolution layers (message passing)
        self.graph_layers = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers)
        ])
        
        # Relation-specific transformations
        self.relation_transforms = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim) for _ in range(num_relation_types)
        ])
        
        # Transfer prediction
        self.transfer_predictor = nn.Linear(hidden_dim * 2, 1)
        
        # Curriculum generation
        self.difficulty_predictor = nn.Linear(hidden_dim, 1)
        self.prerequisite_predictor = nn.Linear(hidden_dim * 2, 1)
        
        self.logger = logging.getLogger(__name__)
    
    def forward(
        self,
        task_features: torch.Tensor,
        adjacency_matrix: torch.Tensor,
        relation_types: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the task relationship graph.
        
        Args:
            task_features: [num_tasks, embedding_dim]
            adjacency_matrix: [num_tasks, num_tasks]
            relation_types: [num_tasks, num_tasks] (relation type indices)
        
        Returns:
            Dict with task embeddings, transfer predictions, etc.
        """
        # Encode tasks
        h = F.relu(self.task_encoder(task_features))  # [num_tasks, hidden_dim]
        
        # Graph convolution (message passing)
        for layer in self.graph_layers:
            # Aggregate neighbor information
            messages = torch.matmul(adjacency_matrix, h)  # [num_tasks, hidden_dim]
            h = F.relu(layer(messages)) + h  # Residual connection
        
        task_embeddings = h
        
        # Predict transfer coefficients between all pairs
        num_tasks = task_embeddings.size(0)
        transfer_predictions = torch.zeros(num_tasks, num_tasks)
        
        for i in range(num_tasks):
            for j in range(num_tasks):
                if i != j:
                    # Concatenate source and target embeddings
                    pair_embedding = torch.cat([task_embeddings[i], task_embeddings[j]])
                    transfer_predictions[i, j] = torch.sigmoid(
                        self.transfer_predictor(pair_embedding)
                    )
        
        # Predict task difficulties
        difficulties = torch.sigmoid(self.difficulty_predictor(task_embeddings))
        
        return {
            "task_embeddings": task_embeddings,
            "transfer_predictions": transfer_predictions,
            "task_difficulties": difficulties
        }
    
    def predict_transfer_efficiency(
        self,
        source_embedding: torch.Tensor,
        target_embedding: torch.Tensor
    ) -> float:
        """Predict how efficiently knowledge transfers between tasks."""
        pair = torch.cat([source_embedding, target_embedding])
        return torch.sigmoid(self.transfer_predictor(pair)).item()
    
    def generate_curriculum_order(
        self,
        task_embeddings: torch.Tensor,
        target_task_idx: int,
        strategy: CurriculumStrategy = CurriculumStrategy.EASY_TO_HARD
    ) -> List[int]:
        """Generate optimal task ordering for curriculum learning."""
        num_tasks = task_embeddings.size(0)
        target_emb = task_embeddings[target_task_idx]
        
        if strategy == CurriculumStrategy.EASY_TO_HARD:
            # Order by difficulty, easiest first
            difficulties = self.difficulty_predictor(task_embeddings).squeeze()
            return torch.argsort(difficulties).tolist()
        
        elif strategy == CurriculumStrategy.TRANSFER_POTENTIAL:
            # Order by transfer potential to target task
            transfer_scores = []
            for i in range(num_tasks):
                if i != target_task_idx:
                    score = self.predict_transfer_efficiency(
                        task_embeddings[i], target_emb
                    )
                    transfer_scores.append((i, score))
            
            # Sort by transfer score descending
            transfer_scores.sort(key=lambda x: x[1], reverse=True)
            return [idx for idx, _ in transfer_scores]
        
        else:
            # Default: random ordering
            indices = list(range(num_tasks))
            random.shuffle(indices)
            return indices


class MetaKnowledgeDistiller:
    """
    Distills meta-knowledge across domains using advanced distillation techniques.
    Extracts domain-invariant representations and transferable strategies.
    """
    
    def __init__(
        self,
        embedding_dim: int = 128,
        temperature: float = 2.0,
        alpha: float = 0.7
    ):
        self.embedding_dim = embedding_dim
        self.temperature = temperature
        self.alpha = alpha  # Weight for distillation loss
        
        # Meta-knowledge repository
        self.knowledge_base: Dict[str, MetaKnowledge] = {}
        
        # Domain-specific encoders
        self.domain_encoders: Dict[str, nn.Module] = {}
        
        # Universal encoder (cross-domain)
        self.universal_encoder = nn.Linear(embedding_dim, embedding_dim)
        
        self.logger = logging.getLogger(__name__)
    
    def distill_from_tasks(
        self,
        task_models: Dict[str, nn.Module],
        task_descriptors: Dict[str, TaskDescriptor],
        distillation_samples: Dict[str, List[Any]]
    ) -> MetaKnowledge:
        """
        Distill meta-knowledge from multiple task-specific models.
        
        Args:
            task_models: Task ID -> trained model
            task_descriptors: Task ID -> descriptor
            distillation_samples: Task ID -> sample data
        
        Returns:
            MetaKnowledge instance
        """
        self.logger.info(f"Distilling meta-knowledge from {len(task_models)} tasks")
        
        # Extract representations from each task
        task_representations = {}
        for task_id, model in task_models.items():
            if task_id in distillation_samples:
                samples = distillation_samples[task_id][:10]  # Sample for efficiency
                representations = self._extract_representations(model, samples)
                task_representations[task_id] = representations
        
        # Find common patterns across tasks
        common_patterns = self._find_common_patterns(task_representations)
        
        # Create meta-knowledge
        knowledge = MetaKnowledge(
            knowledge_id=f"meta_knowledge_{datetime.utcnow().timestamp()}",
            knowledge_type="representation",
            source_tasks=list(task_models.keys()),
            source_domains=list(set(d.domain for d in task_descriptors.values())),
            knowledge_embedding=common_patterns,
            generalization_score=self._estimate_generalization(common_patterns),
            applicability_count=len(task_models)
        )
        
        self.knowledge_base[knowledge.knowledge_id] = knowledge
        
        self.logger.info(
            f"Created meta-knowledge {knowledge.knowledge_id} "
            f"with generalization score {knowledge.generalization_score:.3f}"
        )
        
        return knowledge
    
    def _extract_representations(
        self,
        model: nn.Module,
        samples: List[Any]
    ) -> List[float]:
        """Extract intermediate representations from a model."""
        # Mock implementation
        # In production, would use hooks to extract activations
        return [random.random() for _ in range(self.embedding_dim)]
    
    def _find_common_patterns(
        self,
        task_representations: Dict[str, List[float]]
    ) -> List[float]:
        """Find common patterns across task representations."""
        if not task_representations:
            return [0.0] * self.embedding_dim
        
        # Mock: Average representations
        all_reps = list(task_representations.values())
        avg_rep = [
            sum(rep[i] for rep in all_reps) / len(all_reps)
            for i in range(len(all_reps[0]))
        ]
        return avg_rep
    
    def _estimate_generalization(self, knowledge_embedding: List[float]) -> float:
        """Estimate how well knowledge will generalize."""
        # Mock: Based on embedding variance
        variance = np.var(knowledge_embedding) if knowledge_embedding else 0.0
        # Higher variance = more general knowledge
        return min(1.0, variance * 10)
    
    def apply_meta_knowledge(
        self,
        target_model: nn.Module,
        knowledge: MetaKnowledge,
        adaptation_rate: float = 0.1
    ) -> nn.Module:
        """Apply meta-knowledge to a target model."""
        self.logger.info(
            f"Applying meta-knowledge {knowledge.knowledge_id} to target model"
        )
        
        # Mock: In production, would modify model weights based on knowledge
        # Could use techniques like:
        # - Weight initialization from knowledge embedding
        # - Adding learned priors
        # - Modifying learning dynamics
        
        return target_model


class ZeroShotTaskSynthesizer:
    """
    Synthesizes solutions for new tasks without training by combining
    knowledge from related tasks (zero-shot task synthesis).
    """
    
    def __init__(
        self,
        task_graph: TaskRelationshipGraph,
        knowledge_distiller: MetaKnowledgeDistiller
    ):
        self.task_graph = task_graph
        self.knowledge_distiller = knowledge_distiller
        
        # Synthesis strategies
        self.synthesis_strategies = {
            "weighted_ensemble": self._weighted_ensemble_synthesis,
            "knowledge_composition": self._knowledge_composition_synthesis,
            "analogy_transfer": self._analogy_transfer_synthesis
        }
        
        self.logger = logging.getLogger(__name__)
    
    async def synthesize_for_new_task(
        self,
        new_task: TaskDescriptor,
        available_tasks: Dict[str, TaskDescriptor],
        task_models: Dict[str, nn.Module],
        strategy: str = "weighted_ensemble"
    ) -> nn.Module:
        """
        Synthesize a model for a new task without training.
        
        Args:
            new_task: Descriptor for the new task
            available_tasks: Available tasks with trained models
            task_models: Task ID -> trained model
            strategy: Synthesis strategy to use
        
        Returns:
            Synthesized model for the new task
        """
        self.logger.info(
            f"Synthesizing zero-shot solution for task '{new_task.task_name}'"
        )
        
        # Find most related tasks
        related_tasks = self._find_related_tasks(new_task, available_tasks)
        
        self.logger.info(
            f"Found {len(related_tasks)} related tasks: {list(related_tasks.keys())}"
        )
        
        # Select synthesis strategy
        synthesis_fn = self.synthesis_strategies.get(
            strategy,
            self._weighted_ensemble_synthesis
        )
        
        # Synthesize model
        synthesized_model = await synthesis_fn(
            new_task, related_tasks, task_models
        )
        
        self.logger.info("Zero-shot synthesis complete")
        
        return synthesized_model
    
    def _find_related_tasks(
        self,
        new_task: TaskDescriptor,
        available_tasks: Dict[str, TaskDescriptor],
        top_k: int = 5
    ) -> Dict[str, TaskDescriptor]:
        """Find tasks most related to the new task."""
        # Mock: In production, would use task embeddings for similarity
        similarities = {}
        
        for task_id, task in available_tasks.items():
            # Simple heuristic: same domain and type
            similarity = 0.0
            if task.domain == new_task.domain:
                similarity += 0.5
            if task.task_type == new_task.task_type:
                similarity += 0.3
            
            # Skill overlap
            common_skills = set(task.required_skills) & set(new_task.required_skills)
            if common_skills:
                similarity += 0.2 * (len(common_skills) / max(
                    len(task.required_skills), len(new_task.required_skills)
                ))
            
            similarities[task_id] = similarity
        
        # Sort and take top-k
        sorted_tasks = sorted(
            similarities.items(), key=lambda x: x[1], reverse=True
        )[:top_k]
        
        return {
            task_id: available_tasks[task_id]
            for task_id, _ in sorted_tasks if similarities[task_id] > 0
        }
    
    async def _weighted_ensemble_synthesis(
        self,
        new_task: TaskDescriptor,
        related_tasks: Dict[str, TaskDescriptor],
        task_models: Dict[str, nn.Module]
    ) -> nn.Module:
        """Synthesize by weighted ensemble of related models."""
        self.logger.info("Using weighted ensemble synthesis")
        
        # Mock: Create an ensemble model
        # In production, would combine models with learned weights
        
        # For now, return a reference to the most related model
        if related_tasks and task_models:
            most_related = list(related_tasks.keys())[0]
            if most_related in task_models:
                return task_models[most_related]
        
        # Fallback: create a new model
        return nn.Linear(10, 10)  # Mock model
    
    async def _knowledge_composition_synthesis(
        self,
        new_task: TaskDescriptor,
        related_tasks: Dict[str, TaskDescriptor],
        task_models: Dict[str, nn.Module]
    ) -> nn.Module:
        """Synthesize by composing meta-knowledge."""
        self.logger.info("Using knowledge composition synthesis")
        
        # Extract meta-knowledge from related tasks
        # Compose into solution for new task
        
        return nn.Linear(10, 10)  # Mock
    
    async def _analogy_transfer_synthesis(
        self,
        new_task: TaskDescriptor,
        related_tasks: Dict[str, TaskDescriptor],
        task_models: Dict[str, nn.Module]
    ) -> nn.Module:
        """Synthesize using analogical reasoning."""
        self.logger.info("Using analogy transfer synthesis")
        
        # Find structural analogies between tasks
        # Transfer solution structure
        
        return nn.Linear(10, 10)  # Mock


class CrossTaskTransferEngine:
    @staticmethod
    def _make_json_safe(obj: Any) -> Any:
        """Convert objects with complex types (Enum, datetime, dataclass) into JSON-safe values."""
        if is_dataclass(obj):
            obj = asdict(obj)

        if isinstance(obj, Enum):
            return obj.value

        if isinstance(obj, datetime):
            return obj.isoformat()

        if isinstance(obj, Path):
            return str(obj)

        if isinstance(obj, dict):
            return {key: CrossTaskTransferEngine._make_json_safe(value) for key, value in obj.items()}

        if isinstance(obj, (list, tuple, set)):
            return [CrossTaskTransferEngine._make_json_safe(value) for value in obj]

        return obj

    """
    Main engine for cross-task transfer learning.
    
    Integrates task relationship modeling, curriculum generation,
    meta-knowledge distillation, and zero-shot synthesis.
    """
    
    def __init__(
        self,
        task_embedding_dim: int = 128,
        hidden_dim: int = 256,
        auto_discover: bool = True
    ):
        self.task_embedding_dim = task_embedding_dim
        self.hidden_dim = hidden_dim
        self.auto_discover = auto_discover
        
        # Core components
        self.task_graph = TaskRelationshipGraph(
            task_embedding_dim=task_embedding_dim,
            hidden_dim=hidden_dim
        )
        
        self.knowledge_distiller = MetaKnowledgeDistiller(
            embedding_dim=task_embedding_dim
        )
        
        self.zero_shot_synthesizer = ZeroShotTaskSynthesizer(
            task_graph=self.task_graph,
            knowledge_distiller=self.knowledge_distiller
        )
        
        # Data structures
        self.tasks: Dict[str, TaskDescriptor] = {}
        self.task_models: Dict[str, nn.Module] = {}
        self.transfer_edges: List[TransferEdge] = []
        self.curricula: Dict[str, Curriculum] = {}
        self.meta_knowledge: Dict[str, MetaKnowledge] = {}
        
        # Tracking
        self.transfer_history: List[Dict[str, Any]] = []
        self.discovery_log: List[Dict[str, Any]] = []
        
        self.logger = logging.getLogger(__name__)
    
    def register_task(
        self,
        task: TaskDescriptor,
        trained_model: Optional[nn.Module] = None
    ) -> None:
        """Register a new task with the engine."""
        self.tasks[task.task_id] = task
        
        if trained_model is not None:
            self.task_models[task.task_id] = trained_model
        
        self.logger.info(
            f"Registered task '{task.task_name}' (ID: {task.task_id}, "
            f"Type: {task.task_type}, Domain: {task.domain})"
        )
        
        # Auto-discover relationships if enabled
        if self.auto_discover and len(self.tasks) > 1:
            asyncio.create_task(self._discover_relationships(task.task_id))
    
    async def _discover_relationships(self, new_task_id: str) -> None:
        """Automatically discover relationships with existing tasks."""
        new_task = self.tasks[new_task_id]
        
        self.logger.info(f"Auto-discovering relationships for task '{new_task_id}'")
        
        for task_id, task in self.tasks.items():
            if task_id == new_task_id:
                continue
            
            # Compute relationship
            relation = await self._analyze_task_relationship(new_task, task)
            
            if relation.relation_type != TaskRelationType.INDEPENDENT:
                self.transfer_edges.append(relation)
                
                self.logger.info(
                    f"Discovered {relation.relation_type.value} relationship: "
                    f"{new_task_id} ↔ {task_id}"
                )
                
                # Log discovery
                self.discovery_log.append({
                    "timestamp": datetime.utcnow().isoformat(),
                    "source": new_task_id,
                    "target": task_id,
                    "relation": relation.relation_type.value,
                    "transfer_coefficient": relation.transfer_coefficient
                })
    
    async def _analyze_task_relationship(
        self,
        task1: TaskDescriptor,
        task2: TaskDescriptor
    ) -> TransferEdge:
        """Analyze relationship between two tasks."""
        # Compute similarity score
        similarity = self._compute_task_similarity(task1, task2)
        
        # Determine relation type
        if similarity > 0.9:
            relation_type = TaskRelationType.HIGHLY_SIMILAR
        elif similarity > 0.7:
            relation_type = TaskRelationType.MODERATELY_SIMILAR
        elif similarity > 0.4:
            relation_type = TaskRelationType.WEAKLY_SIMILAR
        elif similarity > 0.2:
            relation_type = TaskRelationType.COMPLEMENTARY
        else:
            relation_type = TaskRelationType.INDEPENDENT
        
        # Create transfer edge
        edge = TransferEdge(
            source_task=task1.task_id,
            target_task=task2.task_id,
            relation_type=relation_type,
            transfer_direction=TransferDirection.BIDIRECTIONAL,
            transfer_coefficient=similarity,
            shared_representations=self._identify_shared_skills(task1, task2)
        )
        
        return edge
    
    def _compute_task_similarity(
        self,
        task1: TaskDescriptor,
        task2: TaskDescriptor
    ) -> float:
        """Compute similarity between two tasks."""
        score = 0.0
        
        # Domain similarity
        if task1.domain == task2.domain:
            score += 0.3
        
        # Task type similarity
        if task1.task_type == task2.task_type:
            score += 0.3
        
        # Skill overlap
        skills1 = set(task1.required_skills)
        skills2 = set(task2.required_skills)
        if skills1 and skills2:
            overlap = len(skills1 & skills2)
            total = len(skills1 | skills2)
            score += 0.4 * (overlap / total)
        
        return min(1.0, score)
    
    def _identify_shared_skills(
        self,
        task1: TaskDescriptor,
        task2: TaskDescriptor
    ) -> List[str]:
        """Identify skills shared between tasks."""
        return list(set(task1.required_skills) & set(task2.required_skills))
    
    async def generate_curriculum(
        self,
        target_task: str,
        strategy: CurriculumStrategy = CurriculumStrategy.TRANSFER_POTENTIAL,
        max_tasks: int = 10
    ) -> Curriculum:
        """
        Generate automatic curriculum for learning a target task.
        
        Args:
            target_task: Target task ID
            strategy: Curriculum generation strategy
            max_tasks: Maximum number of tasks in curriculum
        
        Returns:
            Generated curriculum
        """
        self.logger.info(
            f"Generating curriculum for task '{target_task}' "
            f"using strategy '{strategy.value}'"
        )
        
        if target_task not in self.tasks:
            raise ValueError(f"Unknown task: {target_task}")
        
        # Find related tasks
        related_tasks = self._find_curriculum_tasks(target_task, strategy, max_tasks)
        
        # Order tasks
        task_sequence = self._order_curriculum_tasks(
            related_tasks, target_task, strategy
        )
        
        # Estimate difficulties
        task_difficulties = {
            task_id: self._estimate_task_difficulty(task_id)
            for task_id in task_sequence
        }
        
        # Build dependencies
        task_dependencies = self._compute_task_dependencies(task_sequence)
        
        # Create curriculum
        curriculum = Curriculum(
            curriculum_id=f"curriculum_{target_task}_{datetime.utcnow().timestamp()}",
            target_task=target_task,
            strategy=strategy,
            task_sequence=task_sequence,
            task_difficulties=task_difficulties,
            task_dependencies=task_dependencies,
            expected_performance=self._estimate_curriculum_performance(task_sequence, target_task)
        )
        
        self.curricula[curriculum.curriculum_id] = curriculum
        
        self.logger.info(
            f"Generated curriculum with {len(task_sequence)} tasks, "
            f"expected performance: {curriculum.expected_performance:.3f}"
        )
        
        OBSERVABILITY.emit_gauge(
            "transfer.curriculum_length",
            len(task_sequence),
            target_task=target_task,
            strategy=strategy.value
        )
        
        return curriculum
    
    def _find_curriculum_tasks(
        self,
        target_task: str,
        strategy: CurriculumStrategy,
        max_tasks: int
    ) -> List[str]:
        """Find tasks suitable for curriculum."""
        # Find tasks with transfer relationships
        related = []
        
        for edge in self.transfer_edges:
            if edge.target_task == target_task or edge.source_task == target_task:
                other_task = (
                    edge.source_task if edge.target_task == target_task
                    else edge.target_task
                )
                if other_task != target_task:
                    related.append((other_task, edge.transfer_coefficient))
        
        # Sort by transfer coefficient
        related.sort(key=lambda x: x[1], reverse=True)
        
        # Take top-k
        return [task_id for task_id, _ in related[:max_tasks]]
    
    def _order_curriculum_tasks(
        self,
        task_ids: List[str],
        target_task: str,
        strategy: CurriculumStrategy
    ) -> List[str]:
        """Order tasks according to curriculum strategy."""
        if strategy == CurriculumStrategy.EASY_TO_HARD:
            # Sort by difficulty
            difficulties = [(tid, self._estimate_task_difficulty(tid)) for tid in task_ids]
            difficulties.sort(key=lambda x: x[1])
            ordered = [tid for tid, _ in difficulties]
        
        elif strategy == CurriculumStrategy.TRANSFER_POTENTIAL:
            # Already sorted by transfer coefficient
            ordered = task_ids
        
        else:
            # Default: given order
            ordered = task_ids
        
        # Always end with target task
        if target_task not in ordered:
            ordered.append(target_task)
        
        return ordered
    
    def _estimate_task_difficulty(self, task_id: str) -> float:
        """Estimate task difficulty (0=easy, 1=hard)."""
        if task_id not in self.tasks:
            return 0.5
        
        task = self.tasks[task_id]
        
        # Mock: Based on complexity
        difficulty = task.sample_complexity * 0.5 + task.computational_complexity * 0.5
        return min(1.0, difficulty)
    
    def _compute_task_dependencies(
        self,
        task_sequence: List[str]
    ) -> Dict[str, List[str]]:
        """Compute task dependencies (prerequisites)."""
        dependencies = {}
        
        for i, task_id in enumerate(task_sequence):
            # Tasks before this one are potential prerequisites
            deps = []
            for j in range(i):
                prereq_id = task_sequence[j]
                # Check if there's a strong transfer relationship
                for edge in self.transfer_edges:
                    if (edge.source_task == prereq_id and edge.target_task == task_id
                        and edge.transfer_coefficient > 0.5):
                        deps.append(prereq_id)
                        break
            
            dependencies[task_id] = deps
        
        return dependencies
    
    def _estimate_curriculum_performance(
        self,
        task_sequence: List[str],
        target_task: str
    ) -> float:
        """Estimate final performance after curriculum training."""
        # Mock: Based on transfer coefficients
        total_transfer = 0.0
        count = 0
        
        for edge in self.transfer_edges:
            if edge.target_task == target_task and edge.source_task in task_sequence:
                total_transfer += edge.transfer_coefficient
                count += 1
        
        if count == 0:
            return 0.7  # Baseline
        
        avg_transfer = total_transfer / count
        return min(1.0, 0.7 + avg_transfer * 0.3)  # Boost from transfer
    
    async def transfer_knowledge(
        self,
        source_task: str,
        target_task: str,
        transfer_strategy: str = "fine_tuning"
    ) -> Dict[str, Any]:
        """
        Transfer knowledge from source task to target task.
        
        Args:
            source_task: Source task ID
            target_task: Target task ID
            transfer_strategy: Transfer strategy (fine_tuning, feature_extraction, etc.)
        
        Returns:
            Transfer results and metrics
        """
        self.logger.info(
            f"Transferring knowledge: {source_task} → {target_task} "
            f"using {transfer_strategy}"
        )
        
        # Check if tasks exist
        if source_task not in self.tasks or target_task not in self.tasks:
            raise ValueError("Invalid source or target task")
        
        # Check if source model exists
        if source_task not in self.task_models:
            raise ValueError(f"No trained model for source task: {source_task}")
        
        # Get or create transfer edge
        edge = self._get_or_create_transfer_edge(source_task, target_task)
        
        # Perform transfer (mock)
        source_model = self.task_models[source_task]
        
        # Mock: In production, would actually transfer weights, etc.
        transfer_results = {
            "source_task": source_task,
            "target_task": target_task,
            "strategy": transfer_strategy,
            "performance_gain": edge.transfer_coefficient * 0.2,  # Mock gain
            "sample_efficiency_gain": edge.transfer_coefficient * 0.3,
            "convergence_speed_gain": edge.transfer_coefficient * 0.25,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        # Update edge statistics
        edge.transfer_attempts += 1
        edge.successful_transfers += 1
        edge.performance_gain = transfer_results["performance_gain"]
        edge.last_transfer = transfer_results["timestamp"]
        
        # Log transfer
        self.transfer_history.append(transfer_results)
        
        self.logger.info(
            f"Transfer complete. Performance gain: "
            f"{transfer_results['performance_gain']:.3f}"
        )
        
        OBSERVABILITY.emit_counter(
            "transfer.knowledge_transfers",
            source=source_task,
            target=target_task,
            strategy=transfer_strategy
        )
        
        OBSERVABILITY.emit_gauge(
            "transfer.performance_gain",
            transfer_results["performance_gain"],
            source=source_task,
            target=target_task
        )
        
        return transfer_results
    
    def _get_or_create_transfer_edge(
        self,
        source_task: str,
        target_task: str
    ) -> TransferEdge:
        """Get existing transfer edge or create new one."""
        # Look for existing edge
        for edge in self.transfer_edges:
            if edge.source_task == source_task and edge.target_task == target_task:
                return edge
        
        # Create new edge
        source_desc = self.tasks[source_task]
        target_desc = self.tasks[target_task]
        
        edge = TransferEdge(
            source_task=source_task,
            target_task=target_task,
            relation_type=TaskRelationType.MODERATELY_SIMILAR,  # Default
            transfer_direction=TransferDirection.UNIDIRECTIONAL,
            transfer_coefficient=self._compute_task_similarity(source_desc, target_desc)
        )
        
        self.transfer_edges.append(edge)
        return edge
    
    async def synthesize_zero_shot_model(
        self,
        new_task: TaskDescriptor,
        synthesis_strategy: str = "weighted_ensemble"
    ) -> nn.Module:
        """
        Synthesize model for new task without training (zero-shot).
        
        Args:
            new_task: Descriptor for new task
            synthesis_strategy: Synthesis strategy
        
        Returns:
            Synthesized model
        """
        model = await self.zero_shot_synthesizer.synthesize_for_new_task(
            new_task=new_task,
            available_tasks=self.tasks,
            task_models=self.task_models,
            strategy=synthesis_strategy
        )
        
        OBSERVABILITY.emit_counter(
            "transfer.zero_shot_synthesis",
            task_type=new_task.task_type,
            domain=new_task.domain,
            strategy=synthesis_strategy
        )
        
        return model
    
    def get_transfer_graph_metrics(self) -> Dict[str, Any]:
        """Get metrics about the transfer knowledge graph."""
        return {
            "num_tasks": len(self.tasks),
            "num_trained_models": len(self.task_models),
            "num_transfer_edges": len(self.transfer_edges),
            "num_curricula": len(self.curricula),
            "num_meta_knowledge": len(self.meta_knowledge),
            "total_transfers": len(self.transfer_history),
            "avg_transfer_coefficient": (
                sum(e.transfer_coefficient for e in self.transfer_edges) / len(self.transfer_edges)
                if self.transfer_edges else 0.0
            ),
            "discovery_events": len(self.discovery_log)
        }
    
    def export_transfer_graph(self, output_path: Path) -> None:
        """Export transfer graph to file."""
        graph_data = {
            "tasks": {
                task_id: asdict(task)
                for task_id, task in self.tasks.items()
            },
            "transfer_edges": [asdict(edge) for edge in self.transfer_edges],
            "curricula": {
                curriculum_id: asdict(curriculum)
                for curriculum_id, curriculum in self.curricula.items()
            },
            "meta_knowledge": {
                knowledge_id: asdict(knowledge)
                for knowledge_id, knowledge in self.meta_knowledge.items()
            },
            "metrics": self.get_transfer_graph_metrics(),
            "transfer_history": self.transfer_history[-100:],  # Last 100
            "discovery_log": self.discovery_log[-100:],
            "exported_at": datetime.utcnow().isoformat()
        }
        
        serializable_data = self._make_json_safe(graph_data)

        with open(output_path, 'w') as f:
            json.dump(serializable_data, f, indent=2)
        
        self.logger.info(f"Exported transfer graph to {output_path}")


def create_cross_task_transfer_engine(
    task_embedding_dim: int = 128,
    hidden_dim: int = 256,
    auto_discover: bool = True
) -> CrossTaskTransferEngine:
    """
    Factory function to create a configured Cross-Task Transfer Engine.
    
    Args:
        task_embedding_dim: Dimension of task embeddings
        hidden_dim: Hidden dimension for GNN
        auto_discover: Enable automatic relationship discovery
    
    Returns:
        Configured CrossTaskTransferEngine
    """
    engine = CrossTaskTransferEngine(
        task_embedding_dim=task_embedding_dim,
        hidden_dim=hidden_dim,
        auto_discover=auto_discover
    )
    
    logging.info("Created Cross-Task Transfer Learning Engine")
    
    return engine
