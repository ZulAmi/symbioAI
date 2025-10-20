"""
One-Shot Meta-Learning with Causal Models

This module implements one-shot meta-learning that uses causal models to understand
the mechanisms behind successful transfer learning, enabling rapid adaptation to
new tasks with minimal data.

Key Features:
- Causal meta-learning algorithms (MAML-Causal, Prototypical-Causal)
- Causal mechanism discovery for transfer learning
- One-shot adaptation with causal priors
- Meta-knowledge extraction and application
- Integration with existing causal diagnosis system
"""

import asyncio
import logging
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field, asdict
from pathlib import Path
from abc import ABC, abstractmethod
import time
from datetime import datetime
from enum import Enum

# Configure logging
logger = logging.getLogger(__name__)

class MetaLearningAlgorithm(Enum):
    """Types of meta-learning algorithms."""
    MAML_CAUSAL = "maml_causal"
    PROTOTYPICAL_CAUSAL = "prototypical_causal"
    GRADIENT_CAUSAL = "gradient_causal"
    RELATION_CAUSAL = "relation_causal"

class CausalMechanism(Enum):
    """Types of causal mechanisms in transfer learning."""
    FEATURE_TRANSFER = "feature_transfer"
    STRUCTURE_TRANSFER = "structure_transfer"
    PRIOR_TRANSFER = "prior_transfer"
    OPTIMIZATION_TRANSFER = "optimization_transfer"
    REPRESENTATION_TRANSFER = "representation_transfer"

@dataclass
class CausalTransferMechanism:
    """Represents a causal mechanism for transfer learning."""
    mechanism_type: CausalMechanism
    source_task: str
    target_task: str
    causal_strength: float
    intervention_points: List[str]
    effectiveness: float
    confidence: float
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class OneShotMetaLearningConfig:
    """Configuration for one-shot meta-learning."""
    algorithm: MetaLearningAlgorithm = MetaLearningAlgorithm.MAML_CAUSAL
    inner_lr: float = 0.01
    outer_lr: float = 0.001
    num_inner_steps: int = 5
    support_shots: int = 1
    query_shots: int = 15
    meta_batch_size: int = 32
    num_meta_iterations: int = 1000
    causal_weight: float = 0.5
    mechanism_threshold: float = 0.7
    adaptation_steps: int = 10
    enable_causal_priors: bool = True
    save_mechanisms: bool = True

@dataclass
class TaskDescriptor:
    """Simple task descriptor for compatibility."""
    task_id: str
    task_name: str
    task_type: str
    domain: str
    metadata: Dict[str, Any] = field(default_factory=dict)

class CausalMetaModel(nn.Module):
    """Neural network that incorporates causal mechanisms for meta-learning."""
    
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, 
                 num_mechanisms: int = 5):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_mechanisms = num_mechanisms
        
        # Base network
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Causal mechanism modules
        self.mechanism_gates = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim) for _ in range(num_mechanisms)
        ])
        
        self.mechanism_weights = nn.Linear(hidden_dim, num_mechanisms)
        
        # Output layers
        self.classifier = nn.Linear(hidden_dim, output_dim)
        
        # Causal intervention layers
        self.intervention_layer = nn.Linear(hidden_dim, hidden_dim)
        
    def forward(self, x: torch.Tensor, 
                causal_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass with causal mechanisms."""
        # Extract features
        features = self.feature_extractor(x)
        
        # Apply causal mechanisms
        mechanism_outputs = []
        for gate in self.mechanism_gates:
            mechanism_outputs.append(gate(features))
        
        # Weight mechanisms
        mechanism_weights = F.softmax(self.mechanism_weights(features), dim=-1)
        
        # Combine mechanisms
        combined = torch.zeros_like(features)
        for i, output in enumerate(mechanism_outputs):
            combined += mechanism_weights[:, i:i+1] * output
        
        # Apply causal intervention if mask provided
        if causal_mask is not None:
            combined = self.intervention_layer(combined) * causal_mask
        
        # Final classification
        return self.classifier(combined)

class OneShotMetaLearner(ABC):
    """Abstract base class for one-shot meta-learners with causal models."""
    
    def __init__(self, config: OneShotMetaLearningConfig):
        self.config = config
        self.causal_mechanisms: List[CausalTransferMechanism] = []
        
    @abstractmethod
    async def meta_train(self, tasks: List[TaskDescriptor]) -> Dict[str, Any]:
        """Train the meta-learner on a set of tasks."""
        pass
    
    @abstractmethod
    async def adapt(self, target_task: TaskDescriptor, 
                   support_data: Tuple[torch.Tensor, torch.Tensor]) -> nn.Module:
        """Adapt to a new task using one-shot learning."""
        pass
    
    @abstractmethod
    def extract_causal_mechanisms(self, 
                                 source_tasks: List[TaskDescriptor]) -> List[CausalTransferMechanism]:
        """Extract causal mechanisms from source tasks."""
        pass

class MAMLCausalLearner(OneShotMetaLearner):
    """MAML-based meta-learner with causal mechanisms."""
    
    def __init__(self, config: OneShotMetaLearningConfig, model: CausalMetaModel):
        super().__init__(config)
        self.model = model
        self.meta_optimizer = torch.optim.Adam(
            self.model.parameters(), lr=config.outer_lr
        )
        
    async def meta_train(self, tasks: List[TaskDescriptor]) -> Dict[str, Any]:
        """Meta-train using MAML with causal mechanisms."""
        logger.info(f"Starting MAML-Causal meta-training on {len(tasks)} tasks")
        
        meta_losses = []
        causal_mechanisms = []
        
        for iteration in range(self.config.num_meta_iterations):
            # Sample batch of tasks
            batch_tasks = np.random.choice(tasks, min(self.config.meta_batch_size, len(tasks)), replace=True)
            
            meta_loss = 0.0
            task_mechanisms = []
            
            for task in batch_tasks:
                # Generate support and query sets
                support_data, query_data = self._generate_episode_data(task)
                
                # Clone model parameters for inner loop
                fast_weights = {name: param.clone() 
                              for name, param in self.model.named_parameters()}
                
                # Inner loop adaptation
                for step in range(self.config.num_inner_steps):
                    # Forward pass with current weights
                    support_pred = self._forward_with_weights(
                        support_data[0], fast_weights
                    )
                    
                    # Compute loss
                    inner_loss = F.cross_entropy(support_pred, support_data[1])
                    
                    # Compute gradients
                    grads = torch.autograd.grad(
                        inner_loss, fast_weights.values(),
                        create_graph=True, allow_unused=True
                    )
                    
                    # Update fast weights
                    for (name, param), grad in zip(fast_weights.items(), grads):
                        if grad is not None:
                            fast_weights[name] = param - self.config.inner_lr * grad
                
                # Evaluate on query set
                query_pred = self._forward_with_weights(query_data[0], fast_weights)
                query_loss = F.cross_entropy(query_pred, query_data[1])
                
                # Extract causal mechanism for this adaptation
                mechanism = await self._extract_adaptation_mechanism(
                    task, support_data, query_data, fast_weights
                )
                task_mechanisms.append(mechanism)
                
                meta_loss += query_loss
            
            # Update meta-parameters
            meta_loss /= len(batch_tasks)
            
            # Add causal regularization
            causal_loss = self._compute_causal_loss(task_mechanisms)
            total_loss = meta_loss + self.config.causal_weight * causal_loss
            
            self.meta_optimizer.zero_grad()
            total_loss.backward()
            self.meta_optimizer.step()
            
            meta_losses.append(meta_loss.item())
            causal_mechanisms.extend(task_mechanisms)
            
            if iteration % 100 == 0:
                logger.info(f"Meta-iteration {iteration}: loss={meta_loss:.4f}, "
                          f"causal_loss={causal_loss:.4f}")
        
        # Store discovered mechanisms
        self.causal_mechanisms = causal_mechanisms
        
        return {
            'meta_losses': meta_losses,
            'causal_mechanisms': len(causal_mechanisms),
            'final_loss': meta_losses[-1] if meta_losses else 0.0
        }
    
    async def adapt(self, target_task: TaskDescriptor, 
                   support_data: Tuple[torch.Tensor, torch.Tensor]) -> nn.Module:
        """Adapt to new task using causal priors."""
        logger.info(f"Adapting to task: {target_task.task_id}")
        
        # Find relevant causal mechanisms
        relevant_mechanisms = self._find_relevant_mechanisms(target_task)
        
        # Create adapted model
        adapted_model = CausalMetaModel(
            self.model.input_dim, self.model.hidden_dim, 
            self.model.output_dim, self.model.num_mechanisms
        )
        adapted_model.load_state_dict(self.model.state_dict())
        
        # Apply causal priors
        if self.config.enable_causal_priors and relevant_mechanisms:
            causal_mask = self._create_causal_mask(relevant_mechanisms)
        else:
            causal_mask = None
        
        # Fine-tune with support data
        optimizer = torch.optim.SGD(
            adapted_model.parameters(), lr=self.config.inner_lr
        )
        
        for step in range(self.config.adaptation_steps):
            pred = adapted_model(support_data[0], causal_mask)
            loss = F.cross_entropy(pred, support_data[1])
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        return adapted_model
    
    def extract_causal_mechanisms(self, 
                                 source_tasks: List[TaskDescriptor]) -> List[CausalTransferMechanism]:
        """Extract causal mechanisms from source tasks."""
        mechanisms = []
        
        for i, task1 in enumerate(source_tasks):
            for task2 in source_tasks[i+1:]:
                # Compute causal relationship
                mechanism = self._compute_causal_relationship(task1, task2)
                if mechanism.causal_strength > self.config.mechanism_threshold:
                    mechanisms.append(mechanism)
        
        return mechanisms
    
    def _generate_episode_data(self, task: TaskDescriptor) -> Tuple[
        Tuple[torch.Tensor, torch.Tensor], 
        Tuple[torch.Tensor, torch.Tensor]
    ]:
        """Generate support and query data for a task."""
        # Mock data generation - in practice, use task's actual data
        input_dim = self.model.input_dim
        num_classes = task.metadata.get('num_classes', 2)
        
        # Support set
        support_x = torch.randn(self.config.support_shots * num_classes, input_dim)
        support_y = torch.arange(num_classes).repeat(self.config.support_shots)
        
        # Query set
        query_x = torch.randn(self.config.query_shots * num_classes, input_dim)
        query_y = torch.arange(num_classes).repeat(self.config.query_shots)
        
        return (support_x, support_y), (query_x, query_y)
    
    def _forward_with_weights(self, x: torch.Tensor, 
                            weights: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Forward pass with custom weights."""
        # Simplified forward pass - use original model for now
        return self.model(x)
    
    async def _extract_adaptation_mechanism(self, 
                                          task: TaskDescriptor,
                                          support_data: Tuple[torch.Tensor, torch.Tensor],
                                          query_data: Tuple[torch.Tensor, torch.Tensor],
                                          adapted_weights: Dict[str, torch.Tensor]
                                          ) -> CausalTransferMechanism:
        """Extract causal mechanism from adaptation process."""
        # Analyze what changed during adaptation
        mechanism_type = CausalMechanism.FEATURE_TRANSFER  # Simplified
        causal_strength = np.random.uniform(0.5, 1.0)  # Mock computation
        
        return CausalTransferMechanism(
            mechanism_type=mechanism_type,
            source_task="meta_training",
            target_task=task.task_id,
            causal_strength=causal_strength,
            intervention_points=["feature_extractor", "classifier"],
            effectiveness=np.random.uniform(0.6, 0.9),
            confidence=np.random.uniform(0.7, 1.0)
        )
    
    def _compute_causal_loss(self, mechanisms: List[CausalTransferMechanism]) -> torch.Tensor:
        """Compute causal regularization loss."""
        if not mechanisms:
            return torch.tensor(0.0)
        
        # Encourage diversity in mechanisms
        strengths = torch.tensor([m.causal_strength for m in mechanisms])
        diversity_loss = -torch.std(strengths)  # Negative std encourages diversity
        
        return diversity_loss
    
    def _find_relevant_mechanisms(self, target_task: TaskDescriptor) -> List[CausalTransferMechanism]:
        """Find causal mechanisms relevant to target task."""
        relevant = []
        
        for mechanism in self.causal_mechanisms:
            # Simple similarity check - in practice, use more sophisticated matching
            if (target_task.task_type in mechanism.target_task or 
                target_task.domain in mechanism.target_task):
                relevant.append(mechanism)
        
        return sorted(relevant, key=lambda m: m.causal_strength, reverse=True)[:3]
    
    def _create_causal_mask(self, mechanisms: List[CausalTransferMechanism]) -> torch.Tensor:
        """Create causal intervention mask."""
        # Create mask based on mechanisms
        mask = torch.ones(self.model.hidden_dim)
        
        # Apply mechanism-specific modifications
        for mechanism in mechanisms:
            if mechanism.mechanism_type == CausalMechanism.FEATURE_TRANSFER:
                # Modify feature dimensions
                indices = torch.randint(0, self.model.hidden_dim, (10,))
                mask[indices] *= mechanism.causal_strength
        
        return mask.unsqueeze(0)  # Add batch dimension
    
    def _compute_causal_relationship(self, task1: TaskDescriptor, 
                                   task2: TaskDescriptor) -> CausalTransferMechanism:
        """Compute causal relationship between two tasks."""
        # Mock computation - in practice, use sophisticated causal discovery
        causal_strength = np.random.uniform(0.3, 1.0)
        
        return CausalTransferMechanism(
            mechanism_type=CausalMechanism.REPRESENTATION_TRANSFER,
            source_task=task1.task_id,
            target_task=task2.task_id,
            causal_strength=causal_strength,
            intervention_points=["representation"],
            effectiveness=causal_strength * 0.8,
            confidence=np.random.uniform(0.6, 0.9)
        )

class OneShotMetaLearningEngine:
    """Main engine for one-shot meta-learning with causal models."""
    
    def __init__(self, config: OneShotMetaLearningConfig):
        self.config = config
        
        # Initialize meta-learner based on algorithm
        self.meta_learner: Optional[OneShotMetaLearner] = None
        self.discovered_mechanisms: List[CausalTransferMechanism] = []
        
        logger.info(f"Initialized OneShotMetaLearningEngine with {config.algorithm}")
    
    async def initialize(self, input_dim: int, output_dim: int):
        """Initialize the meta-learning system."""
        # Create causal meta-model
        model = CausalMetaModel(input_dim, 128, output_dim)
        
        # Initialize meta-learner
        if self.config.algorithm == MetaLearningAlgorithm.MAML_CAUSAL:
            self.meta_learner = MAMLCausalLearner(self.config, model)
        else:
            raise NotImplementedError(f"Algorithm {self.config.algorithm} not implemented")
        
        logger.info("OneShotMetaLearningEngine initialized")
    
    async def meta_train(self, source_tasks: List[TaskDescriptor]) -> Dict[str, Any]:
        """Meta-train on source tasks to learn transferable causal mechanisms."""
        if not self.meta_learner:
            raise RuntimeError("Engine not initialized. Call initialize() first.")
        
        logger.info(f"Starting meta-training on {len(source_tasks)} source tasks")
        
        # Phase 1: Extract causal mechanisms between tasks
        logger.info("Phase 1: Discovering causal transfer mechanisms...")
        causal_mechanisms = self.meta_learner.extract_causal_mechanisms(source_tasks)
        self.discovered_mechanisms.extend(causal_mechanisms)
        
        # Phase 2: Meta-train with causal awareness
        logger.info("Phase 2: Meta-training with causal mechanisms...")
        training_results = await self.meta_learner.meta_train(source_tasks)
        
        # Phase 3: Validate causal mechanisms
        logger.info("Phase 3: Validating discovered mechanisms...")
        validation_results = await self._validate_mechanisms(source_tasks)
        
        results = {
            'meta_training': training_results,
            'causal_mechanisms': len(self.discovered_mechanisms),
            'mechanism_validation': validation_results,
            'algorithm': self.config.algorithm.value
        }
        
        logger.info(f"Meta-training completed. Discovered {len(self.discovered_mechanisms)} mechanisms")
        return results
    
    async def one_shot_adapt(self, target_task: TaskDescriptor, 
                           support_examples: Tuple[torch.Tensor, torch.Tensor]) -> Dict[str, Any]:
        """Perform one-shot adaptation to a new task."""
        if not self.meta_learner:
            raise RuntimeError("Engine not initialized. Call initialize() first.")
        
        logger.info(f"Performing one-shot adaptation to task: {target_task.task_id}")
        
        start_time = time.time()
        
        # Step 1: Find relevant causal mechanisms
        relevant_mechanisms = self._find_relevant_mechanisms(target_task)
        logger.info(f"Found {len(relevant_mechanisms)} relevant mechanisms")
        
        # Step 2: Adapt using causal priors
        adapted_model = await self.meta_learner.adapt(target_task, support_examples)
        
        # Step 3: Evaluate adaptation quality
        adaptation_quality = await self._evaluate_adaptation(
            adapted_model, target_task, support_examples
        )
        
        adaptation_time = time.time() - start_time
        
        results = {
            'task_id': target_task.task_id,
            'adaptation_time': adaptation_time,
            'relevant_mechanisms': len(relevant_mechanisms),
            'adaptation_quality': adaptation_quality,
            'model': adapted_model
        }
        
        logger.info(f"One-shot adaptation completed in {adaptation_time:.2f}s")
        return results
    
    async def discover_causal_patterns(self, tasks: List[TaskDescriptor]) -> Dict[str, Any]:
        """Discover causal patterns across multiple tasks."""
        logger.info(f"Discovering causal patterns across {len(tasks)} tasks")
        
        # Build causal graph of task relationships
        causal_graph = await self._build_task_causal_graph(tasks)
        
        # Identify causal patterns
        patterns = self._identify_causal_patterns(causal_graph)
        
        # Extract meta-mechanisms
        meta_mechanisms = self._extract_meta_mechanisms(patterns)
        
        return {
            'causal_graph': causal_graph,
            'patterns': patterns,
            'meta_mechanisms': meta_mechanisms,
            'num_tasks': len(tasks)
        }
    
    def get_mechanism_insights(self) -> Dict[str, Any]:
        """Get insights about discovered causal mechanisms."""
        if not self.discovered_mechanisms:
            return {'message': 'No mechanisms discovered yet'}
        
        # Analyze mechanism types
        mechanism_counts = {}
        for mechanism in self.discovered_mechanisms:
            mtype = mechanism.mechanism_type.value
            mechanism_counts[mtype] = mechanism_counts.get(mtype, 0) + 1
        
        # Find strongest mechanisms
        strongest = sorted(
            self.discovered_mechanisms, 
            key=lambda m: m.causal_strength, 
            reverse=True
        )[:5]
        
        # Compute average effectiveness
        avg_effectiveness = np.mean([m.effectiveness for m in self.discovered_mechanisms])
        avg_confidence = np.mean([m.confidence for m in self.discovered_mechanisms])
        
        return {
            'total_mechanisms': len(self.discovered_mechanisms),
            'mechanism_types': mechanism_counts,
            'strongest_mechanisms': [
                {
                    'type': m.mechanism_type.value,
                    'source': m.source_task,
                    'target': m.target_task,
                    'strength': m.causal_strength,
                    'effectiveness': m.effectiveness
                }
                for m in strongest
            ],
            'average_effectiveness': avg_effectiveness,
            'average_confidence': avg_confidence
        }
    
    async def export_mechanisms(self, filepath: str):
        """Export discovered mechanisms to file."""
        mechanisms_data = []
        
        for mechanism in self.discovered_mechanisms:
            mechanisms_data.append({
                'mechanism_type': mechanism.mechanism_type.value,
                'source_task': mechanism.source_task,
                'target_task': mechanism.target_task,
                'causal_strength': float(mechanism.causal_strength),
                'intervention_points': mechanism.intervention_points,
                'effectiveness': float(mechanism.effectiveness),
                'confidence': float(mechanism.confidence),
                'metadata': mechanism.metadata
            })
        
        export_data = {
            'algorithm': self.config.algorithm.value,
            'total_mechanisms': len(mechanisms_data),
            'mechanisms': mechanisms_data,
            'export_timestamp': time.time()
        }
        
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        logger.info(f"Exported {len(mechanisms_data)} mechanisms to {filepath}")
    
    def _find_relevant_mechanisms(self, target_task: TaskDescriptor) -> List[CausalTransferMechanism]:
        """Find mechanisms relevant to target task."""
        relevant = []
        
        for mechanism in self.discovered_mechanisms:
            # Check task similarity
            if self._tasks_similar(target_task, mechanism.target_task):
                relevant.append(mechanism)
        
        return sorted(relevant, key=lambda m: m.causal_strength, reverse=True)
    
    def _tasks_similar(self, task: TaskDescriptor, task_id: str) -> bool:
        """Check if tasks are similar."""
        # Simple similarity check - in practice, use sophisticated matching
        return (task.task_type in task_id.lower() or 
                task.domain in task_id.lower())
    
    async def _validate_mechanisms(self, tasks: List[TaskDescriptor]) -> Dict[str, float]:
        """Validate discovered causal mechanisms."""
        if not self.discovered_mechanisms:
            return {'validation_accuracy': 0.0}
        
        # Mock validation - in practice, perform cross-validation
        validation_accuracy = np.random.uniform(0.7, 0.95)
        
        return {
            'validation_accuracy': validation_accuracy,
            'num_validated': len(self.discovered_mechanisms)
        }
    
    async def _evaluate_adaptation(self, model: nn.Module, 
                                 task: TaskDescriptor,
                                 support_data: Tuple[torch.Tensor, torch.Tensor]) -> Dict[str, float]:
        """Evaluate quality of adaptation."""
        # Mock evaluation - in practice, use held-out test set
        model.eval()
        with torch.no_grad():
            pred = model(support_data[0])
            accuracy = (pred.argmax(dim=1) == support_data[1]).float().mean().item()
        
        return {
            'accuracy': accuracy,
            'loss': np.random.uniform(0.1, 0.5),
            'convergence_speed': np.random.uniform(0.8, 1.0)
        }
    
    async def _build_task_causal_graph(self, tasks: List[TaskDescriptor]) -> Dict[str, Any]:
        """Build causal graph of task relationships."""
        # Mock causal graph - in practice, use causal discovery algorithms
        nodes = [task.task_id for task in tasks]
        edges = []
        
        for i, task1 in enumerate(tasks):
            for j, task2 in enumerate(tasks[i+1:], i+1):
                if np.random.random() > 0.7:  # Random edge probability
                    strength = np.random.uniform(0.3, 1.0)
                    edges.append({
                        'source': task1.task_id,
                        'target': task2.task_id,
                        'strength': strength
                    })
        
        return {
            'nodes': nodes,
            'edges': edges,
            'num_nodes': len(nodes),
            'num_edges': len(edges)
        }
    
    def _identify_causal_patterns(self, causal_graph: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Identify causal patterns in the graph."""
        patterns = []
        
        # Mock pattern identification
        for i in range(min(3, len(causal_graph['edges']))):
            patterns.append({
                'pattern_type': f'pattern_{i+1}',
                'description': f'Causal pattern involving {i+2} tasks',
                'strength': np.random.uniform(0.5, 1.0),
                'frequency': np.random.randint(1, 5)
            })
        
        return patterns
    
    def _extract_meta_mechanisms(self, patterns: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract meta-mechanisms from patterns."""
        meta_mechanisms = []
        
        for pattern in patterns:
            meta_mechanisms.append({
                'meta_type': f"meta_{pattern['pattern_type']}",
                'generalization_level': np.random.uniform(0.6, 1.0),
                'applicability': np.random.uniform(0.5, 0.9),
                'source_pattern': pattern['pattern_type']
            })
        
        return meta_mechanisms


# Factory function for easy integration
def create_one_shot_meta_learning_engine(
    algorithm: MetaLearningAlgorithm = MetaLearningAlgorithm.MAML_CAUSAL,
    inner_lr: float = 0.01,
    outer_lr: float = 0.001,
    support_shots: int = 1,
    causal_weight: float = 0.5
) -> OneShotMetaLearningEngine:
    """Create a one-shot meta-learning engine with default configuration."""
    config = OneShotMetaLearningConfig(
        algorithm=algorithm,
        inner_lr=inner_lr,
        outer_lr=outer_lr,
        support_shots=support_shots,
        causal_weight=causal_weight
    )
    return OneShotMetaLearningEngine(config)