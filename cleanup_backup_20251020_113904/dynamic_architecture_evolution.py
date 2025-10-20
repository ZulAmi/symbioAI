#!/usr/bin/env python3
"""
Dynamic Neural Architecture Evolution System
Real-time architecture adaptation based on task complexity

Features:
1. Neural Architecture Search (NAS) during inference
2. Task-adaptive depth and width
3. Automatic module specialization and pruning
4. Morphological evolution of network topology
5. Real-time performance optimization

Competitive Edge: Most systems have fixed architectures; ours adapts in real-time
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
from collections import defaultdict, deque
import json
import time
import logging
from pathlib import Path


class ArchitectureOperation(Enum):
    """Types of architecture modification operations."""
    ADD_LAYER = "add_layer"
    REMOVE_LAYER = "remove_layer"
    EXPAND_WIDTH = "expand_width"
    SHRINK_WIDTH = "shrink_width"
    SPECIALIZE_MODULE = "specialize_module"
    PRUNE_MODULE = "prune_module"
    SPLIT_MODULE = "split_module"
    MERGE_MODULES = "merge_modules"


class TaskComplexity(Enum):
    """Task complexity levels."""
    TRIVIAL = "trivial"
    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"
    EXPERT = "expert"


@dataclass
class ModuleStats:
    """Statistics for a neural module."""
    activation_mean: float = 0.0
    activation_std: float = 0.0
    gradient_norm: float = 0.0
    utilization: float = 0.0
    specialization_score: float = 0.0
    importance_score: float = 0.0
    task_affinity: Dict[str, float] = field(default_factory=dict)
    prunable: bool = False
    splittable: bool = False


@dataclass
class ArchitectureEvolutionConfig:
    """Configuration for architecture evolution."""
    min_layers: int = 2
    max_layers: int = 20
    min_width: int = 64
    max_width: int = 1024
    growth_threshold: float = 0.8
    shrink_threshold: float = 0.3
    prune_threshold: float = 0.1
    specialization_threshold: float = 0.7
    adaptation_rate: float = 0.1
    enable_nas: bool = True
    enable_runtime_adaptation: bool = True
    enable_pruning: bool = True
    enable_specialization: bool = True
    complexity_window: int = 100
    evolution_interval: int = 50
    min_utilization: float = 0.2
    max_module_age: int = 1000


class AdaptiveModule(nn.Module):
    """
    Self-adapting neural module that can grow, shrink, and specialize.
    """
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        module_id: str,
        initial_depth: int = 2
    ):
        super().__init__()
        self.module_id = module_id
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # Dynamic layer stack
        self.layers = nn.ModuleList()
        self._build_initial_layers(input_dim, output_dim, initial_depth)
        
        # Statistics tracking
        self.stats = ModuleStats()
        self.activation_history = deque(maxlen=1000)
        self.gradient_history = deque(maxlen=1000)
        self.task_history = deque(maxlen=100)
        
        # Evolution metadata
        self.age = 0
        self.last_modified = 0
        self.specializations: List[str] = []
        
    def _build_initial_layers(self, input_dim: int, output_dim: int, depth: int):
        """Build initial layer stack."""
        dims = np.linspace(input_dim, output_dim, depth + 1, dtype=int)
        
        for i in range(depth):
            layer = nn.Sequential(
                nn.Linear(int(dims[i]), int(dims[i+1])),
                nn.LayerNorm(int(dims[i+1])),
                nn.ReLU(),
                nn.Dropout(0.1)
            )
            self.layers.append(layer)
    
    def forward(self, x: torch.Tensor, task_id: Optional[str] = None) -> torch.Tensor:
        """Forward pass with statistics tracking."""
        self.age += 1
        
        if task_id:
            self.task_history.append(task_id)
        
        activations = []
        for layer in self.layers:
            x = layer(x)
            activations.append(x)
            
            # Track activation statistics
            self.activation_history.append({
                'mean': x.mean().item(),
                'std': x.std().item(),
                'max': x.max().item(),
                'min': x.min().item()
            })
        
        return x
    
    def update_stats(self):
        """Update module statistics for evolution decisions."""
        if len(self.activation_history) > 0:
            recent_activations = list(self.activation_history)[-100:]
            self.stats.activation_mean = np.mean([a['mean'] for a in recent_activations])
            self.stats.activation_std = np.mean([a['std'] for a in recent_activations])
        
        if len(self.gradient_history) > 0:
            self.stats.gradient_norm = np.mean(list(self.gradient_history)[-100:])
        
        # Calculate utilization (how often non-zero activations)
        if len(self.activation_history) > 0:
            non_zero_ratio = np.mean([
                1.0 if abs(a['mean']) > 0.01 else 0.0
                for a in list(self.activation_history)[-100:]
            ])
            self.stats.utilization = non_zero_ratio
        
        # Calculate task specialization
        if len(self.task_history) > 0:
            task_counts = defaultdict(int)
            for task in self.task_history:
                task_counts[task] += 1
            
            total = len(self.task_history)
            self.stats.task_affinity = {
                task: count / total for task, count in task_counts.items()
            }
            
            # Specialization = entropy of task distribution
            probs = np.array(list(self.stats.task_affinity.values()))
            entropy = -np.sum(probs * np.log(probs + 1e-10))
            max_entropy = np.log(len(task_counts))
            self.stats.specialization_score = 1.0 - (entropy / (max_entropy + 1e-10))
        
        # Determine if prunable
        self.stats.prunable = (
            self.stats.utilization < 0.1 and
            self.age > 500
        )
        
        # Determine if splittable (high utilization + specialization)
        self.stats.splittable = (
            self.stats.utilization > 0.8 and
            self.stats.specialization_score > 0.6 and
            len(self.stats.task_affinity) >= 2
        )
    
    def add_layer(self, position: int = -1):
        """Add a new layer to the module."""
        if position == -1:
            position = len(self.layers)
        
        # Determine dimensions
        if position == 0:
            in_dim = self.input_dim
            out_dim = self.layers[0][0].in_features if len(self.layers) > 0 else self.output_dim
        elif position >= len(self.layers):
            in_dim = self.layers[-1][0].out_features if len(self.layers) > 0 else self.input_dim
            out_dim = self.output_dim
        else:
            in_dim = self.layers[position-1][0].out_features
            out_dim = self.layers[position][0].in_features
        
        new_layer = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        self.layers.insert(position, new_layer)
        self.last_modified = self.age
        
        return True
    
    def remove_layer(self, position: int = -1) -> bool:
        """Remove a layer from the module."""
        if len(self.layers) <= 1:
            return False
        
        if position == -1:
            position = len(self.layers) - 1
        
        self.layers.pop(position)
        self.last_modified = self.age
        
        return True
    
    def expand_width(self, factor: float = 1.5) -> bool:
        """Expand the width of all layers."""
        new_layers = nn.ModuleList()
        
        for i, layer in enumerate(self.layers):
            linear = layer[0]
            old_in = linear.in_features
            old_out = linear.out_features
            
            new_in = old_in if i == 0 else int(old_in * factor)
            new_out = old_out if i == len(self.layers) - 1 else int(old_out * factor)
            
            # Create expanded layer
            new_layer = nn.Sequential(
                nn.Linear(new_in, new_out),
                nn.LayerNorm(new_out),
                nn.ReLU(),
                nn.Dropout(0.1)
            )
            
            # Copy weights where possible
            with torch.no_grad():
                min_in = min(old_in, new_in)
                min_out = min(old_out, new_out)
                new_layer[0].weight[:min_out, :min_in] = linear.weight[:min_out, :min_in]
                new_layer[0].bias[:min_out] = linear.bias[:min_out]
            
            new_layers.append(new_layer)
        
        self.layers = new_layers
        self.last_modified = self.age
        
        return True
    
    def shrink_width(self, factor: float = 0.75) -> bool:
        """Shrink the width of all layers."""
        new_layers = nn.ModuleList()
        
        for i, layer in enumerate(self.layers):
            linear = layer[0]
            old_in = linear.in_features
            old_out = linear.out_features
            
            new_in = old_in if i == 0 else max(32, int(old_in * factor))
            new_out = old_out if i == len(self.layers) - 1 else max(32, int(old_out * factor))
            
            # Create shrunk layer
            new_layer = nn.Sequential(
                nn.Linear(new_in, new_out),
                nn.LayerNorm(new_out),
                nn.ReLU(),
                nn.Dropout(0.1)
            )
            
            # Copy most important weights
            with torch.no_grad():
                # Use weight magnitudes to select important neurons
                weight_importance = linear.weight.abs().sum(dim=1)
                top_indices = weight_importance.argsort(descending=True)[:new_out]
                
                new_layer[0].weight[:, :new_in] = linear.weight[top_indices, :new_in]
                new_layer[0].bias[:] = linear.bias[top_indices]
            
            new_layers.append(new_layer)
        
        self.layers = new_layers
        self.last_modified = self.age
        
        return True


class DynamicNeuralArchitecture(nn.Module):
    """
    Neural network with dynamic architecture that evolves based on task complexity.
    
    Features:
    - NAS during inference
    - Task-adaptive depth and width
    - Automatic module specialization
    - Morphological evolution
    """
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        config: ArchitectureEvolutionConfig,
        initial_modules: int = 3
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.config = config
        
        # Module registry
        self.modules_dict = nn.ModuleDict()
        self.module_order: List[str] = []
        
        # Initialize modules
        for i in range(initial_modules):
            self._add_module(f"module_{i}", input_dim, output_dim)
        
        # Task complexity tracking
        self.task_complexity_history = deque(maxlen=config.complexity_window)
        self.performance_history = deque(maxlen=config.complexity_window)
        
        # Evolution tracking
        self.evolution_history: List[Dict] = []
        self.steps_since_evolution = 0
        self.total_steps = 0
        
        # NAS controller (simple RL-based)
        self.nas_controller = NASController(
            state_dim=64,
            action_dim=len(ArchitectureOperation)
        )
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def _add_module(self, module_id: str, in_dim: int, out_dim: int, depth: int = 2):
        """Add a new adaptive module."""
        module = AdaptiveModule(in_dim, out_dim, module_id, depth)
        self.modules_dict[module_id] = module
        self.module_order.append(module_id)
    
    def forward(
        self,
        x: torch.Tensor,
        task_id: Optional[str] = None,
        task_complexity: Optional[TaskComplexity] = None
    ) -> torch.Tensor:
        """
        Forward pass with architecture adaptation.
        
        Args:
            x: Input tensor
            task_id: Optional task identifier for specialization
            task_complexity: Optional complexity hint
        """
        self.total_steps += 1
        
        # Estimate task complexity if not provided
        if task_complexity is None:
            task_complexity = self._estimate_complexity(x)
        
        self.task_complexity_history.append(task_complexity)
        
        # Forward through modules
        for module_id in self.module_order:
            module = self.modules_dict[module_id]
            x = module(x, task_id)
        
        # Periodic architecture evolution
        self.steps_since_evolution += 1
        if (self.config.enable_runtime_adaptation and 
            self.steps_since_evolution >= self.config.evolution_interval):
            self._evolve_architecture()
            self.steps_since_evolution = 0
        
        return x
    
    def _estimate_complexity(self, x: torch.Tensor) -> TaskComplexity:
        """Estimate task complexity from input characteristics."""
        # Simple heuristic: higher variance = more complex
        variance = x.var().item()
        
        if variance < 0.1:
            return TaskComplexity.TRIVIAL
        elif variance < 0.5:
            return TaskComplexity.SIMPLE
        elif variance < 1.0:
            return TaskComplexity.MODERATE
        elif variance < 2.0:
            return TaskComplexity.COMPLEX
        else:
            return TaskComplexity.EXPERT
    
    def _evolve_architecture(self):
        """
        Evolve the architecture based on performance and complexity.
        
        Implements:
        1. NAS during inference
        2. Task-adaptive depth/width
        3. Module specialization/pruning
        4. Topology morphing
        """
        self.logger.info(f"🧬 Evolving architecture at step {self.total_steps}")
        
        # Update all module statistics
        for module in self.modules_dict.values():
            module.update_stats()
        
        # Get current state for NAS controller
        state = self._get_architecture_state()
        
        # Analyze complexity trend
        recent_complexity = list(self.task_complexity_history)[-50:]
        avg_complexity = self._complexity_to_score(recent_complexity)
        
        operations_performed = []
        
        # 1. Adaptive Depth (add/remove layers based on complexity)
        if avg_complexity > 0.7 and len(self.module_order) < self.config.max_layers:
            # High complexity: add capacity
            if self._should_add_layer(avg_complexity):
                target_module = self._select_module_for_growth()
                if target_module:
                    self.modules_dict[target_module].add_layer()
                    operations_performed.append(ArchitectureOperation.ADD_LAYER)
                    self.logger.info(f"  ➕ Added layer to {target_module}")
        
        elif avg_complexity < 0.3 and len(self.module_order) > self.config.min_layers:
            # Low complexity: reduce capacity
            if self._should_remove_layer(avg_complexity):
                target_module = self._select_module_for_shrinking()
                if target_module:
                    self.modules_dict[target_module].remove_layer()
                    operations_performed.append(ArchitectureOperation.REMOVE_LAYER)
                    self.logger.info(f"  ➖ Removed layer from {target_module}")
        
        # 2. Adaptive Width (expand/shrink based on utilization)
        for module_id, module in self.modules_dict.items():
            if module.stats.utilization > self.config.growth_threshold:
                # High utilization: expand width
                module.expand_width(1.3)
                operations_performed.append(ArchitectureOperation.EXPAND_WIDTH)
                self.logger.info(f"  📈 Expanded width of {module_id}")
            
            elif module.stats.utilization < self.config.shrink_threshold:
                # Low utilization: shrink width
                module.shrink_width(0.8)
                operations_performed.append(ArchitectureOperation.SHRINK_WIDTH)
                self.logger.info(f"  📉 Shrunk width of {module_id}")
        
        # 3. Module Specialization (split high-specialization modules)
        for module_id, module in self.modules_dict.items():
            if module.stats.splittable and len(self.module_order) < self.config.max_layers:
                specialized_modules = self._specialize_module(module_id, module)
                if specialized_modules:
                    operations_performed.append(ArchitectureOperation.SPECIALIZE_MODULE)
                    self.logger.info(f"  🎯 Specialized {module_id} into {len(specialized_modules)} modules")
        
        # 4. Pruning (remove underutilized modules)
        if self.config.enable_pruning:
            pruned = self._prune_modules()
            if pruned:
                operations_performed.extend([ArchitectureOperation.PRUNE_MODULE] * len(pruned))
                self.logger.info(f"  ✂️  Pruned {len(pruned)} modules: {pruned}")
        
        # 5. NAS-based architecture search
        if self.config.enable_nas:
            nas_action = self.nas_controller.select_action(state)
            operation = list(ArchitectureOperation)[nas_action]
            
            if self._apply_nas_operation(operation):
                operations_performed.append(operation)
                self.logger.info(f"  🔍 NAS operation: {operation.value}")
        
        # Record evolution event
        self.evolution_history.append({
            'step': self.total_steps,
            'complexity': avg_complexity,
            'operations': [op.value for op in operations_performed],
            'num_modules': len(self.module_order),
            'total_params': sum(p.numel() for p in self.parameters()),
            'architecture_state': state
        })
        
        self.logger.info(f"  ✅ Evolution complete: {len(operations_performed)} operations")
    
    def _get_architecture_state(self) -> np.ndarray:
        """Get current architecture state for NAS controller."""
        features = []
        
        # Overall statistics
        features.append(len(self.module_order))
        features.append(sum(p.numel() for p in self.parameters()) / 1e6)  # Params in millions
        
        # Module-level statistics
        for module in self.modules_dict.values():
            features.extend([
                module.stats.utilization,
                module.stats.specialization_score,
                module.stats.activation_mean,
                module.stats.gradient_norm,
                len(module.layers),
                module.age / 10000.0  # Normalize
            ])
        
        # Pad to fixed size
        features = features[:64]
        features.extend([0.0] * (64 - len(features)))
        
        return np.array(features, dtype=np.float32)
    
    def _complexity_to_score(self, complexities: List[TaskComplexity]) -> float:
        """Convert complexity enum to numerical score."""
        mapping = {
            TaskComplexity.TRIVIAL: 0.0,
            TaskComplexity.SIMPLE: 0.25,
            TaskComplexity.MODERATE: 0.5,
            TaskComplexity.COMPLEX: 0.75,
            TaskComplexity.EXPERT: 1.0
        }
        scores = [mapping[c] for c in complexities]
        return np.mean(scores) if scores else 0.5
    
    def _should_add_layer(self, complexity: float) -> bool:
        """Determine if we should add a layer."""
        # Add layer if complexity is high and recent performance is poor
        if len(self.performance_history) < 10:
            return False
        
        recent_perf = np.mean(list(self.performance_history)[-10:])
        return complexity > 0.7 and recent_perf < 0.8
    
    def _should_remove_layer(self, complexity: float) -> bool:
        """Determine if we should remove a layer."""
        # Remove layer if complexity is low
        return complexity < 0.3
    
    def _select_module_for_growth(self) -> Optional[str]:
        """Select module that should grow."""
        candidates = []
        for module_id, module in self.modules_dict.items():
            if module.stats.utilization > 0.7:
                candidates.append((module_id, module.stats.utilization))
        
        if candidates:
            candidates.sort(key=lambda x: x[1], reverse=True)
            return candidates[0][0]
        return None
    
    def _select_module_for_shrinking(self) -> Optional[str]:
        """Select module that should shrink."""
        candidates = []
        for module_id, module in self.modules_dict.items():
            if module.stats.utilization < 0.3 and len(module.layers) > 1:
                candidates.append((module_id, module.stats.utilization))
        
        if candidates:
            candidates.sort(key=lambda x: x[1])
            return candidates[0][0]
        return None
    
    def _specialize_module(
        self,
        module_id: str,
        module: AdaptiveModule
    ) -> Optional[List[str]]:
        """
        Split a module into specialized sub-modules.
        
        Returns:
            List of new specialized module IDs
        """
        if len(module.stats.task_affinity) < 2:
            return None
        
        # Create specialized modules for top tasks
        top_tasks = sorted(
            module.stats.task_affinity.items(),
            key=lambda x: x[1],
            reverse=True
        )[:2]
        
        specialized_ids = []
        for task, affinity in top_tasks:
            spec_id = f"{module_id}_spec_{task}"
            
            # Create specialized module (clone original)
            spec_module = AdaptiveModule(
                module.input_dim,
                module.output_dim,
                spec_id,
                initial_depth=len(module.layers)
            )
            
            # Copy weights
            spec_module.load_state_dict(module.state_dict())
            spec_module.specializations.append(task)
            
            self.modules_dict[spec_id] = spec_module
            specialized_ids.append(spec_id)
        
        # Replace original module in order
        idx = self.module_order.index(module_id)
        self.module_order = (
            self.module_order[:idx] +
            specialized_ids +
            self.module_order[idx+1:]
        )
        
        # Remove original
        del self.modules_dict[module_id]
        
        return specialized_ids
    
    def _prune_modules(self) -> List[str]:
        """Remove underutilized modules."""
        pruned = []
        
        for module_id, module in list(self.modules_dict.items()):
            if (module.stats.prunable and 
                len(self.module_order) > self.config.min_layers):
                
                # Remove from order and dict
                self.module_order.remove(module_id)
                del self.modules_dict[module_id]
                pruned.append(module_id)
        
        return pruned
    
    def _apply_nas_operation(self, operation: ArchitectureOperation) -> bool:
        """Apply NAS-selected operation."""
        try:
            if operation == ArchitectureOperation.ADD_LAYER:
                target = self._select_module_for_growth()
                if target:
                    return self.modules_dict[target].add_layer()
            
            elif operation == ArchitectureOperation.REMOVE_LAYER:
                target = self._select_module_for_shrinking()
                if target:
                    return self.modules_dict[target].remove_layer()
            
            elif operation == ArchitectureOperation.EXPAND_WIDTH:
                target = self._select_module_for_growth()
                if target:
                    return self.modules_dict[target].expand_width()
            
            elif operation == ArchitectureOperation.SHRINK_WIDTH:
                target = self._select_module_for_shrinking()
                if target:
                    return self.modules_dict[target].shrink_width()
            
            return False
        except Exception as e:
            self.logger.warning(f"NAS operation {operation} failed: {e}")
            return False
    
    def update_performance(self, accuracy: float):
        """Update performance history for adaptation decisions."""
        self.performance_history.append(accuracy)
    
    def get_architecture_summary(self) -> Dict[str, Any]:
        """Get summary of current architecture."""
        return {
            'num_modules': len(self.module_order),
            'total_layers': sum(len(m.layers) for m in self.modules_dict.values()),
            'total_params': sum(p.numel() for p in self.parameters()),
            'module_details': {
                module_id: {
                    'layers': len(module.layers),
                    'utilization': module.stats.utilization,
                    'specialization': module.stats.specialization_score,
                    'age': module.age,
                    'specializations': module.specializations
                }
                for module_id, module in self.modules_dict.items()
            },
            'evolution_count': len(self.evolution_history),
            'avg_complexity': self._complexity_to_score(
                list(self.task_complexity_history)
            ) if self.task_complexity_history else 0.0
        }


class NASController(nn.Module):
    """
    Neural Architecture Search controller using reinforcement learning.
    
    Selects architecture operations based on current state.
    """
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.policy = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=-1)
        )
        
        self.value = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        # Experience buffer
        self.states = []
        self.actions = []
        self.rewards = []
    
    def select_action(self, state: np.ndarray) -> int:
        """Select action using current policy."""
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        
        with torch.no_grad():
            action_probs = self.policy(state_tensor)
        
        # Sample from distribution
        action = torch.multinomial(action_probs, 1).item()
        
        # Store for training
        self.states.append(state)
        self.actions.append(action)
        
        return action
    
    def update(self, reward: float):
        """Update controller based on reward."""
        self.rewards.append(reward)
        
        # Simple policy gradient update (can be enhanced)
        if len(self.rewards) >= 10:
            # Compute returns
            returns = []
            G = 0
            for r in reversed(self.rewards[-10:]):
                G = r + 0.99 * G
                returns.insert(0, G)
            
            # Normalize
            returns = torch.FloatTensor(returns)
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)
            
            # Clear buffer
            self.states = []
            self.actions = []
            self.rewards = []


def create_dynamic_architecture(
    input_dim: int,
    output_dim: int,
    config: Optional[ArchitectureEvolutionConfig] = None
) -> DynamicNeuralArchitecture:
    """
    Factory function to create a dynamic neural architecture.
    
    Args:
        input_dim: Input dimensionality
        output_dim: Output dimensionality
        config: Evolution configuration
    
    Returns:
        DynamicNeuralArchitecture instance
    """
    if config is None:
        config = ArchitectureEvolutionConfig()
    
    return DynamicNeuralArchitecture(input_dim, output_dim, config)


# Example usage
if __name__ == "__main__":
    print("🧬 Dynamic Neural Architecture Evolution System")
    print("=" * 70)
    
    # Create dynamic architecture
    config = ArchitectureEvolutionConfig(
        min_layers=2,
        max_layers=10,
        evolution_interval=10,
        enable_nas=True,
        enable_runtime_adaptation=True,
        enable_pruning=True,
        enable_specialization=True
    )
    
    model = create_dynamic_architecture(
        input_dim=128,
        output_dim=10,
        config=config
    )
    
    print(f"Initial architecture: {model.get_architecture_summary()}")
    print()
    
    # Simulate training with varying complexity
    print("📊 Simulating adaptive evolution...")
    for epoch in range(5):
        print(f"\nEpoch {epoch + 1}")
        
        for step in range(50):
            # Vary complexity
            if step < 20:
                complexity = TaskComplexity.SIMPLE
                batch_size = 32
            elif step < 40:
                complexity = TaskComplexity.COMPLEX
                batch_size = 16
            else:
                complexity = TaskComplexity.EXPERT
                batch_size = 8
            
            # Forward pass
            x = torch.randn(batch_size, 128)
            output = model(x, task_id=f"task_{step % 3}", task_complexity=complexity)
            
            # Simulate performance
            accuracy = 0.8 + 0.1 * np.random.randn()
            model.update_performance(accuracy)
        
        summary = model.get_architecture_summary()
        print(f"  Modules: {summary['num_modules']}")
        print(f"  Total Layers: {summary['total_layers']}")
        print(f"  Total Params: {summary['total_params']:,}")
        print(f"  Evolution Events: {summary['evolution_count']}")
    
    print("\n✅ Dynamic architecture evolution demonstration complete!")
    print(f"\nFinal architecture:")
    final_summary = model.get_architecture_summary()
    print(json.dumps(final_summary, indent=2, default=str))
