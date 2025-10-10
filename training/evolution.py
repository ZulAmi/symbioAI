#!/usr/bin/env python3
"""
Evolutionary Training: Evolve a population of small models to acquire diverse skills

1. Initialize a population of N agent models with different random seeds or initializations.
2. For each generation:
   a. Evaluate each agent on a set of tasks (compute a fitness score per agent).
   b. Select top-performing agents (elitism) for breeding.
   c. Generate new agents by crossover (merge parameters of two parents) and mutation (perturb weights or use SVD on weights).
   d. Replace the worst-performing agents with new offspring.
3. Iterate for G generations or until convergence.
4. Return the best agent models and their niche specializations.

This evolutionary learning algorithm incorporates concepts from nature (selection, crossover, mutation) 
to train a population of models, following patterns similar to Sakana's CycleQD approach with 
quality-diversity and model merging operations for robust, adaptable performance.
"""

import asyncio
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import random
import logging
import json
import copy
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union, Callable, Set
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from enum import Enum
from datetime import datetime
import time
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib

# Import from our existing system
import sys
sys.path.append(str(Path(__file__).parent.parent))

from core.pipeline import Config, TrainingCallback
from models.registry import BaseModel, ModelMetadata, ModelFramework
from monitoring.production import MetricsCollector, ProductionLogger
from evaluation.benchmarks import BenchmarkResult, MetricType


class EvolutionStrategy(Enum):
    """Evolution strategy types."""
    ELITISM = "elitism"
    TOURNAMENT = "tournament"
    ROULETTE_WHEEL = "roulette_wheel"
    RANK_BASED = "rank_based"
    DIVERSITY_PRESERVING = "diversity_preserving"


class MutationStrategy(Enum):
    """Mutation strategy types."""
    GAUSSIAN_NOISE = "gaussian_noise"
    SVD_PERTURBATION = "svd_perturbation"
    DROPOUT_MUTATION = "dropout_mutation"
    LAYER_WISE = "layer_wise"
    ADAPTIVE = "adaptive"


class CrossoverStrategy(Enum):
    """Crossover strategy types."""
    PARAMETER_AVERAGING = "parameter_averaging"
    LAYER_SWAPPING = "layer_swapping"
    WEIGHTED_MERGE = "weighted_merge"
    GENETIC_CROSSOVER = "genetic_crossover"
    TASK_SPECIFIC = "task_specific"


class TaskType(Enum):
    """Task types for evaluation."""
    CLASSIFICATION = "classification"
    REGRESSION = "regression"
    GENERATION = "generation"
    REINFORCEMENT = "reinforcement"
    MULTIMODAL = "multimodal"
    REASONING = "reasoning"


@dataclass
class AgentModel:
    """Individual agent model in the population."""
    id: str
    model: nn.Module
    fitness_score: float = 0.0
    task_performances: Dict[str, float] = field(default_factory=dict)
    specializations: List[str] = field(default_factory=list)
    generation: int = 0
    parent_ids: List[str] = field(default_factory=list)
    mutation_history: List[str] = field(default_factory=list)
    diversity_metrics: Dict[str, float] = field(default_factory=dict)
    
    def __post_init__(self):
        if not self.id:
            self.id = f"agent_{hashlib.md5(str(time.time()).encode()).hexdigest()[:8]}"


@dataclass
class EvolutionConfig:
    """Configuration for evolutionary training."""
    population_size: int = 50
    num_generations: int = 100
    elitism_ratio: float = 0.2  # Top 20% survive
    crossover_ratio: float = 0.6  # 60% from crossover
    mutation_ratio: float = 0.2  # 20% from mutation
    mutation_rate: float = 0.1
    mutation_strength: float = 0.01
    tournament_size: int = 5
    diversity_threshold: float = 0.1
    convergence_threshold: float = 0.001
    max_stagnation: int = 10
    fitness_sharing: bool = True
    niche_capacity: int = 5
    task_weights: Dict[str, float] = field(default_factory=lambda: {"default": 1.0})
    evolution_strategy: EvolutionStrategy = EvolutionStrategy.ELITISM
    mutation_strategy: MutationStrategy = MutationStrategy.GAUSSIAN_NOISE
    crossover_strategy: CrossoverStrategy = CrossoverStrategy.PARAMETER_AVERAGING
    
    def __post_init__(self):
        # Validate ratios sum to approximately 1.0
        total_ratio = self.elitism_ratio + self.crossover_ratio + self.mutation_ratio
        if abs(total_ratio - 1.0) > 0.01:
            raise ValueError(f"Population ratios must sum to 1.0, got {total_ratio}")


class TaskEvaluator(ABC):
    """Abstract base class for task evaluation."""
    
    @abstractmethod
    async def evaluate(self, model: nn.Module, task_id: str) -> float:
        """Evaluate a model on a specific task."""
        pass
    
    @abstractmethod
    def get_task_list(self) -> List[str]:
        """Get list of available tasks."""
        pass


class MultiTaskEvaluator(TaskEvaluator):
    """Multi-task evaluator for diverse skill assessment."""
    
    def __init__(self, tasks: Dict[str, Callable], task_weights: Dict[str, float] = None):
        self.tasks = tasks
        self.task_weights = task_weights or {task: 1.0 for task in tasks}
        self.logger = ProductionLogger(__name__)
    
    async def evaluate(self, model: nn.Module, task_id: str) -> float:
        """Evaluate model on a specific task."""
        if task_id not in self.tasks:
            raise ValueError(f"Unknown task: {task_id}")
        
        try:
            model.eval()
            with torch.no_grad():
                score = await self.tasks[task_id](model)
                return float(score)
        except Exception as e:
            self.logger.error(f"Task evaluation failed for {task_id}: {e}")
            return 0.0
    
    def get_task_list(self) -> List[str]:
        """Get list of available tasks."""
        return list(self.tasks.keys())
    
    async def evaluate_all_tasks(self, model: nn.Module) -> Dict[str, float]:
        """Evaluate model on all tasks."""
        results = {}
        for task_id in self.tasks:
            results[task_id] = await self.evaluate(model, task_id)
        return results


class PopulationManager:
    """Manages the population of agent models."""
    
    def __init__(self, config: EvolutionConfig, model_factory: Callable, evaluator: TaskEvaluator):
        self.config = config
        self.model_factory = model_factory
        self.evaluator = evaluator
        self.population: List[AgentModel] = []
        self.generation = 0
        self.best_fitness_history: List[float] = []
        self.diversity_history: List[float] = []
        self.stagnation_counter = 0
        self.logger = ProductionLogger(__name__)
        self.metrics = MetricsCollector()
    
    async def initialize_population(self) -> None:
        """Initialize a population of N agent models with different random seeds or initializations."""
        self.logger.info(f"Initializing population of {self.config.population_size} agents")
        
        self.population = []
        for i in range(self.config.population_size):
            # Create model with different random seed
            torch.manual_seed(i * 42 + random.randint(0, 10000))
            np.random.seed(i * 42 + random.randint(0, 10000))
            
            model = self.model_factory()
            agent = AgentModel(
                id=f"gen0_agent_{i:03d}",
                model=model,
                generation=0
            )
            self.population.append(agent)
        
        self.logger.info(f"Population initialized with {len(self.population)} agents")
    
    async def evaluate_population(self) -> None:
        """Evaluate each agent on a set of tasks (compute a fitness score per agent)."""
        self.logger.info(f"Evaluating population - Generation {self.generation}")
        
        tasks = self.evaluator.get_task_list()
        
        # Evaluate all agents concurrently for better performance
        evaluation_tasks = []
        for agent in self.population:
            evaluation_tasks.append(self._evaluate_agent(agent, tasks))
        
        await asyncio.gather(*evaluation_tasks)
        
        # Calculate fitness scores
        for agent in self.population:
            agent.fitness_score = self._calculate_fitness(agent.task_performances)
        
        # Sort population by fitness
        self.population.sort(key=lambda x: x.fitness_score, reverse=True)
        
        # Track metrics
        best_fitness = self.population[0].fitness_score
        avg_fitness = np.mean([agent.fitness_score for agent in self.population])
        diversity = self._calculate_population_diversity()
        
        self.best_fitness_history.append(best_fitness)
        self.diversity_history.append(diversity)
        
        self.logger.info(f"Generation {self.generation} - Best: {best_fitness:.4f}, Avg: {avg_fitness:.4f}, Diversity: {diversity:.4f}")
    
    async def _evaluate_agent(self, agent: AgentModel, tasks: List[str]) -> None:
        """Evaluate a single agent on all tasks."""
        agent.task_performances = {}
        
        for task_id in tasks:
            try:
                score = await self.evaluator.evaluate(agent.model, task_id)
                agent.task_performances[task_id] = score
            except Exception as e:
                self.logger.warning(f"Failed to evaluate agent {agent.id} on task {task_id}: {e}")
                agent.task_performances[task_id] = 0.0
    
    def _calculate_fitness(self, task_performances: Dict[str, float]) -> float:
        """Calculate overall fitness from task performances."""
        if not task_performances:
            return 0.0
        
        weighted_sum = 0.0
        total_weight = 0.0
        
        for task_id, score in task_performances.items():
            weight = self.config.task_weights.get(task_id, 1.0)
            weighted_sum += score * weight
            total_weight += weight
        
        return weighted_sum / total_weight if total_weight > 0 else 0.0
    
    def select_elite(self) -> List[AgentModel]:
        """Select top-performing agents (elitism) for breeding."""
        num_elite = max(1, int(self.config.population_size * self.config.elitism_ratio))
        elite = self.population[:num_elite]
        
        self.logger.debug(f"Selected {len(elite)} elite agents")
        return elite
    
    def select_parents(self, num_pairs: int) -> List[Tuple[AgentModel, AgentModel]]:
        """Select parent pairs based on evolution strategy."""
        pairs = []
        
        if self.config.evolution_strategy == EvolutionStrategy.TOURNAMENT:
            for _ in range(num_pairs):
                parent1 = self._tournament_selection()
                parent2 = self._tournament_selection()
                pairs.append((parent1, parent2))
        
        elif self.config.evolution_strategy == EvolutionStrategy.ROULETTE_WHEEL:
            for _ in range(num_pairs):
                parent1 = self._roulette_wheel_selection()
                parent2 = self._roulette_wheel_selection()
                pairs.append((parent1, parent2))
        
        elif self.config.evolution_strategy == EvolutionStrategy.RANK_BASED:
            for _ in range(num_pairs):
                parent1 = self._rank_based_selection()
                parent2 = self._rank_based_selection()
                pairs.append((parent1, parent2))
        
        else:  # Default to elite selection
            elite = self.select_elite()
            for _ in range(num_pairs):
                parent1 = random.choice(elite)
                parent2 = random.choice(elite)
                pairs.append((parent1, parent2))
        
        return pairs
    
    def _tournament_selection(self) -> AgentModel:
        """Tournament selection method."""
        tournament = random.sample(self.population, min(self.config.tournament_size, len(self.population)))
        return max(tournament, key=lambda x: x.fitness_score)
    
    def _roulette_wheel_selection(self) -> AgentModel:
        """Roulette wheel selection method."""
        total_fitness = sum(agent.fitness_score for agent in self.population)
        if total_fitness <= 0:
            return random.choice(self.population)
        
        pick = random.uniform(0, total_fitness)
        current = 0
        for agent in self.population:
            current += agent.fitness_score
            if current >= pick:
                return agent
        return self.population[-1]
    
    def _rank_based_selection(self) -> AgentModel:
        """Rank-based selection method."""
        # Population is already sorted by fitness
        ranks = list(range(len(self.population), 0, -1))  # Higher rank for better fitness
        total_rank = sum(ranks)
        
        pick = random.uniform(0, total_rank)
        current = 0
        for i, rank in enumerate(ranks):
            current += rank
            if current >= pick:
                return self.population[i]
        return self.population[-1]
    
    def crossover(self, parent1: AgentModel, parent2: AgentModel) -> AgentModel:
        """Generate new agents by crossover (merge parameters of two parents)."""
        child_model = self.model_factory()
        
        if self.config.crossover_strategy == CrossoverStrategy.PARAMETER_AVERAGING:
            self._parameter_averaging_crossover(child_model, parent1.model, parent2.model)
        
        elif self.config.crossover_strategy == CrossoverStrategy.LAYER_SWAPPING:
            self._layer_swapping_crossover(child_model, parent1.model, parent2.model)
        
        elif self.config.crossover_strategy == CrossoverStrategy.WEIGHTED_MERGE:
            self._weighted_merge_crossover(child_model, parent1.model, parent2.model)
        
        elif self.config.crossover_strategy == CrossoverStrategy.GENETIC_CROSSOVER:
            self._genetic_crossover(child_model, parent1.model, parent2.model)
        
        else:  # Default to parameter averaging
            self._parameter_averaging_crossover(child_model, parent1.model, parent2.model)
        
        child = AgentModel(
            model=child_model,
            generation=self.generation + 1,
            parent_ids=[parent1.id, parent2.id]
        )
        
        return child
    
    def _parameter_averaging_crossover(self, child: nn.Module, parent1: nn.Module, parent2: nn.Module) -> None:
        """Simple parameter averaging crossover."""
        with torch.no_grad():
            for (child_param, p1_param, p2_param) in zip(child.parameters(), parent1.parameters(), parent2.parameters()):
                alpha = random.uniform(0.3, 0.7)  # Mixing ratio
                child_param.data.copy_(alpha * p1_param.data + (1 - alpha) * p2_param.data)
    
    def _layer_swapping_crossover(self, child: nn.Module, parent1: nn.Module, parent2: nn.Module) -> None:
        """Layer-wise swapping crossover."""
        with torch.no_grad():
            child_params = list(child.parameters())
            p1_params = list(parent1.parameters())
            p2_params = list(parent2.parameters())
            
            for i, (child_param, p1_param, p2_param) in enumerate(zip(child_params, p1_params, p2_params)):
                # Randomly choose which parent to inherit from for each layer
                if random.random() < 0.5:
                    child_param.data.copy_(p1_param.data)
                else:
                    child_param.data.copy_(p2_param.data)
    
    def _weighted_merge_crossover(self, child: nn.Module, parent1: nn.Module, parent2: nn.Module) -> None:
        """Fitness-weighted parameter merging."""
        fitness1 = getattr(parent1, '_fitness', 1.0)
        fitness2 = getattr(parent2, '_fitness', 1.0)
        total_fitness = fitness1 + fitness2
        
        if total_fitness > 0:
            w1 = fitness1 / total_fitness
            w2 = fitness2 / total_fitness
        else:
            w1 = w2 = 0.5
        
        with torch.no_grad():
            for (child_param, p1_param, p2_param) in zip(child.parameters(), parent1.parameters(), parent2.parameters()):
                child_param.data.copy_(w1 * p1_param.data + w2 * p2_param.data)
    
    def _genetic_crossover(self, child: nn.Module, parent1: nn.Module, parent2: nn.Module) -> None:
        """Genetic-style crossover with random crossover points."""
        with torch.no_grad():
            for (child_param, p1_param, p2_param) in zip(child.parameters(), parent1.parameters(), parent2.parameters()):
                # Create random mask for crossover
                mask = torch.rand_like(child_param.data) < 0.5
                child_param.data = torch.where(mask, p1_param.data, p2_param.data)
    
    def mutate(self, agent: AgentModel) -> AgentModel:
        """Mutation (perturb weights or use SVD on weights)."""
        mutated_model = copy.deepcopy(agent.model)
        
        if self.config.mutation_strategy == MutationStrategy.GAUSSIAN_NOISE:
            self._gaussian_noise_mutation(mutated_model)
        
        elif self.config.mutation_strategy == MutationStrategy.SVD_PERTURBATION:
            self._svd_perturbation_mutation(mutated_model)
        
        elif self.config.mutation_strategy == MutationStrategy.DROPOUT_MUTATION:
            self._dropout_mutation(mutated_model)
        
        elif self.config.mutation_strategy == MutationStrategy.LAYER_WISE:
            self._layer_wise_mutation(mutated_model)
        
        elif self.config.mutation_strategy == MutationStrategy.ADAPTIVE:
            self._adaptive_mutation(mutated_model, agent)
        
        else:  # Default to Gaussian noise
            self._gaussian_noise_mutation(mutated_model)
        
        mutated_agent = AgentModel(
            model=mutated_model,
            generation=self.generation + 1,
            parent_ids=[agent.id],
            mutation_history=agent.mutation_history + [self.config.mutation_strategy.value]
        )
        
        return mutated_agent
    
    def _gaussian_noise_mutation(self, model: nn.Module) -> None:
        """Add Gaussian noise to parameters."""
        with torch.no_grad():
            for param in model.parameters():
                if random.random() < self.config.mutation_rate:
                    noise = torch.randn_like(param) * self.config.mutation_strength
                    param.data.add_(noise)
    
    def _svd_perturbation_mutation(self, model: nn.Module) -> None:
        """SVD-based parameter perturbation."""
        with torch.no_grad():
            for param in model.parameters():
                if random.random() < self.config.mutation_rate and param.dim() >= 2:
                    # Apply SVD perturbation
                    U, S, V = torch.svd(param.data)
                    # Perturb singular values
                    S_perturbed = S + torch.randn_like(S) * self.config.mutation_strength
                    S_perturbed = torch.clamp(S_perturbed, min=0)  # Keep positive
                    param.data.copy_(U @ torch.diag(S_perturbed) @ V.t())
    
    def _dropout_mutation(self, model: nn.Module) -> None:
        """Dropout-style mutation (randomly set some weights to zero)."""
        with torch.no_grad():
            for param in model.parameters():
                if random.random() < self.config.mutation_rate:
                    dropout_mask = torch.rand_like(param) > self.config.mutation_strength
                    param.data.mul_(dropout_mask.float())
    
    def _layer_wise_mutation(self, model: nn.Module) -> None:
        """Layer-wise adaptive mutation."""
        with torch.no_grad():
            for i, param in enumerate(model.parameters()):
                # Different mutation strength per layer
                layer_mutation_strength = self.config.mutation_strength * (1 + i * 0.1)
                if random.random() < self.config.mutation_rate:
                    noise = torch.randn_like(param) * layer_mutation_strength
                    param.data.add_(noise)
    
    def _adaptive_mutation(self, model: nn.Module, agent: AgentModel) -> None:
        """Adaptive mutation based on agent performance."""
        # Adapt mutation strength based on fitness
        fitness_factor = 1.0 - min(agent.fitness_score, 1.0)  # Lower fitness -> higher mutation
        adaptive_strength = self.config.mutation_strength * (1 + fitness_factor)
        
        with torch.no_grad():
            for param in model.parameters():
                if random.random() < self.config.mutation_rate:
                    noise = torch.randn_like(param) * adaptive_strength
                    param.data.add_(noise)
    
    async def evolve_generation(self) -> None:
        """Execute one generation of evolution."""
        self.logger.info(f"Starting evolution - Generation {self.generation}")
        
        # 1. Evaluate current population
        await self.evaluate_population()
        
        # 2. Check convergence
        if self._check_convergence():
            self.logger.info("Convergence reached, stopping evolution")
            return
        
        # 3. Create next generation
        new_population = []
        
        # Elite selection (keep best performers)
        elite = self.select_elite()
        new_population.extend(copy.deepcopy(elite))
        
        # Crossover to generate offspring
        num_crossover = int(self.config.population_size * self.config.crossover_ratio)
        parent_pairs = self.select_parents(num_crossover)
        
        crossover_tasks = []
        for parent1, parent2 in parent_pairs:
            crossover_tasks.append(self._create_crossover_child(parent1, parent2))
        
        crossover_children = await asyncio.gather(*crossover_tasks)
        new_population.extend(crossover_children)
        
        # Mutation to generate diverse offspring
        num_mutation = int(self.config.population_size * self.config.mutation_ratio)
        mutation_parents = random.choices(self.population, k=num_mutation)
        
        mutation_tasks = []
        for parent in mutation_parents:
            mutation_tasks.append(self._create_mutation_child(parent))
        
        mutation_children = await asyncio.gather(*mutation_tasks)
        new_population.extend(mutation_children)
        
        # Ensure exact population size
        if len(new_population) > self.config.population_size:
            new_population = new_population[:self.config.population_size]
        elif len(new_population) < self.config.population_size:
            # Fill remaining slots with random mutations
            while len(new_population) < self.config.population_size:
                parent = random.choice(self.population)
                child = self.mutate(parent)
                new_population.append(child)
        
        # Replace population
        self.population = new_population
        self.generation += 1
        
        # Update diversity and specialization
        await self._update_specializations()
        
        self.logger.info(f"Generation {self.generation} complete - Population size: {len(self.population)}")
    
    async def _create_crossover_child(self, parent1: AgentModel, parent2: AgentModel) -> AgentModel:
        """Create child through crossover."""
        return self.crossover(parent1, parent2)
    
    async def _create_mutation_child(self, parent: AgentModel) -> AgentModel:
        """Create child through mutation."""
        return self.mutate(parent)
    
    def _check_convergence(self) -> bool:
        """Check if evolution has converged."""
        if len(self.best_fitness_history) < 2:
            return False
        
        # Check for stagnation
        recent_improvement = self.best_fitness_history[-1] - self.best_fitness_history[-2]
        if recent_improvement < self.config.convergence_threshold:
            self.stagnation_counter += 1
        else:
            self.stagnation_counter = 0
        
        return self.stagnation_counter >= self.config.max_stagnation
    
    def _calculate_population_diversity(self) -> float:
        """Calculate population diversity metric."""
        if len(self.population) < 2:
            return 0.0
        
        # Calculate parameter diversity
        param_vectors = []
        for agent in self.population:
            params = []
            for param in agent.model.parameters():
                params.extend(param.data.flatten().cpu().numpy())
            param_vectors.append(np.array(params))
        
        if not param_vectors:
            return 0.0
        
        # Calculate pairwise distances
        distances = []
        for i in range(len(param_vectors)):
            for j in range(i + 1, len(param_vectors)):
                dist = np.linalg.norm(param_vectors[i] - param_vectors[j])
                distances.append(dist)
        
        return np.mean(distances) if distances else 0.0
    
    async def _update_specializations(self) -> None:
        """Update agent specializations based on task performance."""
        for agent in self.population:
            if not agent.task_performances:
                continue
            
            # Find tasks where agent performs well
            avg_performance = np.mean(list(agent.task_performances.values()))
            specializations = []
            
            for task, performance in agent.task_performances.items():
                if performance > avg_performance + self.config.diversity_threshold:
                    specializations.append(task)
            
            agent.specializations = specializations
    
    def get_best_agents(self, n: int = 5) -> List[AgentModel]:
        """Return the best agent models and their niche specializations."""
        return sorted(self.population, key=lambda x: x.fitness_score, reverse=True)[:n]
    
    def get_diverse_agents(self, n: int = 5) -> List[AgentModel]:
        """Return diverse agents covering different specializations."""
        # Group agents by specialization
        specialization_groups = defaultdict(list)
        for agent in self.population:
            if agent.specializations:
                for spec in agent.specializations:
                    specialization_groups[spec].append(agent)
            else:
                specialization_groups['generalist'].append(agent)
        
        # Select best agent from each specialization
        diverse_agents = []
        for spec, agents in specialization_groups.items():
            if agents:
                best_in_spec = max(agents, key=lambda x: x.fitness_score)
                diverse_agents.append(best_in_spec)
        
        # Sort by fitness and return top n
        diverse_agents.sort(key=lambda x: x.fitness_score, reverse=True)
        return diverse_agents[:n]


class EvolutionaryTrainer:
    """Main evolutionary training orchestrator."""
    
    def __init__(self, config: EvolutionConfig, model_factory: Callable, 
                 evaluator: TaskEvaluator, callbacks: List[TrainingCallback] = None):
        self.config = config
        self.model_factory = model_factory
        self.evaluator = evaluator
        self.callbacks = callbacks or []
        self.population_manager = PopulationManager(config, model_factory, evaluator)
        self.logger = ProductionLogger(__name__)
        self.metrics = MetricsCollector()
        self.training_start_time = None
        
    async def train(self) -> Dict[str, Any]:
        """
        Iterate for G generations or until convergence.
        Return the best agent models and their niche specializations.
        """
        self.logger.info("Starting evolutionary training")
        self.training_start_time = time.time()
        
        # Initialize callbacks
        for callback in self.callbacks:
            await callback.on_train_start()
        
        try:
            # Initialize population
            await self.population_manager.initialize_population()
            
            # Evolution loop
            for generation in range(self.config.num_generations):
                generation_start = time.time()
                
                # Execute generation
                await self.population_manager.evolve_generation()
                
                # Callback hooks
                for callback in self.callbacks:
                    await callback.on_generation_end(generation, self.population_manager.population)
                
                # Log generation metrics
                generation_time = time.time() - generation_start
                best_fitness = self.population_manager.population[0].fitness_score
                diversity = self.population_manager._calculate_population_diversity()
                
                self.metrics.record_metric("generation_time", generation_time)
                self.metrics.record_metric("best_fitness", best_fitness)
                self.metrics.record_metric("population_diversity", diversity)
                
                self.logger.info(f"Generation {generation} completed in {generation_time:.2f}s")
                
                # Check early stopping
                if self.population_manager._check_convergence():
                    self.logger.info(f"Early stopping at generation {generation}")
                    break
            
            # Training complete
            training_time = time.time() - self.training_start_time
            
            # Get final results
            best_agents = self.population_manager.get_best_agents(5)
            diverse_agents = self.population_manager.get_diverse_agents(5)
            
            results = {
                "best_agents": best_agents,
                "diverse_agents": diverse_agents,
                "final_generation": self.population_manager.generation,
                "training_time": training_time,
                "best_fitness_history": self.population_manager.best_fitness_history,
                "diversity_history": self.population_manager.diversity_history,
                "population_size": len(self.population_manager.population),
                "convergence_achieved": self.population_manager._check_convergence()
            }
            
            # Final callbacks
            for callback in self.callbacks:
                await callback.on_train_end(results)
            
            self.logger.info(f"Evolutionary training completed in {training_time:.2f}s")
            return results
            
        except Exception as e:
            self.logger.error(f"Training failed: {e}")
            for callback in self.callbacks:
                await callback.on_train_error(e)
            raise


# Specialized task evaluators for different domains
class ClassificationTaskEvaluator(TaskEvaluator):
    """Evaluator for classification tasks."""
    
    def __init__(self, datasets: Dict[str, DataLoader]):
        self.datasets = datasets
        self.tasks = list(datasets.keys())
    
    async def evaluate(self, model: nn.Module, task_id: str) -> float:
        """Evaluate classification accuracy."""
        if task_id not in self.datasets:
            return 0.0
        
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, targets in self.datasets[task_id]:
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
        
        return correct / total if total > 0 else 0.0
    
    def get_task_list(self) -> List[str]:
        return self.tasks


class RegressionTaskEvaluator(TaskEvaluator):
    """Evaluator for regression tasks."""
    
    def __init__(self, datasets: Dict[str, DataLoader]):
        self.datasets = datasets
        self.tasks = list(datasets.keys())
    
    async def evaluate(self, model: nn.Module, task_id: str) -> float:
        """Evaluate regression MSE (inverted for maximization)."""
        if task_id not in self.datasets:
            return 0.0
        
        model.eval()
        total_loss = 0.0
        num_samples = 0
        
        with torch.no_grad():
            for inputs, targets in self.datasets[task_id]:
                outputs = model(inputs)
                loss = F.mse_loss(outputs, targets, reduction='sum')
                total_loss += loss.item()
                num_samples += targets.size(0)
        
        mse = total_loss / num_samples if num_samples > 0 else float('inf')
        return 1.0 / (1.0 + mse)  # Convert to maximization objective


# Example usage and factory functions
def create_simple_mlp(input_dim: int, hidden_dim: int, output_dim: int, num_layers: int = 2) -> nn.Module:
    """Factory function for creating simple MLP models."""
    layers = []
    
    # Input layer
    layers.append(nn.Linear(input_dim, hidden_dim))
    layers.append(nn.ReLU())
    
    # Hidden layers
    for _ in range(num_layers - 2):
        layers.append(nn.Linear(hidden_dim, hidden_dim))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(0.1))
    
    # Output layer
    layers.append(nn.Linear(hidden_dim, output_dim))
    
    return nn.Sequential(*layers)


async def create_mock_tasks() -> Dict[str, Callable]:
    """Create mock tasks for demonstration."""
    
    async def classification_task(model: nn.Module) -> float:
        """Mock classification task."""
        # Generate random data
        x = torch.randn(32, 10)
        y = torch.randint(0, 3, (32,))
        
        model.eval()
        with torch.no_grad():
            outputs = model(x)
            if outputs.shape[1] >= 3:
                _, predicted = torch.max(outputs[:, :3], 1)
                accuracy = (predicted == y).float().mean().item()
                return accuracy
            else:
                return random.uniform(0.3, 0.9)
    
    async def regression_task(model: nn.Module) -> float:
        """Mock regression task."""
        # Generate random data
        x = torch.randn(32, 10)
        y = torch.randn(32, 1)
        
        model.eval()
        with torch.no_grad():
            outputs = model(x)
            if outputs.shape[1] >= 1:
                mse = F.mse_loss(outputs[:, :1], y).item()
                return 1.0 / (1.0 + mse)
            else:
                return random.uniform(0.3, 0.9)
    
    async def reasoning_task(model: nn.Module) -> float:
        """Mock reasoning task."""
        # Simulate reasoning capability
        return random.uniform(0.1, 0.8)
    
    return {
        "classification": classification_task,
        "regression": regression_task,
        "reasoning": reasoning_task
    }


# Example training function
async def run_evolutionary_training_example():
    """Example of running evolutionary training."""
    
    # Configuration
    config = EvolutionConfig(
        population_size=20,
        num_generations=10,
        elitism_ratio=0.2,
        crossover_ratio=0.6,
        mutation_ratio=0.2,
        mutation_rate=0.1,
        evolution_strategy=EvolutionStrategy.TOURNAMENT,
        mutation_strategy=MutationStrategy.GAUSSIAN_NOISE,
        crossover_strategy=CrossoverStrategy.PARAMETER_AVERAGING
    )
    
    # Model factory
    def model_factory():
        return create_simple_mlp(input_dim=10, hidden_dim=64, output_dim=3, num_layers=3)
    
    # Task evaluator
    tasks = await create_mock_tasks()
    evaluator = MultiTaskEvaluator(tasks)
    
    # Training callbacks
    class EvolutionCallback(TrainingCallback):
        async def on_generation_end(self, generation: int, population: List[AgentModel]):
            best_agent = max(population, key=lambda x: x.fitness_score)
            print(f"Generation {generation}: Best fitness = {best_agent.fitness_score:.4f}")
    
    callbacks = [EvolutionCallback()]
    
    # Create trainer
    trainer = EvolutionaryTrainer(config, model_factory, evaluator, callbacks)
    
    # Run training
    results = await trainer.train()
    
    # Print results
    print(f"\nEvolutionary Training Results:")
    print(f"Training completed in {results['training_time']:.2f} seconds")
    print(f"Final generation: {results['final_generation']}")
    print(f"Convergence achieved: {results['convergence_achieved']}")
    
    print(f"\nTop 3 Best Agents:")
    for i, agent in enumerate(results['best_agents'][:3], 1):
        print(f"{i}. Agent {agent.id}: Fitness = {agent.fitness_score:.4f}, Specializations = {agent.specializations}")
    
    print(f"\nDiverse Agents:")
    for i, agent in enumerate(results['diverse_agents'], 1):
        print(f"{i}. Agent {agent.id}: Fitness = {agent.fitness_score:.4f}, Specializations = {agent.specializations}")


if __name__ == "__main__":
    # Run example
    asyncio.run(run_evolutionary_training_example())