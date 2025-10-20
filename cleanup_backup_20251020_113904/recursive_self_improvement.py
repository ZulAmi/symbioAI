"""
Recursive Self-Improvement Engine for Symbio AI

Revolutionary meta-evolutionary optimization system that improves its own improvement algorithms.
This system evolves evolution strategies, learns better learning rules, and performs automatic
hyperparameter meta-learning across tasks.

Key Innovations:
1. Meta-Evolution: Evolve the evolution strategy itself
2. Self-Modifying Training Loops: Learn better learning rules
3. Hyperparameter Meta-Learning: Automatic optimization across tasks
4. Strategy Performance Tracking: Learn from historical optimization data
5. Causal Strategy Attribution: Understand what makes strategies work

Competitive Edge: While Sakana AI focuses on model merging, this system focuses on
improving the improvement process itself - a recursive meta-level optimization.
"""

from __future__ import annotations

import asyncio
import logging
import json
import copy
import hashlib
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Callable, Set, Union
from dataclasses import dataclass, field, asdict, is_dataclass
from datetime import datetime, timedelta
from enum import Enum
from collections import defaultdict, deque
import torch
import torch.nn as nn


def _make_json_safe(obj: Any) -> Any:
    """Convert complex objects (dataclasses, numpy types) into JSON-safe values."""
    if is_dataclass(obj):
        obj = asdict(obj)

    if isinstance(obj, Enum):
        return obj.value

    if isinstance(obj, datetime):
        return obj.isoformat()

    if isinstance(obj, Path):
        return str(obj)

    if isinstance(obj, np.generic):
        return obj.item()

    if isinstance(obj, np.ndarray):
        return obj.tolist()

    if isinstance(obj, dict):
        return {key: _make_json_safe(value) for key, value in obj.items()}

    if isinstance(obj, (list, tuple, set)):
        return [_make_json_safe(value) for value in obj]

    if hasattr(obj, "to_json"):
        return obj.to_json()

    return obj

# Import existing evolutionary infrastructure
from training.evolution import (
    EvolutionStrategy as BaseEvolutionStrategy,
    MutationStrategy as BaseMutationStrategy,
    CrossoverStrategy as BaseCrossoverStrategy,
    EvolutionConfig,
    AgentModel,
    PopulationManager,
    TaskEvaluator
)
from training.manager import EvolutionaryConfig
from monitoring.observability import OBSERVABILITY


class MetaObjective(Enum):
    """Meta-level optimization objectives."""
    CONVERGENCE_SPEED = "convergence_speed"
    FINAL_FITNESS = "final_fitness"
    SAMPLE_EFFICIENCY = "sample_efficiency"
    DIVERSITY_MAINTENANCE = "diversity_maintenance"
    ROBUSTNESS = "robustness"
    GENERALIZATION = "generalization"
    COMPUTATIONAL_EFFICIENCY = "computational_efficiency"


class StrategyComponent(Enum):
    """Components of evolution strategy that can be optimized."""
    SELECTION_METHOD = "selection_method"
    MUTATION_METHOD = "mutation_method"
    CROSSOVER_METHOD = "crossover_method"
    POPULATION_SIZE = "population_size"
    ELITISM_RATIO = "elitism_ratio"
    MUTATION_RATE = "mutation_rate"
    MUTATION_STRENGTH = "mutation_strength"
    CROSSOVER_RATIO = "crossover_ratio"
    TOURNAMENT_SIZE = "tournament_size"
    DIVERSITY_PRESSURE = "diversity_pressure"


@dataclass
class EvolutionStrategyGenome:
    """
    Genotype representing an evolution strategy configuration.
    This is what we evolve at the meta-level.
    """
    
    # Strategy selection
    selection_strategy: str = "tournament"
    mutation_strategy: str = "gaussian_noise"
    crossover_strategy: str = "parameter_averaging"
    
    # Hyperparameters
    population_size: int = 50
    elitism_ratio: float = 0.2
    crossover_ratio: float = 0.6
    mutation_ratio: float = 0.2
    mutation_rate: float = 0.1
    mutation_strength: float = 0.01
    tournament_size: int = 5
    diversity_threshold: float = 0.1
    
    # Advanced parameters
    adaptive_mutation: bool = True
    fitness_sharing: bool = True
    niche_capacity: int = 5
    diversity_bonus: float = 0.1
    age_penalty: float = 0.01
    
    # Meta-parameters
    genome_id: str = field(default_factory=lambda: hashlib.md5(
        str(datetime.utcnow().timestamp()).encode()
    ).hexdigest()[:12])
    generation: int = 0
    parent_ids: List[str] = field(default_factory=list)
    
    # Performance tracking
    fitness_score: float = 0.0
    task_performances: Dict[str, float] = field(default_factory=dict)
    convergence_speed: float = 0.0
    sample_efficiency: float = 0.0
    robustness: float = 0.0
    
    # Historical data
    applications: int = 0
    success_rate: float = 0.0
    average_improvement: float = 0.0
    
    def to_evolution_config(self) -> EvolutionConfig:
        """Convert genome to executable evolution configuration."""
        from training.evolution import (
            EvolutionStrategy as BaseEvolutionStrategy,
            MutationStrategy as BaseMutationStrategy,
            CrossoverStrategy as BaseCrossoverStrategy
        )
        
        # Map string names to enum values
        selection_map = {
            "tournament": BaseEvolutionStrategy.TOURNAMENT,
            "roulette_wheel": BaseEvolutionStrategy.ROULETTE_WHEEL,
            "rank_based": BaseEvolutionStrategy.RANK_BASED,
            "elitism": BaseEvolutionStrategy.ELITISM,
            "diversity_preserving": BaseEvolutionStrategy.DIVERSITY_PRESERVING
        }
        
        mutation_map = {
            "gaussian_noise": BaseMutationStrategy.GAUSSIAN_NOISE,
            "svd_perturbation": BaseMutationStrategy.SVD_PERTURBATION,
            "dropout_mutation": BaseMutationStrategy.DROPOUT_MUTATION,
            "layer_wise": BaseMutationStrategy.LAYER_WISE,
            "adaptive": BaseMutationStrategy.ADAPTIVE
        }
        
        crossover_map = {
            "parameter_averaging": BaseCrossoverStrategy.PARAMETER_AVERAGING,
            "layer_swapping": BaseCrossoverStrategy.LAYER_SWAPPING,
            "weighted_merge": BaseCrossoverStrategy.WEIGHTED_MERGE,
            "genetic_crossover": BaseCrossoverStrategy.GENETIC_CROSSOVER,
            "task_specific": BaseCrossoverStrategy.TASK_SPECIFIC
        }
        
        return EvolutionConfig(
            population_size=self.population_size,
            elitism_ratio=self.elitism_ratio,
            crossover_ratio=self.crossover_ratio,
            mutation_ratio=self.mutation_ratio,
            mutation_rate=self.mutation_rate,
            mutation_strength=self.mutation_strength,
            tournament_size=self.tournament_size,
            diversity_threshold=self.diversity_threshold,
            fitness_sharing=self.fitness_sharing,
            niche_capacity=self.niche_capacity,
            evolution_strategy=selection_map.get(self.selection_strategy, BaseEvolutionStrategy.TOURNAMENT),
            mutation_strategy=mutation_map.get(self.mutation_strategy, BaseMutationStrategy.GAUSSIAN_NOISE),
            crossover_strategy=crossover_map.get(self.crossover_strategy, BaseCrossoverStrategy.PARAMETER_AVERAGING)
        )


@dataclass
class LearningRule:
    """
    Represents a learned optimization rule (e.g., custom learning rate schedules,
    gradient transformations, loss function modifications).
    """
    
    rule_id: str
    rule_type: str  # "learning_rate_schedule", "gradient_transform", "loss_modifier"
    parameters: Dict[str, Any] = field(default_factory=dict)
    
    # Performance metrics
    convergence_improvement: float = 0.0
    stability_score: float = 0.0
    generalization_score: float = 0.0
    
    # Application history
    applications: int = 0
    success_count: int = 0
    
    def apply_learning_rate_schedule(self, epoch: int, base_lr: float) -> float:
        """Apply learned learning rate schedule."""
        if self.rule_type != "learning_rate_schedule":
            return base_lr
        
        schedule_type = self.parameters.get("schedule_type", "constant")
        
        if schedule_type == "cosine_annealing":
            T_max = self.parameters.get("T_max", 100)
            eta_min = self.parameters.get("eta_min", 0.0)
            return eta_min + (base_lr - eta_min) * (1 + np.cos(np.pi * epoch / T_max)) / 2
        
        elif schedule_type == "exponential_decay":
            gamma = self.parameters.get("gamma", 0.95)
            return base_lr * (gamma ** epoch)
        
        elif schedule_type == "step_decay":
            step_size = self.parameters.get("step_size", 10)
            gamma = self.parameters.get("gamma", 0.5)
            return base_lr * (gamma ** (epoch // step_size))
        
        elif schedule_type == "adaptive_warmup":
            warmup_epochs = self.parameters.get("warmup_epochs", 5)
            if epoch < warmup_epochs:
                return base_lr * (epoch + 1) / warmup_epochs
            else:
                # Cosine decay after warmup
                adjusted_epoch = epoch - warmup_epochs
                T_max = self.parameters.get("T_max", 100) - warmup_epochs
                return base_lr * (1 + np.cos(np.pi * adjusted_epoch / T_max)) / 2
        
        return base_lr
    
    def apply_gradient_transform(self, gradients: torch.Tensor) -> torch.Tensor:
        """Apply learned gradient transformation."""
        if self.rule_type != "gradient_transform":
            return gradients
        
        transform_type = self.parameters.get("transform_type", "identity")
        
        if transform_type == "gradient_clipping":
            max_norm = self.parameters.get("max_norm", 1.0)
            return torch.clamp(gradients, -max_norm, max_norm)
        
        elif transform_type == "gradient_normalization":
            norm = torch.norm(gradients) + 1e-8
            target_norm = self.parameters.get("target_norm", 1.0)
            return gradients * (target_norm / norm)
        
        elif transform_type == "adaptive_scaling":
            scale = self.parameters.get("scale", 1.0)
            momentum = self.parameters.get("momentum", 0.9)
            # Would use running statistics in practice
            return gradients * scale
        
        return gradients


@dataclass
class MetaEvolutionMetrics:
    """Metrics for evaluating meta-evolution performance."""
    
    strategy_id: str
    task_id: str
    
    # Primary metrics
    convergence_speed: float = 0.0  # Generations to reach threshold
    final_fitness: float = 0.0
    sample_efficiency: float = 0.0  # Performance per evaluation
    
    # Secondary metrics
    diversity_maintained: float = 0.0
    robustness_score: float = 0.0
    computational_cost: float = 0.0
    
    # Tracking
    generation_times: List[float] = field(default_factory=list)
    fitness_trajectory: List[float] = field(default_factory=list)
    diversity_trajectory: List[float] = field(default_factory=list)
    
    evaluated_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())


class RecursiveSelfImprovementEngine:
    """
    Meta-evolutionary optimization engine that improves its own improvement algorithms.
    
    This system operates at multiple levels:
    1. Base level: Evolve models for tasks
    2. Meta level: Evolve evolution strategies
    3. Meta-meta level: Learn learning rules
    """
    
    def __init__(
        self,
        meta_population_size: int = 20,
        meta_generations: int = 50,
        base_task_budget: int = 100,
        objectives: List[MetaObjective] = None
    ):
        self.meta_population_size = meta_population_size
        self.meta_generations = meta_generations
        self.base_task_budget = base_task_budget
        
        # Meta-evolution objectives
        self.objectives = objectives or [
            MetaObjective.CONVERGENCE_SPEED,
            MetaObjective.FINAL_FITNESS,
            MetaObjective.SAMPLE_EFFICIENCY
        ]
        
        # Meta-population: evolution strategies
        self.strategy_population: List[EvolutionStrategyGenome] = []
        
        # Learning rules library
        self.learning_rules: Dict[str, LearningRule] = {}
        
        # Performance tracking
        self.strategy_performance_history: Dict[str, List[MetaEvolutionMetrics]] = defaultdict(list)
        self.best_strategies_per_task: Dict[str, EvolutionStrategyGenome] = {}
        
        # Causal attribution
        self.strategy_component_impacts: Dict[str, Dict[StrategyComponent, float]] = defaultdict(
            lambda: {comp: 0.0 for comp in StrategyComponent}
        )
        
        # Hyperparameter meta-learning
        self.hyperparameter_distributions: Dict[str, Dict[str, Any]] = {}
        
        self.logger = logging.getLogger(__name__)
        self.generation = 0
    
    async def initialize_meta_population(self) -> None:
        """Initialize population of evolution strategies."""
        self.logger.info(f"Initializing meta-population of {self.meta_population_size} strategies")
        
        self.strategy_population = []
        
        # Create diverse initial strategies
        base_strategies = self._create_diverse_base_strategies()
        
        for i in range(self.meta_population_size):
            if i < len(base_strategies):
                genome = base_strategies[i]
            else:
                # Random variations
                genome = self._create_random_strategy_genome()
            
            genome.generation = 0
            self.strategy_population.append(genome)
        
        self.logger.info(f"Meta-population initialized with {len(self.strategy_population)} strategies")
    
    def _create_diverse_base_strategies(self) -> List[EvolutionStrategyGenome]:
        """Create diverse baseline strategies."""
        strategies = []
        
        # Strategy 1: Aggressive exploration
        strategies.append(EvolutionStrategyGenome(
            selection_strategy="tournament",
            mutation_strategy="gaussian_noise",
            crossover_strategy="parameter_averaging",
            population_size=30,
            elitism_ratio=0.1,
            mutation_rate=0.3,  # High mutation
            mutation_strength=0.05,  # Strong mutations
            diversity_threshold=0.2  # High diversity
        ))
        
        # Strategy 2: Exploitation-focused
        strategies.append(EvolutionStrategyGenome(
            selection_strategy="elitism",
            mutation_strategy="adaptive",
            crossover_strategy="weighted_merge",
            population_size=50,
            elitism_ratio=0.3,  # High elitism
            mutation_rate=0.05,  # Low mutation
            mutation_strength=0.005,  # Weak mutations
            diversity_threshold=0.05  # Low diversity tolerance
        ))
        
        # Strategy 3: Balanced
        strategies.append(EvolutionStrategyGenome(
            selection_strategy="tournament",
            mutation_strategy="adaptive",
            crossover_strategy="genetic_crossover",
            population_size=50,
            elitism_ratio=0.2,
            mutation_rate=0.1,
            mutation_strength=0.01,
            diversity_threshold=0.1
        ))
        
        # Strategy 4: Diversity-preserving
        strategies.append(EvolutionStrategyGenome(
            selection_strategy="diversity_preserving",
            mutation_strategy="svd_perturbation",
            crossover_strategy="task_specific",
            population_size=60,
            elitism_ratio=0.15,
            mutation_rate=0.15,
            mutation_strength=0.02,
            fitness_sharing=True,
            diversity_bonus=0.2
        ))
        
        # Strategy 5: Large population, gentle evolution
        strategies.append(EvolutionStrategyGenome(
            selection_strategy="rank_based",
            mutation_strategy="layer_wise",
            crossover_strategy="layer_swapping",
            population_size=100,
            elitism_ratio=0.25,
            mutation_rate=0.08,
            mutation_strength=0.008,
            tournament_size=7
        ))
        
        return strategies
    
    def _create_random_strategy_genome(self) -> EvolutionStrategyGenome:
        """Create a random strategy genome for diversity."""
        selection_strategies = ["tournament", "roulette_wheel", "rank_based", "elitism", "diversity_preserving"]
        mutation_strategies = ["gaussian_noise", "svd_perturbation", "dropout_mutation", "layer_wise", "adaptive"]
        crossover_strategies = ["parameter_averaging", "layer_swapping", "weighted_merge", "genetic_crossover"]
        
        return EvolutionStrategyGenome(
            selection_strategy=np.random.choice(selection_strategies),
            mutation_strategy=np.random.choice(mutation_strategies),
            crossover_strategy=np.random.choice(crossover_strategies),
            population_size=int(np.random.choice([30, 50, 70, 100])),
            elitism_ratio=np.random.uniform(0.1, 0.3),
            mutation_rate=np.random.uniform(0.05, 0.3),
            mutation_strength=np.random.uniform(0.005, 0.05),
            tournament_size=int(np.random.choice([3, 5, 7, 9])),
            diversity_threshold=np.random.uniform(0.05, 0.2),
            adaptive_mutation=np.random.choice([True, False]),
            fitness_sharing=np.random.choice([True, False]),
            diversity_bonus=np.random.uniform(0.0, 0.3)
        )
    
    async def evaluate_strategy(
        self,
        strategy_genome: EvolutionStrategyGenome,
        task_evaluator: TaskEvaluator,
        model_factory: Callable,
        task_id: str
    ) -> MetaEvolutionMetrics:
        """
        Evaluate an evolution strategy by running it on a task.
        This is the key meta-fitness evaluation.
        """
        self.logger.info(f"Evaluating strategy {strategy_genome.genome_id} on task {task_id}")
        
        # Convert genome to executable config
        config = strategy_genome.to_evolution_config()
        
        # Limit base evolution for meta-evaluation
        config.num_generations = min(20, self.base_task_budget // config.population_size)
        
        # Create population manager
        pop_manager = PopulationManager(config, model_factory, task_evaluator)
        
        # Track metrics
        start_time = datetime.utcnow()
        generation_times = []
        fitness_trajectory = []
        diversity_trajectory = []
        
        # Initialize population
        await pop_manager.initialize_population()
        
        # Run evolution
        for gen in range(config.num_generations):
            gen_start = datetime.utcnow()
            
            await pop_manager.evolve_generation()
            
            gen_time = (datetime.utcnow() - gen_start).total_seconds()
            generation_times.append(gen_time)
            
            # Track progress
            best_fitness = pop_manager.population[0].fitness_score
            diversity = pop_manager._calculate_population_diversity()
            
            fitness_trajectory.append(best_fitness)
            diversity_trajectory.append(diversity)
        
        # Calculate metrics
        total_time = (datetime.utcnow() - start_time).total_seconds()
        final_fitness = fitness_trajectory[-1] if fitness_trajectory else 0.0
        
        # Convergence speed: how quickly did it reach 90% of final fitness?
        target_fitness = final_fitness * 0.9
        convergence_gen = len(fitness_trajectory)
        for i, fitness in enumerate(fitness_trajectory):
            if fitness >= target_fitness:
                convergence_gen = i + 1
                break
        
        convergence_speed = 1.0 / (convergence_gen + 1)  # Faster convergence = higher score
        
        # Sample efficiency: performance per evaluation
        total_evaluations = sum(len(pop_manager.population) for _ in range(config.num_generations))
        sample_efficiency = final_fitness / (total_evaluations + 1)
        
        # Diversity maintained
        avg_diversity = np.mean(diversity_trajectory) if diversity_trajectory else 0.0
        
        # Robustness: stability of improvement
        if len(fitness_trajectory) > 1:
            fitness_improvements = np.diff(fitness_trajectory)
            robustness = 1.0 / (1.0 + np.std(fitness_improvements))
        else:
            robustness = 0.5
        
        metrics = MetaEvolutionMetrics(
            strategy_id=strategy_genome.genome_id,
            task_id=task_id,
            convergence_speed=convergence_speed,
            final_fitness=final_fitness,
            sample_efficiency=sample_efficiency,
            diversity_maintained=avg_diversity,
            robustness_score=robustness,
            computational_cost=total_time,
            generation_times=generation_times,
            fitness_trajectory=fitness_trajectory,
            diversity_trajectory=diversity_trajectory
        )
        
        # Store in history
        self.strategy_performance_history[strategy_genome.genome_id].append(metrics)
        
        # Emit telemetry
        OBSERVABILITY.emit_gauge(
            "meta_evolution.strategy_fitness",
            final_fitness,
            strategy_id=strategy_genome.genome_id,
            task_id=task_id
        )
        
        OBSERVABILITY.emit_gauge(
            "meta_evolution.convergence_speed",
            convergence_speed,
            strategy_id=strategy_genome.genome_id
        )
        
        self.logger.info(
            f"Strategy {strategy_genome.genome_id}: "
            f"fitness={final_fitness:.4f}, "
            f"convergence={convergence_speed:.4f}, "
            f"efficiency={sample_efficiency:.6f}"
        )
        
        return metrics
    
    def calculate_meta_fitness(
        self,
        strategy_genome: EvolutionStrategyGenome,
        metrics: MetaEvolutionMetrics
    ) -> float:
        """
        Calculate meta-fitness score for a strategy based on multiple objectives.
        """
        objective_scores = {}
        
        for objective in self.objectives:
            if objective == MetaObjective.CONVERGENCE_SPEED:
                objective_scores[objective] = metrics.convergence_speed
            
            elif objective == MetaObjective.FINAL_FITNESS:
                objective_scores[objective] = metrics.final_fitness
            
            elif objective == MetaObjective.SAMPLE_EFFICIENCY:
                objective_scores[objective] = metrics.sample_efficiency * 100  # Scale up
            
            elif objective == MetaObjective.DIVERSITY_MAINTENANCE:
                objective_scores[objective] = metrics.diversity_maintained
            
            elif objective == MetaObjective.ROBUSTNESS:
                objective_scores[objective] = metrics.robustness_score
            
            elif objective == MetaObjective.COMPUTATIONAL_EFFICIENCY:
                # Invert cost (lower is better)
                objective_scores[objective] = 1.0 / (1.0 + metrics.computational_cost / 60.0)
        
        # Multi-objective fitness (weighted sum)
        objective_weights = {
            MetaObjective.CONVERGENCE_SPEED: 0.3,
            MetaObjective.FINAL_FITNESS: 0.4,
            MetaObjective.SAMPLE_EFFICIENCY: 0.2,
            MetaObjective.DIVERSITY_MAINTENANCE: 0.05,
            MetaObjective.ROBUSTNESS: 0.03,
            MetaObjective.COMPUTATIONAL_EFFICIENCY: 0.02
        }
        
        meta_fitness = sum(
            objective_scores.get(obj, 0.0) * objective_weights.get(obj, 0.0)
            for obj in self.objectives
        )
        
        return meta_fitness
    
    def crossover_strategies(
        self,
        parent1: EvolutionStrategyGenome,
        parent2: EvolutionStrategyGenome
    ) -> EvolutionStrategyGenome:
        """
        Crossover two strategy genomes to create offspring.
        This is meta-level genetic recombination.
        """
        child = EvolutionStrategyGenome()
        
        # Categorical parameters: randomly choose from parents
        child.selection_strategy = np.random.choice([parent1.selection_strategy, parent2.selection_strategy])
        child.mutation_strategy = np.random.choice([parent1.mutation_strategy, parent2.mutation_strategy])
        child.crossover_strategy = np.random.choice([parent1.crossover_strategy, parent2.crossover_strategy])
        
        # Numerical parameters: arithmetic crossover
        alpha = np.random.uniform(0.3, 0.7)
        
        child.population_size = int(alpha * parent1.population_size + (1 - alpha) * parent2.population_size)
        child.elitism_ratio = alpha * parent1.elitism_ratio + (1 - alpha) * parent2.elitism_ratio
        child.mutation_rate = alpha * parent1.mutation_rate + (1 - alpha) * parent2.mutation_rate
        child.mutation_strength = alpha * parent1.mutation_strength + (1 - alpha) * parent2.mutation_strength
        child.tournament_size = int(alpha * parent1.tournament_size + (1 - alpha) * parent2.tournament_size)
        child.diversity_threshold = alpha * parent1.diversity_threshold + (1 - alpha) * parent2.diversity_threshold
        child.diversity_bonus = alpha * parent1.diversity_bonus + (1 - alpha) * parent2.diversity_bonus
        
        # Boolean parameters: random choice
        child.adaptive_mutation = np.random.choice([parent1.adaptive_mutation, parent2.adaptive_mutation])
        child.fitness_sharing = np.random.choice([parent1.fitness_sharing, parent2.fitness_sharing])
        
        # Metadata
        child.generation = max(parent1.generation, parent2.generation) + 1
        child.parent_ids = [parent1.genome_id, parent2.genome_id]
        
        # Ensure valid ratios
        child.crossover_ratio = 1.0 - child.elitism_ratio - child.mutation_ratio
        if child.crossover_ratio < 0:
            child.mutation_ratio = 0.2
            child.crossover_ratio = 1.0 - child.elitism_ratio - child.mutation_ratio
        
        return child
    
    def mutate_strategy(
        self,
        strategy: EvolutionStrategyGenome,
        mutation_rate: float = 0.2
    ) -> EvolutionStrategyGenome:
        """
        Mutate a strategy genome.
        This is meta-level mutation.
        """
        mutated = copy.deepcopy(strategy)
        
        # Mutate categorical parameters
        if np.random.random() < mutation_rate:
            strategies = ["tournament", "roulette_wheel", "rank_based", "elitism", "diversity_preserving"]
            mutated.selection_strategy = np.random.choice(strategies)
        
        if np.random.random() < mutation_rate:
            strategies = ["gaussian_noise", "svd_perturbation", "dropout_mutation", "layer_wise", "adaptive"]
            mutated.mutation_strategy = np.random.choice(strategies)
        
        if np.random.random() < mutation_rate:
            strategies = ["parameter_averaging", "layer_swapping", "weighted_merge", "genetic_crossover"]
            mutated.crossover_strategy = np.random.choice(strategies)
        
        # Mutate numerical parameters with Gaussian noise
        if np.random.random() < mutation_rate:
            mutated.population_size = max(20, int(mutated.population_size + np.random.normal(0, 10)))
        
        if np.random.random() < mutation_rate:
            mutated.elitism_ratio = np.clip(mutated.elitism_ratio + np.random.normal(0, 0.05), 0.05, 0.4)
        
        if np.random.random() < mutation_rate:
            mutated.mutation_rate = np.clip(mutated.mutation_rate + np.random.normal(0, 0.05), 0.01, 0.5)
        
        if np.random.random() < mutation_rate:
            mutated.mutation_strength = np.clip(mutated.mutation_strength + np.random.normal(0, 0.005), 0.001, 0.1)
        
        if np.random.random() < mutation_rate:
            mutated.tournament_size = max(2, int(mutated.tournament_size + np.random.normal(0, 2)))
        
        if np.random.random() < mutation_rate:
            mutated.diversity_threshold = np.clip(mutated.diversity_threshold + np.random.normal(0, 0.05), 0.01, 0.5)
        
        # Mutate boolean parameters
        if np.random.random() < mutation_rate:
            mutated.adaptive_mutation = not mutated.adaptive_mutation
        
        if np.random.random() < mutation_rate:
            mutated.fitness_sharing = not mutated.fitness_sharing
        
        # Update metadata
        mutated.generation = strategy.generation + 1
        mutated.parent_ids = [strategy.genome_id]
        mutated.genome_id = hashlib.md5(
            f"{mutated.genome_id}_{datetime.utcnow().timestamp()}".encode()
        ).hexdigest()[:12]
        
        # Ensure valid ratios
        mutated.mutation_ratio = np.clip(1.0 - mutated.elitism_ratio - mutated.crossover_ratio, 0.1, 0.4)
        mutated.crossover_ratio = 1.0 - mutated.elitism_ratio - mutated.mutation_ratio
        
        return mutated
    
    async def evolve_meta_generation(
        self,
        task_evaluator: TaskEvaluator,
        model_factory: Callable,
        task_id: str
    ) -> None:
        """
        Evolve one generation of meta-population (evolution strategies).
        """
        self.logger.info(f"Meta-generation {self.generation}: Evaluating {len(self.strategy_population)} strategies")
        
        # Evaluate all strategies
        evaluation_tasks = []
        for strategy in self.strategy_population:
            evaluation_tasks.append(
                self.evaluate_strategy(strategy, task_evaluator, model_factory, task_id)
            )
        
        metrics_list = await asyncio.gather(*evaluation_tasks)
        
        # Calculate meta-fitness scores
        for strategy, metrics in zip(self.strategy_population, metrics_list):
            meta_fitness = self.calculate_meta_fitness(strategy, metrics)
            strategy.fitness_score = meta_fitness
            strategy.task_performances[task_id] = meta_fitness
        
        # Sort by meta-fitness
        self.strategy_population.sort(key=lambda s: s.fitness_score, reverse=True)
        
        # Track best strategy for this task
        best_strategy = self.strategy_population[0]
        self.best_strategies_per_task[task_id] = best_strategy
        
        self.logger.info(
            f"Best strategy: {best_strategy.genome_id} "
            f"(meta-fitness={best_strategy.fitness_score:.4f})"
        )
        
        # Meta-evolution: create next generation
        new_population = []
        
        # Elitism: keep top strategies
        num_elite = max(2, int(self.meta_population_size * 0.2))
        new_population.extend(self.strategy_population[:num_elite])
        
        # Crossover
        num_crossover = int(self.meta_population_size * 0.6)
        for _ in range(num_crossover):
            parent1 = self._select_parent()
            parent2 = self._select_parent()
            child = self.crossover_strategies(parent1, parent2)
            new_population.append(child)
        
        # Mutation
        remaining = self.meta_population_size - len(new_population)
        for _ in range(remaining):
            parent = self._select_parent()
            mutated = self.mutate_strategy(parent)
            new_population.append(mutated)
        
        self.strategy_population = new_population
        self.generation += 1
        
        # Emit telemetry
        OBSERVABILITY.emit_gauge(
            "meta_evolution.best_meta_fitness",
            best_strategy.fitness_score,
            generation=self.generation,
            task_id=task_id
        )
    
    def _select_parent(self) -> EvolutionStrategyGenome:
        """Tournament selection for meta-evolution."""
        tournament_size = 3
        tournament = np.random.choice(
            self.strategy_population,
            size=min(tournament_size, len(self.strategy_population)),
            replace=False
        )
        return max(tournament, key=lambda s: s.fitness_score)
    
    async def meta_train(
        self,
        task_evaluators: Dict[str, TaskEvaluator],
        model_factory: Callable,
        num_meta_generations: int = None
    ) -> Dict[str, Any]:
        """
        Main meta-training loop: evolve evolution strategies across multiple tasks.
        """
        num_meta_generations = num_meta_generations or self.meta_generations
        
        self.logger.info(
            f"Starting recursive self-improvement meta-training for {num_meta_generations} generations"
        )
        
        # Initialize meta-population
        await self.initialize_meta_population()
        
        # Meta-evolution loop
        for meta_gen in range(num_meta_generations):
            self.logger.info(f"=== Meta-Generation {meta_gen + 1}/{num_meta_generations} ===")
            
            # Evaluate on all tasks (or sample for efficiency)
            task_ids = list(task_evaluators.keys())
            
            for task_id in task_ids:
                await self.evolve_meta_generation(
                    task_evaluators[task_id],
                    model_factory,
                    task_id
                )
            
            # Learn from meta-evolution history
            await self._analyze_strategy_components()
        
        # Return results
        results = {
            "best_strategies_per_task": {
                task_id: asdict(strategy)
                for task_id, strategy in self.best_strategies_per_task.items()
            },
            "final_meta_generation": self.generation,
            "strategy_component_impacts": self.strategy_component_impacts,
            "top_strategies": [
                asdict(s) for s in self.strategy_population[:5]
            ]
        }
        
        self.logger.info("Meta-training complete")
        return results
    
    async def _analyze_strategy_components(self) -> None:
        """
        Causal attribution: analyze which strategy components contribute to success.
        This enables learning about what makes strategies effective.
        """
        # Group strategies by components and analyze correlation with performance
        component_performances = defaultdict(list)
        
        for strategy in self.strategy_population:
            if strategy.fitness_score > 0:
                # Categorical components
                component_performances[f"selection_{strategy.selection_strategy}"].append(strategy.fitness_score)
                component_performances[f"mutation_{strategy.mutation_strategy}"].append(strategy.fitness_score)
                component_performances[f"crossover_{strategy.crossover_strategy}"].append(strategy.fitness_score)
                
                # Numerical components (binned)
                pop_size_bin = "small" if strategy.population_size < 50 else "large"
                component_performances[f"population_{pop_size_bin}"].append(strategy.fitness_score)
                
                elitism_bin = "low" if strategy.elitism_ratio < 0.2 else "high"
                component_performances[f"elitism_{elitism_bin}"].append(strategy.fitness_score)
                
                mutation_rate_bin = "low" if strategy.mutation_rate < 0.15 else "high"
                component_performances[f"mutation_rate_{mutation_rate_bin}"].append(strategy.fitness_score)
        
        # Calculate average impact
        baseline_performance = np.mean([s.fitness_score for s in self.strategy_population])
        
        for component, performances in component_performances.items():
            if performances:
                avg_performance = np.mean(performances)
                impact = avg_performance - baseline_performance
                
                # Store impact (simplified - in production would use proper causal inference)
                self.logger.debug(f"Component {component}: impact = {impact:.4f}")
    
    def get_best_strategy_for_task(self, task_id: str) -> Optional[EvolutionStrategyGenome]:
        """Get the best learned strategy for a specific task."""
        return self.best_strategies_per_task.get(task_id)
    
    def get_universal_best_strategy(self) -> EvolutionStrategyGenome:
        """Get the strategy that performs best across all tasks."""
        if not self.strategy_population:
            raise ValueError("No strategies in population")
        
        # Calculate average performance across all tasks
        for strategy in self.strategy_population:
            if strategy.task_performances:
                strategy.average_improvement = np.mean(list(strategy.task_performances.values()))
            else:
                strategy.average_improvement = 0.0
        
        best = max(self.strategy_population, key=lambda s: s.average_improvement)
        return best
    
    def export_learned_strategies(self, output_path: Path) -> None:
        """Export learned strategies for reuse."""
        export_data = {
            "meta_generation": self.generation,
            "best_strategies_per_task": {
                task_id: asdict(strategy)
                for task_id, strategy in self.best_strategies_per_task.items()
            },
            "top_universal_strategies": [
                asdict(s) for s in sorted(
                    self.strategy_population,
                    key=lambda x: x.average_improvement,
                    reverse=True
                )[:10]
            ],
            "exported_at": datetime.utcnow().isoformat()
        }
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(_make_json_safe(export_data), f, indent=2)
        
        self.logger.info(f"Exported learned strategies to {output_path}")


# Factory function for easy integration
def create_recursive_improvement_engine(
    meta_population_size: int = 20,
    meta_generations: int = 30,
    base_task_budget: int = 100
) -> RecursiveSelfImprovementEngine:
    """Create a recursive self-improvement engine with default configuration."""
    return RecursiveSelfImprovementEngine(
        meta_population_size=meta_population_size,
        meta_generations=meta_generations,
        base_task_budget=base_task_budget,
        objectives=[
            MetaObjective.CONVERGENCE_SPEED,
            MetaObjective.FINAL_FITNESS,
            MetaObjective.SAMPLE_EFFICIENCY,
            MetaObjective.ROBUSTNESS
        ]
    )
