"""
Training Management System for Symbio AI

Implements evolutionary training algorithms, population-based optimization,
and adaptive learning strategies for model improvement.
"""

import asyncio
import logging
import random
import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Callable
from dataclasses import dataclass, field, asdict
from abc import ABC, abstractmethod
from enum import Enum
import numpy as np
import time
from datetime import datetime
import copy


class TrainingStatus(Enum):
    """Training status enumeration."""
    INITIALIZING = "initializing"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    STOPPED = "stopped"


class SelectionStrategy(Enum):
    """Selection strategies for evolutionary training."""
    TOURNAMENT = "tournament"
    ROULETTE = "roulette"
    RANK = "rank"
    ELITE = "elite"


class CrossoverStrategy(Enum):
    """Crossover strategies for model breeding."""
    UNIFORM = "uniform"
    SINGLE_POINT = "single_point"
    TWO_POINT = "two_point"
    ARITHMETIC = "arithmetic"


class MutationStrategy(Enum):
    """Mutation strategies for model variation."""
    GAUSSIAN = "gaussian"
    UNIFORM = "uniform"
    ADAPTIVE = "adaptive"
    LAYER_WISE = "layer_wise"


@dataclass
class Individual:
    """Represents an individual in the evolutionary population."""
    id: str
    model_id: str
    fitness: float = 0.0
    generation: int = 0
    parents: List[str] = field(default_factory=list)
    mutations_applied: List[str] = field(default_factory=list)
    training_time: float = 0.0
    evaluation_metrics: Dict[str, float] = field(default_factory=dict)
    genetic_diversity: float = 0.0
    age: int = 0
    is_elite: bool = False


@dataclass
class Generation:
    """Represents a generation in evolutionary training."""
    number: int
    individuals: List[Individual]
    best_fitness: float
    average_fitness: float
    diversity_score: float
    created_at: str
    evaluation_time: float
    breeding_time: float
    selection_pressure: float


@dataclass
class EvolutionaryConfig:
    """Configuration for evolutionary training."""
    population_size: int = 50
    max_generations: int = 100
    mutation_rate: float = 0.1
    crossover_rate: float = 0.7
    selection_pressure: float = 1.2
    elitism_rate: float = 0.1
    tournament_size: int = 3
    diversity_threshold: float = 0.1
    fitness_convergence_threshold: float = 0.001
    early_stopping_patience: int = 10
    selection_strategy: SelectionStrategy = SelectionStrategy.TOURNAMENT
    crossover_strategy: CrossoverStrategy = CrossoverStrategy.UNIFORM
    mutation_strategy: MutationStrategy = MutationStrategy.ADAPTIVE


class FitnessEvaluator(ABC):
    """Abstract base class for fitness evaluation."""
    
    @abstractmethod
    async def evaluate(self, individual: Individual, model: Any, test_data: Any) -> float:
        """Evaluate fitness of an individual."""
        pass
    
    @abstractmethod
    def get_metrics(self, individual: Individual, model: Any, test_data: Any) -> Dict[str, float]:
        """Get detailed evaluation metrics."""
        pass


class MultiObjectiveFitnessEvaluator(FitnessEvaluator):
    """Multi-objective fitness evaluator balancing accuracy, speed, and size."""
    
    def __init__(self, weights: Dict[str, float] = None):
        self.weights = weights or {
            "accuracy": 0.6,
            "speed": 0.2,
            "size": 0.1,
            "robustness": 0.1
        }
        self.logger = logging.getLogger(__name__)
    
    async def evaluate(self, individual: Individual, model: Any, test_data: Any) -> float:
        """Evaluate multi-objective fitness."""
        metrics = self.get_metrics(individual, model, test_data)
        
        # Calculate weighted fitness score
        fitness = 0.0
        for metric, value in metrics.items():
            if metric in self.weights:
                fitness += self.weights[metric] * value
        
        individual.evaluation_metrics = metrics
        return fitness
    
    def get_metrics(self, individual: Individual, model: Any, test_data: Any) -> Dict[str, float]:
        """Get detailed evaluation metrics."""
        # Simulate evaluation metrics
        accuracy = random.uniform(0.7, 0.95)
        speed = random.uniform(0.5, 1.0)  # Normalized speed score
        size = random.uniform(0.6, 0.9)  # Normalized size efficiency
        robustness = random.uniform(0.6, 0.85)
        
        return {
            "accuracy": accuracy,
            "speed": speed,
            "size": size,
            "robustness": robustness
        }


class GeneticOperator(ABC):
    """Abstract base class for genetic operators."""
    
    @abstractmethod
    async def apply(self, *args, **kwargs) -> Any:
        """Apply genetic operation."""
        pass


class SelectionOperator(GeneticOperator):
    """Handles selection of individuals for breeding."""
    
    def __init__(self, strategy: SelectionStrategy, config: EvolutionaryConfig):
        self.strategy = strategy
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    async def apply(self, population: List[Individual], num_select: int) -> List[Individual]:
        """Select individuals for breeding."""
        if self.strategy == SelectionStrategy.TOURNAMENT:
            return await self._tournament_selection(population, num_select)
        elif self.strategy == SelectionStrategy.ROULETTE:
            return await self._roulette_selection(population, num_select)
        elif self.strategy == SelectionStrategy.RANK:
            return await self._rank_selection(population, num_select)
        elif self.strategy == SelectionStrategy.ELITE:
            return await self._elite_selection(population, num_select)
        else:
            raise ValueError(f"Unknown selection strategy: {self.strategy}")
    
    async def _tournament_selection(self, population: List[Individual], num_select: int) -> List[Individual]:
        """Tournament selection."""
        selected = []
        for _ in range(num_select):
            tournament = random.sample(population, min(self.config.tournament_size, len(population)))
            winner = max(tournament, key=lambda x: x.fitness)
            selected.append(winner)
        return selected
    
    async def _roulette_selection(self, population: List[Individual], num_select: int) -> List[Individual]:
        """Roulette wheel selection."""
        # Ensure all fitness values are positive
        min_fitness = min(ind.fitness for ind in population)
        if min_fitness < 0:
            fitness_values = [ind.fitness - min_fitness + 0.1 for ind in population]
        else:
            fitness_values = [ind.fitness for ind in population]
        
        total_fitness = sum(fitness_values)
        selected = []
        
        for _ in range(num_select):
            r = random.uniform(0, total_fitness)
            cumsum = 0
            for i, fitness in enumerate(fitness_values):
                cumsum += fitness
                if cumsum >= r:
                    selected.append(population[i])
                    break
        
        return selected
    
    async def _rank_selection(self, population: List[Individual], num_select: int) -> List[Individual]:
        """Rank-based selection."""
        sorted_population = sorted(population, key=lambda x: x.fitness, reverse=True)
        ranks = range(len(population), 0, -1)
        
        # Apply selection pressure
        adjusted_ranks = [rank ** self.config.selection_pressure for rank in ranks]
        total_rank = sum(adjusted_ranks)
        
        selected = []
        for _ in range(num_select):
            r = random.uniform(0, total_rank)
            cumsum = 0
            for i, rank in enumerate(adjusted_ranks):
                cumsum += rank
                if cumsum >= r:
                    selected.append(sorted_population[i])
                    break
        
        return selected
    
    async def _elite_selection(self, population: List[Individual], num_select: int) -> List[Individual]:
        """Elite selection (top performers)."""
        sorted_population = sorted(population, key=lambda x: x.fitness, reverse=True)
        return sorted_population[:num_select]


class CrossoverOperator(GeneticOperator):
    """Handles crossover (breeding) of individuals."""
    
    def __init__(self, strategy: CrossoverStrategy, config: EvolutionaryConfig):
        self.strategy = strategy
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    async def apply(self, parent1: Individual, parent2: Individual) -> Tuple[Individual, Individual]:
        """Create offspring from two parents."""
        if self.strategy == CrossoverStrategy.UNIFORM:
            return await self._uniform_crossover(parent1, parent2)
        elif self.strategy == CrossoverStrategy.SINGLE_POINT:
            return await self._single_point_crossover(parent1, parent2)
        elif self.strategy == CrossoverStrategy.TWO_POINT:
            return await self._two_point_crossover(parent1, parent2)
        elif self.strategy == CrossoverStrategy.ARITHMETIC:
            return await self._arithmetic_crossover(parent1, parent2)
        else:
            raise ValueError(f"Unknown crossover strategy: {self.strategy}")
    
    async def _uniform_crossover(self, parent1: Individual, parent2: Individual) -> Tuple[Individual, Individual]:
        """Uniform crossover."""
        # Create offspring by combining parent models
        offspring1_id = f"offspring_{parent1.id}_{parent2.id}_1"
        offspring2_id = f"offspring_{parent1.id}_{parent2.id}_2"
        
        offspring1 = Individual(
            id=offspring1_id,
            model_id=f"model_{offspring1_id}",
            generation=max(parent1.generation, parent2.generation) + 1,
            parents=[parent1.id, parent2.id]
        )
        
        offspring2 = Individual(
            id=offspring2_id,
            model_id=f"model_{offspring2_id}",
            generation=max(parent1.generation, parent2.generation) + 1,
            parents=[parent1.id, parent2.id]
        )
        
        return offspring1, offspring2
    
    async def _single_point_crossover(self, parent1: Individual, parent2: Individual) -> Tuple[Individual, Individual]:
        """Single-point crossover."""
        return await self._uniform_crossover(parent1, parent2)
    
    async def _two_point_crossover(self, parent1: Individual, parent2: Individual) -> Tuple[Individual, Individual]:
        """Two-point crossover."""
        return await self._uniform_crossover(parent1, parent2)
    
    async def _arithmetic_crossover(self, parent1: Individual, parent2: Individual) -> Tuple[Individual, Individual]:
        """Arithmetic crossover (parameter averaging)."""
        return await self._uniform_crossover(parent1, parent2)


class MutationOperator(GeneticOperator):
    """Handles mutation of individuals."""
    
    def __init__(self, strategy: MutationStrategy, config: EvolutionaryConfig):
        self.strategy = strategy
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    async def apply(self, individual: Individual, generation: int) -> Individual:
        """Apply mutation to an individual."""
        if random.random() > self.config.mutation_rate:
            return individual  # No mutation
        
        if self.strategy == MutationStrategy.GAUSSIAN:
            return await self._gaussian_mutation(individual, generation)
        elif self.strategy == MutationStrategy.UNIFORM:
            return await self._uniform_mutation(individual, generation)
        elif self.strategy == MutationStrategy.ADAPTIVE:
            return await self._adaptive_mutation(individual, generation)
        elif self.strategy == MutationStrategy.LAYER_WISE:
            return await self._layer_wise_mutation(individual, generation)
        else:
            raise ValueError(f"Unknown mutation strategy: {self.strategy}")
    
    async def _gaussian_mutation(self, individual: Individual, generation: int) -> Individual:
        """Gaussian mutation."""
        # Simulate parameter mutation
        mutation_strength = 0.1 * (1.0 - generation / 100.0)  # Decrease over time
        individual.mutations_applied.append(f"gaussian_{mutation_strength:.3f}")
        return individual
    
    async def _uniform_mutation(self, individual: Individual, generation: int) -> Individual:
        """Uniform mutation."""
        individual.mutations_applied.append("uniform")
        return individual
    
    async def _adaptive_mutation(self, individual: Individual, generation: int) -> Individual:
        """Adaptive mutation based on fitness and diversity."""
        # Adapt mutation rate based on individual performance
        if individual.fitness < 0.5:
            # Increase mutation for poor performers
            individual.mutations_applied.append("adaptive_high")
        else:
            individual.mutations_applied.append("adaptive_low")
        return individual
    
    async def _layer_wise_mutation(self, individual: Individual, generation: int) -> Individual:
        """Layer-wise mutation for neural networks."""
        individual.mutations_applied.append("layer_wise")
        return individual


class EvolutionaryTrainer:
    """Main evolutionary training engine."""
    
    def __init__(self, config: EvolutionaryConfig, fitness_evaluator: FitnessEvaluator):
        self.config = config
        self.fitness_evaluator = fitness_evaluator
        self.population: List[Individual] = []
        self.generations: List[Generation] = []
        self.current_generation = 0
        self.status = TrainingStatus.INITIALIZING
        self.best_individual: Optional[Individual] = None
        self.convergence_history: List[float] = []
        
        # Initialize genetic operators
        self.selection_operator = SelectionOperator(config.selection_strategy, config)
        self.crossover_operator = CrossoverOperator(config.crossover_strategy, config)
        self.mutation_operator = MutationOperator(config.mutation_strategy, config)
        
        self.logger = logging.getLogger(__name__)
    
    async def initialize_population(self, model_registry) -> None:
        """Initialize the population with base models."""
        self.logger.info(f"Initializing population of {self.config.population_size} individuals")
        
        for i in range(self.config.population_size):
            # Create base model
            model_id = await model_registry.create_base_model(
                name=f"individual_{i}",
                framework=model_registry.config.default_type,
                architecture="transformer",
                parameters=random.randint(100000, 1000000),
                size_mb=random.uniform(5.0, 50.0)
            )
            
            individual = Individual(
                id=f"individual_{i}",
                model_id=model_id,
                generation=0
            )
            
            self.population.append(individual)
        
        self.status = TrainingStatus.RUNNING
        self.logger.info("Population initialized successfully")
    
    async def evolve(self, model_registry, test_data: Any = None) -> Individual:
        """Run the evolutionary training process."""
        self.logger.info("Starting evolutionary training")
        
        start_time = time.time()
        stagnation_counter = 0
        
        try:
            for generation in range(self.config.max_generations):
                self.current_generation = generation
                generation_start = time.time()
                
                self.logger.info(f"Processing generation {generation + 1}/{self.config.max_generations}")
                
                # Evaluate population fitness
                await self._evaluate_population(model_registry, test_data)
                
                # Check for convergence
                if await self._check_convergence():
                    stagnation_counter += 1
                    if stagnation_counter >= self.config.early_stopping_patience:
                        self.logger.info(f"Early stopping at generation {generation + 1}")
                        break
                else:
                    stagnation_counter = 0
                
                # Create next generation
                if generation < self.config.max_generations - 1:
                    await self._create_next_generation(model_registry)
                
                # Record generation statistics
                generation_time = time.time() - generation_start
                await self._record_generation_stats(generation, generation_time)
                
                # Log progress
                best_fitness = max(ind.fitness for ind in self.population)
                avg_fitness = sum(ind.fitness for ind in self.population) / len(self.population)
                self.logger.info(
                    f"Generation {generation + 1}: Best={best_fitness:.4f}, "
                    f"Avg={avg_fitness:.4f}, Time={generation_time:.2f}s"
                )
            
            self.status = TrainingStatus.COMPLETED
            total_time = time.time() - start_time
            
            # Find best individual
            self.best_individual = max(self.population, key=lambda x: x.fitness)
            self.best_individual.is_elite = True
            
            self.logger.info(
                f"Evolution completed in {total_time:.2f}s. "
                f"Best fitness: {self.best_individual.fitness:.4f}"
            )
            
            return self.best_individual
            
        except Exception as e:
            self.status = TrainingStatus.FAILED
            self.logger.error(f"Evolution failed: {e}")
            raise
    
    async def _evaluate_population(self, model_registry, test_data: Any) -> None:
        """Evaluate fitness of all individuals in the population."""
        evaluation_tasks = []
        
        for individual in self.population:
            task = self._evaluate_individual(individual, model_registry, test_data)
            evaluation_tasks.append(task)
        
        await asyncio.gather(*evaluation_tasks)
    
    async def _evaluate_individual(self, individual: Individual, model_registry, test_data: Any) -> None:
        """Evaluate a single individual's fitness."""
        try:
            # Get model for evaluation
            model = await model_registry.get_model(individual.model_id)
            
            if model:
                start_time = time.time()
                fitness = await self.fitness_evaluator.evaluate(individual, model, test_data)
                individual.fitness = fitness
                individual.training_time = time.time() - start_time
                
                # Calculate genetic diversity (simplified)
                individual.genetic_diversity = random.uniform(0.1, 0.9)
                individual.age += 1
            else:
                individual.fitness = 0.0
                
        except Exception as e:
            self.logger.warning(f"Failed to evaluate individual {individual.id}: {e}")
            individual.fitness = 0.0
    
    async def _check_convergence(self) -> bool:
        """Check if the population has converged."""
        if len(self.convergence_history) < 2:
            return False
        
        recent_improvement = (
            self.convergence_history[-1] - self.convergence_history[-2]
        )
        
        return abs(recent_improvement) < self.config.fitness_convergence_threshold
    
    async def _create_next_generation(self, model_registry) -> None:
        """Create the next generation through selection, crossover, and mutation."""
        breeding_start = time.time()
        
        # Select elite individuals (preserve best performers)
        elite_count = int(self.config.population_size * self.config.elitism_rate)
        elite_individuals = await self.selection_operator.apply(self.population, elite_count)
        
        # Mark elite individuals
        for individual in elite_individuals:
            individual.is_elite = True
        
        new_population = elite_individuals.copy()
        
        # Generate offspring to fill remaining population
        offspring_needed = self.config.population_size - elite_count
        parents_pool = await self.selection_operator.apply(
            self.population, offspring_needed * 2
        )
        
        # Create offspring through crossover and mutation
        for i in range(0, len(parents_pool), 2):
            if i + 1 < len(parents_pool) and len(new_population) < self.config.population_size:
                parent1 = parents_pool[i]
                parent2 = parents_pool[i + 1]
                
                # Crossover
                if random.random() < self.config.crossover_rate:
                    offspring1, offspring2 = await self.crossover_operator.apply(parent1, parent2)
                    
                    # Register new models
                    offspring1.model_id = await self._create_offspring_model(
                        offspring1, [parent1, parent2], model_registry
                    )
                    offspring2.model_id = await self._create_offspring_model(
                        offspring2, [parent1, parent2], model_registry
                    )
                    
                    # Mutation
                    offspring1 = await self.mutation_operator.apply(offspring1, self.current_generation)
                    offspring2 = await self.mutation_operator.apply(offspring2, self.current_generation)
                    
                    new_population.extend([offspring1, offspring2])
        
        # Trim population to exact size
        self.population = new_population[:self.config.population_size]
        
        breeding_time = time.time() - breeding_start
        self.logger.debug(f"Breeding completed in {breeding_time:.2f}s")
    
    async def _create_offspring_model(
        self, 
        offspring: Individual, 
        parents: List[Individual], 
        model_registry
    ) -> str:
        """Create a new model for offspring by merging parent models."""
        parent_model_ids = [parent.model_id for parent in parents]
        
        # Merge parent models to create offspring model
        merged_model_id = await model_registry.merge_models(
            parent_model_ids, strategy="weighted_average"
        )
        
        return merged_model_id
    
    async def _record_generation_stats(self, generation: int, generation_time: float) -> None:
        """Record statistics for the current generation."""
        fitness_values = [ind.fitness for ind in self.population]
        best_fitness = max(fitness_values)
        avg_fitness = sum(fitness_values) / len(fitness_values)
        
        # Calculate diversity score
        diversity_score = np.std([ind.genetic_diversity for ind in self.population])
        
        generation_record = Generation(
            number=generation,
            individuals=copy.deepcopy(self.population),
            best_fitness=best_fitness,
            average_fitness=avg_fitness,
            diversity_score=diversity_score,
            created_at=datetime.now().isoformat(),
            evaluation_time=0.0,  # Would be calculated in real implementation
            breeding_time=0.0,   # Would be calculated in real implementation
            selection_pressure=self.config.selection_pressure
        )
        
        self.generations.append(generation_record)
        self.convergence_history.append(best_fitness)
    
    def get_training_history(self) -> Dict[str, Any]:
        """Get complete training history."""
        return {
            "config": asdict(self.config),
            "generations": [asdict(gen) for gen in self.generations],
            "convergence_history": self.convergence_history,
            "best_individual": asdict(self.best_individual) if self.best_individual else None,
            "status": self.status.value,
            "total_generations": len(self.generations)
        }


class TrainingManager:
    """
    Central training management system for Symbio AI.
    
    Coordinates evolutionary training, model optimization,
    and performance tracking across the entire system.
    """
    
    def __init__(self, config):
        self.config = config
        self.active_trainers: Dict[str, EvolutionaryTrainer] = {}
        self.training_history: Dict[str, Dict[str, Any]] = {}
        self.logger = logging.getLogger(__name__)
    
    async def initialize(self) -> None:
        """Initialize the training management system."""
        self.logger.info("Initializing training management system")
        
        # Create default fitness evaluator
        self.default_fitness_evaluator = MultiObjectiveFitnessEvaluator()
        
        self.logger.info("Training manager initialized")
    
    async def start_evolutionary_training(
        self,
        training_id: str,
        model_registry,
        config: Optional[EvolutionaryConfig] = None,
        fitness_evaluator: Optional[FitnessEvaluator] = None
    ) -> str:
        """
        Start a new evolutionary training session.
        
        Args:
            training_id: Unique identifier for this training session
            model_registry: Model registry instance
            config: Training configuration (uses default if None)
            fitness_evaluator: Custom fitness evaluator (uses default if None)
            
        Returns:
            Training session ID
        """
        if config is None:
            config = EvolutionaryConfig(
                population_size=self.config.population_size,
                max_generations=self.config.generations,
                mutation_rate=self.config.mutation_rate,
                crossover_rate=self.config.crossover_rate,
                selection_pressure=self.config.selection_pressure
            )
        
        if fitness_evaluator is None:
            fitness_evaluator = self.default_fitness_evaluator
        
        trainer = EvolutionaryTrainer(config, fitness_evaluator)
        self.active_trainers[training_id] = trainer
        
        self.logger.info(f"Starting evolutionary training session: {training_id}")
        
        # Initialize population and start training
        await trainer.initialize_population(model_registry)
        
        # Run evolution in background
        asyncio.create_task(self._run_training_session(training_id, trainer, model_registry))
        
        return training_id
    
    async def _run_training_session(
        self,
        training_id: str,
        trainer: EvolutionaryTrainer,
        model_registry
    ) -> None:
        """Run a training session to completion."""
        try:
            best_individual = await trainer.evolve(model_registry)
            
            # Store training history
            self.training_history[training_id] = trainer.get_training_history()
            
            self.logger.info(
                f"Training session {training_id} completed. "
                f"Best fitness: {best_individual.fitness:.4f}"
            )
            
        except Exception as e:
            self.logger.error(f"Training session {training_id} failed: {e}")
            trainer.status = TrainingStatus.FAILED
        
        finally:
            # Clean up
            if training_id in self.active_trainers:
                del self.active_trainers[training_id]
    
    async def get_training_status(self, training_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a training session."""
        if training_id in self.active_trainers:
            trainer = self.active_trainers[training_id]
            return {
                "status": trainer.status.value,
                "current_generation": trainer.current_generation,
                "max_generations": trainer.config.max_generations,
                "population_size": len(trainer.population),
                "best_fitness": max(ind.fitness for ind in trainer.population) if trainer.population else 0.0,
                "average_fitness": sum(ind.fitness for ind in trainer.population) / len(trainer.population) if trainer.population else 0.0
            }
        elif training_id in self.training_history:
            history = self.training_history[training_id]
            return {
                "status": history["status"],
                "completed": True,
                "total_generations": history["total_generations"],
                "best_fitness": history["best_individual"]["fitness"] if history["best_individual"] else 0.0
            }
        else:
            return None
    
    async def stop_training(self, training_id: str) -> bool:
        """Stop a training session."""
        if training_id in self.active_trainers:
            trainer = self.active_trainers[training_id]
            trainer.status = TrainingStatus.STOPPED
            self.logger.info(f"Stopped training session: {training_id}")
            return True
        return False
    
    async def pause_training(self, training_id: str) -> bool:
        """Pause a training session."""
        if training_id in self.active_trainers:
            trainer = self.active_trainers[training_id]
            trainer.status = TrainingStatus.PAUSED
            self.logger.info(f"Paused training session: {training_id}")
            return True
        return False
    
    async def resume_training(self, training_id: str) -> bool:
        """Resume a paused training session."""
        if training_id in self.active_trainers:
            trainer = self.active_trainers[training_id]
            if trainer.status == TrainingStatus.PAUSED:
                trainer.status = TrainingStatus.RUNNING
                self.logger.info(f"Resumed training session: {training_id}")
                return True
        return False
    
    def list_training_sessions(self) -> Dict[str, Dict[str, Any]]:
        """List all training sessions (active and completed)."""
        sessions = {}
        
        # Active sessions
        for training_id, trainer in self.active_trainers.items():
            sessions[training_id] = {
                "status": trainer.status.value,
                "active": True,
                "current_generation": trainer.current_generation,
                "max_generations": trainer.config.max_generations
            }
        
        # Completed sessions
        for training_id, history in self.training_history.items():
            if training_id not in sessions:  # Don't overwrite active sessions
                sessions[training_id] = {
                    "status": history["status"],
                    "active": False,
                    "total_generations": history["total_generations"]
                }
        
        return sessions
    
    async def get_training_history(self, training_id: str) -> Optional[Dict[str, Any]]:
        """Get complete training history for a session."""
        return self.training_history.get(training_id)
    
    async def cleanup(self) -> None:
        """Clean up training manager resources."""
        # Stop all active training sessions
        for training_id in list(self.active_trainers.keys()):
            await self.stop_training(training_id)
        
        self.logger.info("Training manager cleanup completed")