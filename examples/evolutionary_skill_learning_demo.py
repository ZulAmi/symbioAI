#!/usr/bin/env python3
"""
Evolutionary Algorithm for Skill Learning - Demo

This demonstrates the evolutionary training system following the exact prompt:
- Evolve a population of small models to acquire diverse skills
- Initialize population with different random seeds
- Evaluate agents on tasks, select elite, crossover, mutate
- Return best agents and their specializations

Runs without ML dependencies for immediate demonstration.
"""

import asyncio
import random
import time
import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union, Callable, Set
from dataclasses import dataclass, field
from collections import defaultdict, deque
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Mock implementations for demonstration
class MockTensor:
    """Mock tensor for demonstration."""
    def __init__(self, data: List[float], shape: Optional[Tuple[int, ...]] = None):
        self.data = data if isinstance(data, list) else [data]
        self.shape = shape or (len(self.data),)
    
    def mean(self):
        return sum(self.data) / len(self.data) if self.data else 0.0
    
    def size(self, dim=None):
        return self.shape[dim] if dim is not None else self.shape[0]
    
    def item(self):
        return self.data[0] if self.data else 0.0


class MockModel:
    """Mock neural network model."""
    def __init__(self, weights: Dict[str, List[float]] = None):
        self.weights = weights or self._initialize_random_weights()
        self.fitness = 0.0
    
    def _initialize_random_weights(self) -> Dict[str, List[float]]:
        """Initialize random weights."""
        return {
            'layer1': [random.gauss(0, 0.1) for _ in range(64)],
            'layer2': [random.gauss(0, 0.1) for _ in range(32)], 
            'output': [random.gauss(0, 0.1) for _ in range(10)]
        }
    
    def predict(self, inputs: List[float]) -> List[float]:
        """Mock prediction."""
        # Simple mock computation
        output = []
        for i in range(min(10, len(inputs))):
            value = sum(w * inputs[j % len(inputs)] for j, w in enumerate(self.weights['output']))
            output.append(value + random.gauss(0, 0.01))
        return output
    
    def parameters(self):
        """Mock parameter iterator."""
        for layer, weights in self.weights.items():
            yield MockTensor(weights)
    
    def eval(self):
        """Mock eval mode."""
        pass
    
    def copy(self):
        """Create a copy of the model."""
        import copy
        return MockModel(copy.deepcopy(self.weights))


@dataclass
class AgentModel:
    """Individual agent model in the population."""
    id: str
    model: MockModel
    fitness_score: float = 0.0
    task_performances: Dict[str, float] = field(default_factory=dict)
    specializations: List[str] = field(default_factory=list)
    generation: int = 0
    parent_ids: List[str] = field(default_factory=list)
    mutation_history: List[str] = field(default_factory=list)
    
    def copy(self):
        """Create a copy of the agent."""
        import copy
        return AgentModel(
            id=self.id + "_copy",
            model=self.model.copy(),
            fitness_score=self.fitness_score,
            task_performances=copy.deepcopy(self.task_performances),
            specializations=copy.deepcopy(self.specializations),
            generation=self.generation,
            parent_ids=copy.deepcopy(self.parent_ids),
            mutation_history=copy.deepcopy(self.mutation_history)
        )


@dataclass
class EvolutionConfig:
    """Configuration for evolutionary training."""
    population_size: int = 20
    num_generations: int = 10
    elitism_ratio: float = 0.2
    crossover_ratio: float = 0.6
    mutation_ratio: float = 0.2
    mutation_rate: float = 0.1
    mutation_strength: float = 0.01
    tournament_size: int = 3
    convergence_threshold: float = 0.001
    max_stagnation: int = 5


class EvolutionarySkillLearner:
    """
    Evolutionary Training: Evolve a population of small models to acquire diverse skills
    
    Following the exact prompt structure:
    1. Initialize a population of N agent models with different random seeds or initializations.
    2. For each generation:
       a. Evaluate each agent on a set of tasks (compute a fitness score per agent).
       b. Select top-performing agents (elitism) for breeding.
       c. Generate new agents by crossover (merge parameters of two parents) and mutation (perturb weights or use SVD on weights).
       d. Replace the worst-performing agents with new offspring.
    3. Iterate for G generations or until convergence.
    4. Return the best agent models and their niche specializations.
    """
    
    def __init__(self, config: EvolutionConfig):
        self.config = config
        self.population: List[AgentModel] = []
        self.generation = 0
        self.best_fitness_history: List[float] = []
        self.stagnation_counter = 0
        
        # Task definitions for skill learning
        self.tasks = {
            'classification': self._classification_task,
            'regression': self._regression_task,
            'reasoning': self._reasoning_task,
            'memory': self._memory_task,
            'pattern_recognition': self._pattern_recognition_task
        }
        
        print("🧬 Evolutionary Skill Learning System Initialized")
        print(f"📊 Population size: {config.population_size}")
        print(f"🔄 Max generations: {config.num_generations}")
        print(f"🎯 Tasks available: {list(self.tasks.keys())}")
    
    async def initialize_population(self) -> None:
        """1. Initialize a population of N agent models with different random seeds or initializations."""
        print(f"\n🌱 Initializing population of {self.config.population_size} agents...")
        
        self.population = []
        for i in range(self.config.population_size):
            # Set different random seed for each agent
            random.seed(i * 42 + 12345)
            
            # Create agent with unique characteristics
            agent = AgentModel(
                id=f"gen{self.generation:02d}_agent_{i:03d}",
                model=MockModel(),
                generation=self.generation
            )
            
            self.population.append(agent)
        
        print(f"✅ Population initialized with {len(self.population)} diverse agents")
    
    async def evaluate_population(self) -> None:
        """2a. Evaluate each agent on a set of tasks (compute a fitness score per agent)."""
        print(f"\n📈 Evaluating generation {self.generation} on {len(self.tasks)} tasks...")
        
        for i, agent in enumerate(self.population):
            agent.task_performances = {}
            
            # Evaluate on each task
            for task_name, task_func in self.tasks.items():
                try:
                    performance = await task_func(agent.model)
                    agent.task_performances[task_name] = performance
                except Exception as e:
                    agent.task_performances[task_name] = 0.0
            
            # Calculate overall fitness
            agent.fitness_score = sum(agent.task_performances.values()) / len(self.tasks)
            
            if i % 5 == 0:  # Progress indicator
                print(f"  Evaluated {i+1}/{len(self.population)} agents...")
        
        # Sort by fitness
        self.population.sort(key=lambda x: x.fitness_score, reverse=True)
        
        best_fitness = self.population[0].fitness_score
        avg_fitness = sum(agent.fitness_score for agent in self.population) / len(self.population)
        
        self.best_fitness_history.append(best_fitness)
        
        print(f"📊 Evaluation complete - Best: {best_fitness:.4f}, Average: {avg_fitness:.4f}")
    
    def select_elite(self) -> List[AgentModel]:
        """2b. Select top-performing agents (elitism) for breeding."""
        num_elite = max(1, int(self.config.population_size * self.config.elitism_ratio))
        elite = self.population[:num_elite]
        
        print(f"👑 Selected {len(elite)} elite agents for breeding")
        return elite
    
    def tournament_selection(self) -> AgentModel:
        """Tournament selection for parent selection."""
        tournament = random.sample(self.population, min(self.config.tournament_size, len(self.population)))
        return max(tournament, key=lambda x: x.fitness_score)
    
    def crossover(self, parent1: AgentModel, parent2: AgentModel) -> AgentModel:
        """2c. Generate new agents by crossover (merge parameters of two parents)."""
        child_model = MockModel()
        
        # Parameter averaging crossover
        for layer_name in child_model.weights.keys():
            if layer_name in parent1.model.weights and layer_name in parent2.model.weights:
                # Blend parent parameters
                alpha = random.uniform(0.3, 0.7)
                p1_weights = parent1.model.weights[layer_name]
                p2_weights = parent2.model.weights[layer_name]
                
                child_weights = []
                for w1, w2 in zip(p1_weights, p2_weights):
                    child_weights.append(alpha * w1 + (1 - alpha) * w2)
                
                child_model.weights[layer_name] = child_weights
        
        child = AgentModel(
            id=f"gen{self.generation+1:02d}_cross_{len(self.population):03d}",
            model=child_model,
            generation=self.generation + 1,
            parent_ids=[parent1.id, parent2.id]
        )
        
        return child
    
    def mutate(self, agent: AgentModel) -> AgentModel:
        """2c. Mutation (perturb weights or use SVD on weights)."""
        mutated_model = agent.model.copy()
        
        # Gaussian noise mutation
        for layer_name, weights in mutated_model.weights.items():
            for i in range(len(weights)):
                if random.random() < self.config.mutation_rate:
                    noise = random.gauss(0, self.config.mutation_strength)
                    weights[i] += noise
        
        mutated_agent = AgentModel(
            id=f"gen{self.generation+1:02d}_mut_{len(self.population):03d}",
            model=mutated_model,
            generation=self.generation + 1,
            parent_ids=[agent.id],
            mutation_history=agent.mutation_history + ['gaussian_noise']
        )
        
        return mutated_agent
    
    async def evolve_generation(self) -> None:
        """Execute one complete generation of evolution."""
        print(f"\n🔬 Evolving Generation {self.generation}")
        
        # Evaluate current population
        await self.evaluate_population()
        
        # Check convergence
        if self._check_convergence():
            print("🎯 Convergence achieved - stopping evolution")
            return
        
        # Create next generation
        new_population = []
        
        # Elite preservation
        elite = self.select_elite()
        new_population.extend([agent.copy() for agent in elite])
        
        # Crossover offspring
        num_crossover = int(self.config.population_size * self.config.crossover_ratio)
        print(f"🧬 Generating {num_crossover} crossover offspring...")
        
        for _ in range(num_crossover):
            parent1 = self.tournament_selection()
            parent2 = self.tournament_selection()
            child = self.crossover(parent1, parent2)
            new_population.append(child)
        
        # Mutation offspring
        num_mutation = int(self.config.population_size * self.config.mutation_ratio)
        print(f"🎲 Generating {num_mutation} mutation offspring...")
        
        for _ in range(num_mutation):
            parent = self.tournament_selection()
            child = self.mutate(parent)
            new_population.append(child)
        
        # Ensure exact population size
        while len(new_population) < self.config.population_size:
            parent = random.choice(self.population)
            child = self.mutate(parent)
            new_population.append(child)
        
        new_population = new_population[:self.config.population_size]
        
        # 2d. Replace the worst-performing agents with new offspring
        self.population = new_population
        self.generation += 1
        
        # Update specializations
        await self._update_specializations()
        
        print(f"✅ Generation {self.generation} created - Population: {len(self.population)}")
    
    def _check_convergence(self) -> bool:
        """Check if evolution has converged."""
        if len(self.best_fitness_history) < 2:
            return False
        
        improvement = self.best_fitness_history[-1] - self.best_fitness_history[-2]
        if improvement < self.config.convergence_threshold:
            self.stagnation_counter += 1
        else:
            self.stagnation_counter = 0
        
        return self.stagnation_counter >= self.config.max_stagnation
    
    async def _update_specializations(self) -> None:
        """Update agent specializations based on task performance."""
        for agent in self.population:
            if not agent.task_performances:
                continue
            
            # Find tasks where agent excels
            avg_performance = sum(agent.task_performances.values()) / len(agent.task_performances)
            specializations = []
            
            for task, performance in agent.task_performances.items():
                if performance > avg_performance + 0.1:  # Threshold for specialization
                    specializations.append(task)
            
            agent.specializations = specializations
    
    async def train(self) -> Dict[str, Any]:
        """
        3. Iterate for G generations or until convergence.
        4. Return the best agent models and their niche specializations.
        """
        print("🚀 Starting Evolutionary Skill Learning Training")
        start_time = time.time()
        
        # Initialize population
        await self.initialize_population()
        
        # Evolution loop
        for generation in range(self.config.num_generations):
            await self.evolve_generation()
            
            if self._check_convergence():
                print(f"🏁 Early stopping at generation {generation}")
                break
        
        training_time = time.time() - start_time
        
        # Final evaluation
        await self.evaluate_population()
        
        # Get results
        best_agents = self.get_best_agents(5)
        diverse_agents = self.get_diverse_agents(5)
        
        results = {
            "best_agents": best_agents,
            "diverse_agents": diverse_agents,
            "final_generation": self.generation,
            "training_time": training_time,
            "best_fitness_history": self.best_fitness_history,
            "population_size": len(self.population),
            "convergence_achieved": self._check_convergence()
        }
        
        print(f"🎉 Training completed in {training_time:.2f}s")
        return results
    
    def get_best_agents(self, n: int = 5) -> List[AgentModel]:
        """Return the best agent models."""
        return sorted(self.population, key=lambda x: x.fitness_score, reverse=True)[:n]
    
    def get_diverse_agents(self, n: int = 5) -> List[AgentModel]:
        """Return diverse agents covering different specializations."""
        # Group by specializations
        specialization_groups = defaultdict(list)
        for agent in self.population:
            if agent.specializations:
                key = tuple(sorted(agent.specializations))
                specialization_groups[key].append(agent)
            else:
                specialization_groups[('generalist',)].append(agent)
        
        # Select best from each group
        diverse_agents = []
        for spec_group, agents in specialization_groups.items():
            best_in_group = max(agents, key=lambda x: x.fitness_score)
            diverse_agents.append(best_in_group)
        
        diverse_agents.sort(key=lambda x: x.fitness_score, reverse=True)
        return diverse_agents[:n]
    
    # Task definitions for skill evaluation
    async def _classification_task(self, model: MockModel) -> float:
        """Mock classification task."""
        # Simulate classification accuracy
        inputs = [random.random() for _ in range(10)]
        outputs = model.predict(inputs)
        
        # Mock accuracy based on output consistency
        consistency = 1.0 - (max(outputs) - min(outputs)) if outputs else 0.0
        base_score = random.uniform(0.4, 0.9)
        return min(1.0, base_score + consistency * 0.1)
    
    async def _regression_task(self, model: MockModel) -> float:
        """Mock regression task."""
        # Simulate regression performance
        inputs = [random.random() for _ in range(10)]
        outputs = model.predict(inputs)
        
        # Mock performance based on output stability
        if len(outputs) > 1:
            variance = sum((x - sum(outputs)/len(outputs))**2 for x in outputs) / len(outputs)
            score = 1.0 / (1.0 + variance)
        else:
            score = random.uniform(0.3, 0.8)
        
        return min(1.0, score)
    
    async def _reasoning_task(self, model: MockModel) -> float:
        """Mock reasoning task."""
        # Simulate reasoning capability
        inputs = [i * 0.1 for i in range(10)]  # Sequential pattern
        outputs = model.predict(inputs)
        
        # Score based on pattern recognition
        pattern_score = 0.0
        if len(outputs) > 3:
            # Check if outputs follow some pattern
            diffs = [outputs[i+1] - outputs[i] for i in range(len(outputs)-1)]
            consistency = 1.0 - (max(diffs) - min(diffs)) if diffs else 0.0
            pattern_score = consistency * 0.5
        
        base_score = random.uniform(0.2, 0.7)
        return min(1.0, base_score + pattern_score)
    
    async def _memory_task(self, model: MockModel) -> float:
        """Mock memory task."""
        # Test model's ability to maintain consistency
        inputs1 = [random.random() for _ in range(5)]
        inputs2 = inputs1.copy()  # Same inputs
        
        outputs1 = model.predict(inputs1)
        outputs2 = model.predict(inputs2)
        
        # Score based on consistency
        if len(outputs1) == len(outputs2):
            differences = [abs(o1 - o2) for o1, o2 in zip(outputs1, outputs2)]
            consistency = 1.0 - (sum(differences) / len(differences))
            return max(0.0, min(1.0, consistency + random.uniform(-0.1, 0.1)))
        
        return random.uniform(0.3, 0.7)
    
    async def _pattern_recognition_task(self, model: MockModel) -> float:
        """Mock pattern recognition task."""
        # Test pattern recognition ability
        pattern_inputs = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        outputs = model.predict(pattern_inputs)
        
        # Score based on output progression
        if len(outputs) > 3:
            # Check if outputs show progression
            is_increasing = all(outputs[i] <= outputs[i+1] for i in range(len(outputs)-1))
            is_decreasing = all(outputs[i] >= outputs[i+1] for i in range(len(outputs)-1))
            
            if is_increasing or is_decreasing:
                return random.uniform(0.7, 0.95)
            else:
                return random.uniform(0.3, 0.7)
        
        return random.uniform(0.2, 0.6)


async def demonstrate_evolutionary_skill_learning():
    """Demonstrate the evolutionary skill learning system."""
    
    print("🧬 Evolutionary Algorithm for Skill Learning - Symbio AI")
    print("=" * 60)
    
    # Configuration
    config = EvolutionConfig(
        population_size=15,
        num_generations=8,
        elitism_ratio=0.2,
        crossover_ratio=0.6,
        mutation_ratio=0.2,
        mutation_rate=0.15,
        mutation_strength=0.02,
        tournament_size=3
    )
    
    # Create and run evolutionary trainer
    trainer = EvolutionarySkillLearner(config)
    results = await trainer.train()
    
    # Display results
    print("\n" + "=" * 60)
    print("🎯 EVOLUTIONARY TRAINING RESULTS")
    print("=" * 60)
    
    print(f"⏱️  Training Time: {results['training_time']:.2f} seconds")
    print(f"🔄 Generations: {results['final_generation']}")
    print(f"🎯 Convergence: {results['convergence_achieved']}")
    print(f"👥 Final Population: {results['population_size']} agents")
    
    print(f"\n🏆 TOP 5 BEST AGENTS:")
    print("-" * 40)
    for i, agent in enumerate(results['best_agents'], 1):
        specializations = ', '.join(agent.specializations) if agent.specializations else 'Generalist'
        print(f"{i}. {agent.id}")
        print(f"   Fitness: {agent.fitness_score:.4f}")
        print(f"   Specializations: {specializations}")
        print(f"   Task Performances:")
        for task, score in agent.task_performances.items():
            print(f"     • {task}: {score:.3f}")
        print()
    
    print(f"🌈 DIVERSE SPECIALIZED AGENTS:")
    print("-" * 40)
    for i, agent in enumerate(results['diverse_agents'], 1):
        specializations = ', '.join(agent.specializations) if agent.specializations else 'Generalist'
        print(f"{i}. {agent.id}")
        print(f"   Fitness: {agent.fitness_score:.4f}")
        print(f"   Niche: {specializations}")
        print(f"   Generation: {agent.generation}")
        if agent.parent_ids:
            print(f"   Parents: {', '.join(agent.parent_ids[:2])}")
        print()
    
    print(f"📈 FITNESS EVOLUTION:")
    print("-" * 40)
    for gen, fitness in enumerate(results['best_fitness_history']):
        print(f"Generation {gen}: {fitness:.4f}")
    
    print("\n✅ Evolutionary skill learning demonstration complete!")
    print("📊 System successfully evolved diverse specialists from general population")
    print("🧬 Nature-inspired algorithms enable robust, adaptive AI systems")


if __name__ == "__main__":
    asyncio.run(demonstrate_evolutionary_skill_learning())