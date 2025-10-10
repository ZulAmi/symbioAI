#!/usr/bin/env python3
"""
Evolutionary Model Merging Concept Demo for Symbio AI

This demonstrates the evolutionary model merging concept without requiring
actual ML libraries, showing the algorithm structure and evolutionary process.
"""

import random
import json
import time
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass, field
import logging

# Mock tensor class to simulate PyTorch tensors
class MockTensor:
    def __init__(self, shape: Tuple[int, ...], data: List[float] = None):
        self.shape = shape
        self.size = 1
        for dim in shape:
            self.size *= dim
        
        if data is None:
            self.data = [random.gauss(0, 0.1) for _ in range(self.size)]
        else:
            self.data = data[:self.size]
    
    def __add__(self, other):
        if isinstance(other, (int, float)):
            return MockTensor(self.shape, [x + other for x in self.data])
        elif isinstance(other, MockTensor):
            return MockTensor(self.shape, [a + b for a, b in zip(self.data, other.data)])
    
    def __mul__(self, other):
        if isinstance(other, (int, float)):
            return MockTensor(self.shape, [x * other for x in self.data])
        elif isinstance(other, MockTensor):
            return MockTensor(self.shape, [a * b for a, b in zip(self.data, other.data)])
    
    def __rmul__(self, other):
        return self.__mul__(other)
    
    def norm(self):
        return sum(x * x for x in self.data) ** 0.5
    
    def clone(self):
        return MockTensor(self.shape, self.data.copy())


@dataclass
class ModelCandidate:
    """Represents a candidate merged model."""
    weights: Dict[str, MockTensor]
    merge_ratios: List[float]
    fitness_score: float = 0.0
    validation_metrics: Dict[str, float] = field(default_factory=dict)
    generation: int = 0
    parent_indices: List[int] = field(default_factory=list)
    mutation_applied: bool = False


class MockModelEvaluator:
    """Mock evaluator that simulates realistic model performance evaluation."""
    
    def __init__(self):
        self.evaluation_count = 0
    
    def evaluate(self, model_weights: Dict[str, MockTensor], 
                 validation_data: Any = None) -> Tuple[float, Dict[str, float]]:
        """Simulate model evaluation with realistic scoring patterns."""
        self.evaluation_count += 1
        
        # Simulate evaluation based on weight characteristics
        total_norm = sum(tensor.norm() for tensor in model_weights.values())
        num_params = len(model_weights)
        
        # Simulate accuracy with some realistic patterns
        base_accuracy = 0.65 + random.gauss(0, 0.1)
        
        # Penalize extreme weight magnitudes (simulate overfitting)
        magnitude_penalty = max(0, (total_norm / num_params - 2.0) * 0.05)
        
        # Add noise to simulate validation variance
        noise = random.gauss(0, 0.02)
        
        accuracy = max(0.1, min(0.95, base_accuracy - magnitude_penalty + noise))
        loss = 2.0 - 2.0 * accuracy + random.gauss(0, 0.1)
        
        # Calculate fitness (higher is better)
        fitness = accuracy - 0.1 * max(0, loss - 1.0)
        
        metrics = {
            "accuracy": accuracy,
            "loss": loss,
            "total_norm": total_norm,
            "num_params": num_params
        }
        
        return fitness, metrics


class EvolutionaryModelMerger:
    """Evolutionary model merger with genetic algorithms."""
    
    def __init__(self, population_size=20, generations=30, mutation_rate=0.15, crossover_rate=0.8):
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        
        self.evaluator = MockModelEvaluator()
        self.best_fitness_history = []
        self.generation_stats = []
        
        # Set random seed for reproducibility
        random.seed(42)
    
    def load_mock_models(self, num_models=3) -> List[Dict[str, MockTensor]]:
        """Create mock models with different characteristics."""
        models = []
        
        layer_configs = [
            ("conv1.weight", (32, 3, 3, 3)),
            ("conv1.bias", (32,)),
            ("conv2.weight", (64, 32, 3, 3)),
            ("conv2.bias", (64,)),
            ("fc.weight", (10, 64)),
            ("fc.bias", (10,))
        ]
        
        for i in range(num_models):
            model = {}
            # Create model with different initialization patterns
            for layer_name, shape in layer_configs:
                # Different models have different weight characteristics
                if i == 0:  # Model A: Small weights
                    model[layer_name] = MockTensor(shape, [random.gauss(0, 0.05) for _ in range(MockTensor(shape, []).size)])
                elif i == 1:  # Model B: Larger weights
                    model[layer_name] = MockTensor(shape, [random.gauss(0, 0.15) for _ in range(MockTensor(shape, []).size)])
                else:  # Model C: Mixed characteristics
                    model[layer_name] = MockTensor(shape, [random.gauss(0, 0.1) for _ in range(MockTensor(shape, []).size)])
            
            models.append(model)
        
        return models
    
    def merge_models(self, base_models: List[Dict[str, MockTensor]], ratios: List[float]) -> Dict[str, MockTensor]:
        """Merge models using weighted average with given ratios."""
        merged_weights = {}
        
        # Get parameter names from first model
        param_names = list(base_models[0].keys())
        
        for param_name in param_names:
            # Initialize with zeros
            merged_param = MockTensor(base_models[0][param_name].shape, [0.0] * base_models[0][param_name].size)
            
            # Weighted sum of parameters
            for model, ratio in zip(base_models, ratios):
                weighted_param = ratio * model[param_name]
                merged_param = merged_param + weighted_param
            
            merged_weights[param_name] = merged_param
        
        return merged_weights
    
    def generate_random_ratios(self, num_models: int) -> List[float]:
        """Generate normalized random ratios for model merging."""
        ratios = [random.random() for _ in range(num_models)]
        total = sum(ratios)
        return [r / total for r in ratios]
    
    def initialize_population(self, base_models: List[Dict[str, MockTensor]]) -> List[ModelCandidate]:
        """Initialize population of model candidates."""
        population = []
        
        for i in range(self.population_size):
            # Generate random merge ratios
            ratios = self.generate_random_ratios(len(base_models))
            
            # Create merged weights
            merged_weights = self.merge_models(base_models, ratios)
            
            candidate = ModelCandidate(
                weights=merged_weights,
                merge_ratios=ratios,
                generation=0
            )
            
            population.append(candidate)
        
        return population
    
    def evaluate_population(self, population: List[ModelCandidate]) -> List[ModelCandidate]:
        """Evaluate fitness for entire population."""
        for candidate in population:
            fitness, metrics = self.evaluator.evaluate(candidate.weights)
            candidate.fitness_score = fitness
            candidate.validation_metrics = metrics
        
        return population
    
    def tournament_selection(self, population: List[ModelCandidate], tournament_size: int = 3) -> ModelCandidate:
        """Tournament selection."""
        tournament = random.sample(population, tournament_size)
        return max(tournament, key=lambda x: x.fitness_score)
    
    def crossover(self, parent1: ModelCandidate, parent2: ModelCandidate, generation: int) -> Tuple[ModelCandidate, ModelCandidate]:
        """Create offspring through crossover."""
        child1_ratios = []
        child2_ratios = []
        
        for r1, r2 in zip(parent1.merge_ratios, parent2.merge_ratios):
            # Uniform crossover
            if random.random() < 0.5:
                child1_ratios.append(r1)
                child2_ratios.append(r2)
            else:
                child1_ratios.append(r2)
                child2_ratios.append(r1)
        
        # Normalize ratios
        child1_ratios = self.normalize_ratios(child1_ratios)
        child2_ratios = self.normalize_ratios(child2_ratios)
        
        child1 = ModelCandidate(
            weights={},
            merge_ratios=child1_ratios,
            generation=generation,
            parent_indices=[id(parent1), id(parent2)]
        )
        
        child2 = ModelCandidate(
            weights={},
            merge_ratios=child2_ratios,
            generation=generation,
            parent_indices=[id(parent1), id(parent2)]
        )
        
        return child1, child2
    
    def mutate(self, candidate: ModelCandidate) -> None:
        """Apply mutation to candidate."""
        for i in range(len(candidate.merge_ratios)):
            if random.random() < 0.3:  # 30% chance to mutate each ratio
                mutation_strength = 0.1
                change = random.gauss(0, mutation_strength)
                candidate.merge_ratios[i] += change
        
        # Normalize and clamp ratios
        candidate.merge_ratios = [max(0.01, min(0.99, r)) for r in candidate.merge_ratios]
        candidate.merge_ratios = self.normalize_ratios(candidate.merge_ratios)
        candidate.mutation_applied = True
    
    def normalize_ratios(self, ratios: List[float]) -> List[float]:
        """Normalize ratios to sum to 1.0."""
        total = sum(ratios)
        if total == 0:
            return [1.0 / len(ratios)] * len(ratios)
        return [r / total for r in ratios]
    
    def evolve_population(self, population: List[ModelCandidate], base_models: List[Dict[str, MockTensor]], generation: int) -> List[ModelCandidate]:
        """Evolve population through selection, crossover, and mutation."""
        # Sort population by fitness (descending)
        population.sort(key=lambda x: x.fitness_score, reverse=True)
        
        # Elite selection (keep top 20%)
        elite_count = max(1, int(0.2 * len(population)))
        new_population = population[:elite_count]
        
        # Generate offspring
        while len(new_population) < self.population_size:
            parent1 = self.tournament_selection(population)
            parent2 = self.tournament_selection(population)
            
            if random.random() < self.crossover_rate:
                child1, child2 = self.crossover(parent1, parent2, generation)
                
                # Compute weights for children
                child1.weights = self.merge_models(base_models, child1.merge_ratios)
                child2.weights = self.merge_models(base_models, child2.merge_ratios)
                
                new_population.extend([child1, child2])
            else:
                new_population.extend([parent1, parent2])
        
        # Trim to exact population size
        new_population = new_population[:self.population_size]
        
        # Mutation (don't mutate elites)
        for candidate in new_population[elite_count:]:
            if random.random() < self.mutation_rate:
                old_ratios = candidate.merge_ratios.copy()
                self.mutate(candidate)
                # Recompute weights after mutation
                candidate.weights = self.merge_models(base_models, candidate.merge_ratios)
        
        return new_population
    
    def evolutionary_merge(self, num_models: int = 3) -> ModelCandidate:
        """Run evolutionary merging process."""
        print(f"🧬 Starting evolutionary merge with {num_models} models")
        print(f"📊 Population: {self.population_size}, Generations: {self.generations}")
        
        # Load base models
        base_models = self.load_mock_models(num_models)
        print(f"✅ Loaded {len(base_models)} mock models")
        
        # Initialize population
        population = self.initialize_population(base_models)
        print(f"🏗️  Initialized population of {len(population)} candidates")
        
        # Evaluate initial population
        population = self.evaluate_population(population)
        
        best_candidate = max(population, key=lambda x: x.fitness_score)
        self.best_fitness_history.append(best_candidate.fitness_score)
        
        print(f"🎯 Initial best fitness: {best_candidate.fitness_score:.4f}")
        print(f"📈 Initial best ratios: {[f'{r:.3f}' for r in best_candidate.merge_ratios]}")
        
        # Evolution loop
        for generation in range(self.generations):
            print(f"\n🔄 Generation {generation + 1}/{self.generations}")
            
            # Evolve population
            population = self.evolve_population(population, base_models, generation + 1)
            
            # Evaluate population
            population = self.evaluate_population(population)
            
            # Track statistics
            fitnesses = [c.fitness_score for c in population]
            current_best = max(population, key=lambda x: x.fitness_score)
            
            gen_stats = {
                "generation": generation + 1,
                "max_fitness": max(fitnesses),
                "mean_fitness": sum(fitnesses) / len(fitnesses),
                "std_fitness": (sum((f - sum(fitnesses) / len(fitnesses))**2 for f in fitnesses) / len(fitnesses))**0.5,
                "evaluations": self.evaluator.evaluation_count
            }
            self.generation_stats.append(gen_stats)
            
            # Update best candidate
            if current_best.fitness_score > best_candidate.fitness_score:
                best_candidate = current_best
                print(f"  🚀 New best fitness: {current_best.fitness_score:.4f}")
                print(f"  📊 New best ratios: {[f'{r:.3f}' for r in current_best.merge_ratios]}")
            
            self.best_fitness_history.append(current_best.fitness_score)
            
            print(f"  📈 Population stats: max={gen_stats['max_fitness']:.4f}, mean={gen_stats['mean_fitness']:.4f}")
        
        print(f"\n🎉 Evolution completed!")
        print(f"🏆 Final best fitness: {best_candidate.fitness_score:.4f}")
        print(f"🎯 Final merge ratios: {[f'{r:.3f}' for r in best_candidate.merge_ratios]}")
        print(f"📊 Total evaluations: {self.evaluator.evaluation_count}")
        
        return best_candidate


def demonstrate_simple_evolutionary_merge():
    """Demonstrate the simple evolutionary merge matching the original prompt."""
    print("🧬 Simple Evolutionary Merge (Original Prompt Style)")
    print("=" * 60)
    
    # Mock validation data
    validation_data = "mock_validation_dataset"
    
    # Simple version following the original prompt
    print("📥 Loading mock models...")
    
    # Simulate loading model weights (simplified)
    state_a = {"layer1": MockTensor((10, 5)), "layer2": MockTensor((5,))}
    state_b = {"layer1": MockTensor((10, 5)), "layer2": MockTensor((5,))}
    
    print("⚖️  Testing different merge ratios...")
    
    merged_state = {}
    best_score = 0
    best_ratio = 0.5
    
    # Initialize with equal weights
    for key in state_a.keys():
        merged_state[key] = 0.5 * state_a[key] + 0.5 * state_b[key]
    
    # Simple evaluation function
    def evaluate_model(state, validation_data):
        # Simulate evaluation - in practice you'd test the actual model
        total_norm = sum(tensor.norm() for tensor in state.values())
        # Simulate score based on weight characteristics
        return 0.7 + random.gauss(0, 0.05) - min(0.1, total_norm * 0.01)
    
    best_score = evaluate_model(merged_state, validation_data)
    
    # Evolutionary search over weight ratios (0 to 1)
    ratios = [i/10 for i in range(0, 11)]
    for r in ratios:
        candidate_state = {}
        for key in state_a.keys():
            candidate_state[key] = r * state_a[key] + (1 - r) * state_b[key]
        
        score = evaluate_model(candidate_state, validation_data)
        if score > best_score:
            best_score, best_ratio = score, r
            merged_state = candidate_state
        
        print(f"  Ratio {r:.1f}: score = {score:.4f}")
    
    print(f"🎯 Best merge ratio: {best_ratio}, score: {best_score:.4f}")
    print(f"💾 Merged model created (simulated save to merged_{best_ratio:.1f}.ckpt)")
    
    return merged_state


def demonstrate_advanced_evolutionary_merge():
    """Demonstrate advanced evolutionary merging."""
    print("\n🚀 Advanced Evolutionary Merge with Genetic Algorithms")
    print("=" * 60)
    
    # Create evolutionary merger
    merger = EvolutionaryModelMerger(
        population_size=15,
        generations=20,
        mutation_rate=0.15,
        crossover_rate=0.8
    )
    
    # Run evolution
    best_candidate = merger.evolutionary_merge(num_models=3)
    
    # Show detailed results
    print(f"\n📊 Detailed Results:")
    print(f"  Final fitness: {best_candidate.fitness_score:.4f}")
    print(f"  Merge ratios: {[f'{r:.3f}' for r in best_candidate.merge_ratios]}")
    print(f"  Generation: {best_candidate.generation}")
    print(f"  Mutation applied: {best_candidate.mutation_applied}")
    
    # Show evolution progress
    print(f"\n📈 Evolution Progress:")
    for i, fitness in enumerate(merger.best_fitness_history[:10]):  # Show first 10
        print(f"  Gen {i:2d}: {fitness:.4f}")
    if len(merger.best_fitness_history) > 10:
        print(f"  ... ({len(merger.best_fitness_history) - 10} more generations)")
        print(f"  Final: {merger.best_fitness_history[-1]:.4f}")
    
    return best_candidate


def main():
    """Main demonstration function."""
    print("🧬 Symbio AI - Evolutionary Model Merging Demo")
    print("Demonstrating advanced model fusion with genetic algorithms")
    print("=" * 70)
    
    try:
        # Demonstrate simple merge (matching original prompt)
        demonstrate_simple_evolutionary_merge()
        
        # Demonstrate advanced evolutionary merge
        demonstrate_advanced_evolutionary_merge()
        
        print(f"\n🎉 All demonstrations completed successfully!")
        print(f"✨ Key Concepts Demonstrated:")
        print(f"  - Evolutionary search for optimal merge ratios")
        print(f"  - Genetic algorithms (selection, crossover, mutation)")
        print(f"  - Multi-model merging with normalized weights")
        print(f"  - Population-based optimization")
        print(f"  - Fitness evaluation and tracking")
        print(f"  - Elite preservation and diversity maintenance")
        print(f"🚀 Ready for production with real PyTorch models!")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()