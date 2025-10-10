#!/usr/bin/env python3
"""
Evolutionary Model Merging Demo for Symbio AI

This demonstrates the evolutionary model merging utility with mock models
to show the concept in action without requiring actual trained models.
"""

import torch
import torch.nn as nn
import numpy as np
import sys
from pathlib import Path
import random

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from models.merger import evolutionary_merge, MergeConfig, EvolutionaryModelMerger, DefaultModelEvaluator


# Simple CNN model for demonstration
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(64, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


# Mock dataset for demonstration
class MockDataset:
    def __init__(self, num_samples=100, batch_size=16):
        self.num_samples = num_samples
        self.batch_size = batch_size
        
    def __iter__(self):
        for i in range(0, self.num_samples, self.batch_size):
            batch_size = min(self.batch_size, self.num_samples - i)
            # Generate random data (3 channels, 32x32 images)
            data = torch.randn(batch_size, 3, 32, 32)
            # Random labels (10 classes)
            target = torch.randint(0, 10, (batch_size,))
            yield data, target
    
    def __len__(self):
        return (self.num_samples + self.batch_size - 1) // self.batch_size


def create_mock_models():
    """Create mock model checkpoints for demonstration."""
    print("🏗️  Creating mock model checkpoints...")
    
    # Create different models with slight variations
    models = []
    for i in range(3):
        model = SimpleCNN()
        
        # Initialize with different random weights to simulate different trained models
        torch.manual_seed(42 + i * 100)
        for param in model.parameters():
            nn.init.normal_(param, mean=0.0, std=0.1 + i * 0.05)
        
        model_path = f"mock_model_{i}.pth"
        torch.save(model.state_dict(), model_path)
        models.append(model_path)
        print(f"  ✅ Created {model_path}")
    
    return models


class ImprovedMockEvaluator:
    """Improved mock evaluator that simulates realistic model performance."""
    
    def __init__(self, model_class):
        self.model_class = model_class
        self.device = "cpu"  # Use CPU for demo
    
    def evaluate(self, model_weights, validation_data):
        """Simulate model evaluation with more realistic scoring."""
        try:
            # Load model with weights
            model = self.model_class()
            model.load_state_dict(model_weights)
            model.eval()
            
            total_loss = 0.0
            correct = 0
            total = 0
            
            with torch.no_grad():
                for batch_idx, (data, target) in enumerate(validation_data):
                    output = model(data)
                    loss = nn.functional.cross_entropy(output, target)
                    total_loss += loss.item()
                    
                    pred = output.argmax(dim=1, keepdim=True)
                    correct += pred.eq(target.view_as(pred)).sum().item()
                    total += target.size(0)
            
            accuracy = correct / total
            avg_loss = total_loss / len(validation_data)
            
            # Add some noise to simulate variation
            noise = random.gauss(0, 0.02)
            fitness = accuracy - 0.1 * avg_loss + noise
            
            metrics = {
                "accuracy": accuracy,
                "loss": avg_loss,
                "correct": correct,
                "total": total
            }
            
            return fitness, metrics
            
        except Exception as e:
            print(f"❌ Evaluation error: {e}")
            return -1.0, {"error": str(e)}


def demonstrate_simple_merge():
    """Demonstrate the simple evolutionary merge function."""
    print("\n🧬 Simple Evolutionary Merge Demo")
    print("-" * 40)
    
    # Create mock models
    models = create_mock_models()
    
    # Create mock validation dataset
    validation_data = MockDataset(num_samples=200, batch_size=16)
    
    # Configure merge
    config = MergeConfig(
        population_size=8,
        generations=5,
        mutation_rate=0.2,
        crossover_rate=0.7,
        early_stopping_patience=3,
        parallel_evaluation=False  # Disable for demo
    )
    
    print(f"🔧 Configuration:")
    print(f"  Population size: {config.population_size}")
    print(f"  Generations: {config.generations}")
    print(f"  Mutation rate: {config.mutation_rate}")
    print(f"  Crossover rate: {config.crossover_rate}")
    
    # Run simple merge (using first two models)
    try:
        merged_weights = evolutionary_merge(
            models[0], 
            models[1], 
            validation_data, 
            SimpleCNN, 
            config
        )
        print("✅ Simple merge completed successfully!")
        
    except Exception as e:
        print(f"❌ Simple merge failed: {e}")


def demonstrate_advanced_merge():
    """Demonstrate advanced multi-model evolutionary merging."""
    print("\n🚀 Advanced Multi-Model Merge Demo")
    print("-" * 40)
    
    # Create mock models
    models = create_mock_models()
    
    # Create validation dataset
    validation_data = MockDataset(num_samples=200, batch_size=16)
    
    # Configure advanced merge
    config = MergeConfig(
        population_size=12,
        generations=8,
        mutation_rate=0.15,
        crossover_rate=0.8,
        selection_method="tournament",
        tournament_size=3,
        elite_ratio=0.2,
        early_stopping_patience=5,
        parallel_evaluation=False,
        save_intermediate=False  # Disable for demo
    )
    
    # Create evaluator
    evaluator = ImprovedMockEvaluator(SimpleCNN)
    
    # Create merger
    merger = EvolutionaryModelMerger(config, evaluator)
    
    print(f"🔧 Advanced Configuration:")
    print(f"  Population size: {config.population_size}")
    print(f"  Generations: {config.generations}")
    print(f"  Selection method: {config.selection_method}")
    print(f"  Elite ratio: {config.elite_ratio}")
    
    try:
        # Run evolution
        best_candidate = merger.evolutionary_merge(
            models, 
            validation_data,
            "demo_output"
        )
        
        print("✅ Advanced merge completed successfully!")
        print(f"📊 Best Results:")
        print(f"  Fitness score: {best_candidate.fitness_score:.4f}")
        print(f"  Merge ratios: {[f'{r:.3f}' for r in best_candidate.merge_ratios]}")
        print(f"  Generation: {best_candidate.generation}")
        
        # Show evolution progress
        if merger.best_fitness_history:
            print(f"📈 Evolution Progress:")
            for i, fitness in enumerate(merger.best_fitness_history):
                print(f"  Generation {i}: {fitness:.4f}")
        
    except Exception as e:
        print(f"❌ Advanced merge failed: {e}")
        import traceback
        traceback.print_exc()


def demonstrate_merge_strategies():
    """Demonstrate different merging strategies."""
    print("\n⚙️  Different Merge Strategies Demo")
    print("-" * 40)
    
    from models.merger import ties_merge, dare_merge
    
    # Create mock models
    models = create_mock_models()
    
    print("🔀 Testing TIES merging...")
    try:
        ties_result = ties_merge(models[:2], [0.6, 0.4], trim_ratio=0.1)
        print(f"  ✅ TIES merge: {len(ties_result)} parameters merged")
    except Exception as e:
        print(f"  ❌ TIES merge failed: {e}")
    
    print("🎲 Testing DARE merging...")
    try:
        dare_result = dare_merge(models[:2], [0.5, 0.5], drop_rate=0.1)
        print(f"  ✅ DARE merge: {len(dare_result)} parameters merged")
    except Exception as e:
        print(f"  ❌ DARE merge failed: {e}")


def cleanup_demo_files():
    """Clean up demo files."""
    print("\n🧹 Cleaning up demo files...")
    demo_files = ["mock_model_0.pth", "mock_model_1.pth", "mock_model_2.pth"]
    for file in demo_files:
        if Path(file).exists():
            Path(file).unlink()
            print(f"  🗑️  Removed {file}")


def main():
    """Main demo function."""
    print("🧬 Symbio AI - Evolutionary Model Merging Demo")
    print("=" * 60)
    
    try:
        # Run demonstrations
        demonstrate_simple_merge()
        demonstrate_advanced_merge()
        demonstrate_merge_strategies()
        
        print("\n🎉 All demonstrations completed successfully!")
        print("💡 Key Features Demonstrated:")
        print("  - Evolutionary algorithm with genetic operations")
        print("  - Multi-model merging capabilities") 
        print("  - Different selection strategies")
        print("  - Advanced merging techniques (TIES, DARE)")
        print("  - Fitness tracking and evolution monitoring")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        cleanup_demo_files()


if __name__ == "__main__":
    main()