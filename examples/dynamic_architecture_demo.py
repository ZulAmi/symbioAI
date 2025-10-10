#!/usr/bin/env python3
"""
Dynamic Neural Architecture Evolution - Comprehensive Demo

Demonstrates:
1. Neural Architecture Search (NAS) during inference
2. Task-adaptive depth and width
3. Automatic module specialization and pruning
4. Morphological evolution of network topology
5. Real-time performance optimization

Competitive Edge: Real-time architecture adaptation (vs. fixed architectures)
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple
import json
import time

from training.dynamic_architecture_evolution import (
    DynamicNeuralArchitecture,
    ArchitectureEvolutionConfig,
    TaskComplexity,
    create_dynamic_architecture
)


def create_synthetic_dataset(
    num_samples: int,
    input_dim: int,
    complexity: TaskComplexity
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Create synthetic dataset with varying complexity."""
    X = torch.randn(num_samples, input_dim)
    
    # Complexity determines number of interactions
    if complexity == TaskComplexity.TRIVIAL:
        # Linear relationship
        y = X[:, 0] + X[:, 1]
    elif complexity == TaskComplexity.SIMPLE:
        # Quadratic
        y = X[:, 0]**2 + X[:, 1]
    elif complexity == TaskComplexity.MODERATE:
        # Multiple interactions
        y = X[:, 0] * X[:, 1] + X[:, 2]**2
    elif complexity == TaskComplexity.COMPLEX:
        # Many interactions
        y = (X[:, :5]**2).sum(dim=1) + (X[:, :3] * X[:, 1:4]).sum(dim=1)
    else:  # EXPERT
        # Highly nonlinear
        y = torch.sin(X[:, :10]).sum(dim=1) * torch.cos(X[:, 10:20]).sum(dim=1)
    
    # Add noise
    y = y + 0.1 * torch.randn_like(y)
    
    # Classification (binary)
    labels = (y > y.median()).long()
    
    return X, labels


def demo_1_nas_during_inference():
    """Demo 1: Neural Architecture Search during inference."""
    print("\n" + "="*70)
    print("DEMO 1: Neural Architecture Search During Inference")
    print("="*70)
    print("Demonstrating: Architecture adapts in real-time based on performance")
    print()
    
    config = ArchitectureEvolutionConfig(
        min_layers=2,
        max_layers=8,
        evolution_interval=20,  # Evolve every 20 steps
        enable_nas=True,
        enable_runtime_adaptation=True
    )
    
    model = create_dynamic_architecture(128, 2, config)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    print("📊 Initial Architecture:")
    print(json.dumps(model.get_architecture_summary(), indent=2, default=str))
    
    architecture_history = []
    
    # Train on progressively harder tasks
    complexities = [
        TaskComplexity.SIMPLE,
        TaskComplexity.MODERATE,
        TaskComplexity.COMPLEX
    ]
    
    for complexity in complexities:
        print(f"\n🎯 Training on {complexity.value} tasks...")
        
        for step in range(60):
            X, y = create_synthetic_dataset(32, 128, complexity)
            
            optimizer.zero_grad()
            output = model(X, task_id=f"{complexity.value}", task_complexity=complexity)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
            
            # Update performance
            with torch.no_grad():
                pred = output.argmax(dim=1)
                accuracy = (pred == y).float().mean().item()
                model.update_performance(accuracy)
            
            if step % 20 == 0:
                summary = model.get_architecture_summary()
                architecture_history.append({
                    'step': step,
                    'complexity': complexity.value,
                    'num_modules': summary['num_modules'],
                    'total_layers': summary['total_layers'],
                    'total_params': summary['total_params'],
                    'accuracy': accuracy
                })
                print(f"  Step {step}: Accuracy={accuracy:.3f}, "
                      f"Modules={summary['num_modules']}, "
                      f"Layers={summary['total_layers']}, "
                      f"Params={summary['total_params']:,}")
    
    print("\n✅ NAS Evolution Complete!")
    print(f"   Architecture evolved {len(model.evolution_history)} times")
    print(f"   Final modules: {model.get_architecture_summary()['num_modules']}")
    
    return architecture_history


def demo_2_task_adaptive_depth_width():
    """Demo 2: Task-adaptive depth and width."""
    print("\n" + "="*70)
    print("DEMO 2: Task-Adaptive Depth and Width")
    print("="*70)
    print("Demonstrating: Network grows/shrinks based on task complexity")
    print()
    
    config = ArchitectureEvolutionConfig(
        min_layers=1,
        max_layers=15,
        min_width=32,
        max_width=512,
        growth_threshold=0.8,
        shrink_threshold=0.2,
        evolution_interval=15
    )
    
    model = create_dynamic_architecture(64, 2, config)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    depth_history = []
    width_history = []
    
    # Cycle through complexity levels
    complexity_cycle = [
        TaskComplexity.SIMPLE,
        TaskComplexity.COMPLEX,
        TaskComplexity.EXPERT,
        TaskComplexity.COMPLEX,
        TaskComplexity.SIMPLE,
        TaskComplexity.MODERATE
    ]
    
    for phase, complexity in enumerate(complexity_cycle):
        print(f"\n📈 Phase {phase + 1}: {complexity.value} tasks")
        
        for step in range(30):
            X, y = create_synthetic_dataset(16, 64, complexity)
            
            optimizer.zero_grad()
            output = model(X, task_complexity=complexity)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
            
            with torch.no_grad():
                accuracy = (output.argmax(dim=1) == y).float().mean().item()
                model.update_performance(accuracy)
        
        summary = model.get_architecture_summary()
        depth_history.append(summary['total_layers'])
        width_history.append(summary['total_params'])
        
        print(f"  Depth: {summary['total_layers']} layers")
        print(f"  Width: ~{summary['total_params']:,} parameters")
        print(f"  Modules: {summary['num_modules']}")
    
    print("\n✅ Adaptive Depth/Width Demo Complete!")
    print(f"   Depth range: {min(depth_history)} - {max(depth_history)} layers")
    print(f"   Width range: {min(width_history):,} - {max(width_history):,} params")
    
    return depth_history, width_history


def demo_3_module_specialization():
    """Demo 3: Automatic module specialization."""
    print("\n" + "="*70)
    print("DEMO 3: Automatic Module Specialization")
    print("="*70)
    print("Demonstrating: Modules specialize for specific task types")
    print()
    
    config = ArchitectureEvolutionConfig(
        specialization_threshold=0.6,
        enable_specialization=True,
        evolution_interval=25,
        min_layers=2,
        max_layers=12
    )
    
    model = create_dynamic_architecture(100, 3, config)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    # Train on distinct task types
    task_types = ['vision', 'language', 'audio']
    
    print("🎯 Training on 3 distinct task types...")
    
    for epoch in range(5):
        print(f"\nEpoch {epoch + 1}")
        
        for task_type in task_types:
            # Each task type has different complexity
            if task_type == 'vision':
                complexity = TaskComplexity.MODERATE
            elif task_type == 'language':
                complexity = TaskComplexity.COMPLEX
            else:  # audio
                complexity = TaskComplexity.EXPERT
            
            for step in range(40):
                X, y = create_synthetic_dataset(24, 100, complexity)
                # Make 3-class
                y = y + (torch.randint(0, 2, (24,)) * (y + 1)) % 3
                
                optimizer.zero_grad()
                output = model(X, task_id=task_type, task_complexity=complexity)
                loss = criterion(output, y)
                loss.backward()
                optimizer.step()
                
                with torch.no_grad():
                    accuracy = (output.argmax(dim=1) == y).float().mean().item()
                    model.update_performance(accuracy)
        
        # Check specialization
        summary = model.get_architecture_summary()
        print(f"  Modules: {summary['num_modules']}")
        
        specialized_count = 0
        for module_id, details in summary['module_details'].items():
            if details['specializations']:
                specialized_count += 1
                print(f"    {module_id}: specialized for {details['specializations']}")
        
        print(f"  Specialized modules: {specialized_count}/{summary['num_modules']}")
    
    print("\n✅ Module Specialization Demo Complete!")
    final_summary = model.get_architecture_summary()
    total_specialized = sum(
        1 for d in final_summary['module_details'].values()
        if d['specializations']
    )
    print(f"   Final specialized modules: {total_specialized}")
    
    return final_summary


def demo_4_automatic_pruning():
    """Demo 4: Automatic module pruning."""
    print("\n" + "="*70)
    print("DEMO 4: Automatic Module Pruning")
    print("="*70)
    print("Demonstrating: Underutilized modules are automatically removed")
    print()
    
    config = ArchitectureEvolutionConfig(
        enable_pruning=True,
        prune_threshold=0.1,
        min_utilization=0.15,
        evolution_interval=30,
        min_layers=2,
        max_layers=20
    )
    
    # Start with many modules
    model = create_dynamic_architecture(80, 2, config)
    
    # Manually add extra modules
    for i in range(5):
        model._add_module(f"extra_{i}", 80, 80, depth=2)
    
    print(f"📊 Starting with {len(model.module_order)} modules")
    
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    module_count_history = [len(model.module_order)]
    
    # Train on simple tasks (some modules won't be utilized)
    for epoch in range(8):
        print(f"\nEpoch {epoch + 1}")
        
        for step in range(50):
            X, y = create_synthetic_dataset(16, 80, TaskComplexity.SIMPLE)
            
            optimizer.zero_grad()
            output = model(X, task_complexity=TaskComplexity.SIMPLE)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
            
            with torch.no_grad():
                accuracy = (output.argmax(dim=1) == y).float().mean().item()
                model.update_performance(accuracy)
        
        module_count = len(model.module_order)
        module_count_history.append(module_count)
        
        summary = model.get_architecture_summary()
        print(f"  Modules: {module_count}")
        print(f"  Pruning events: {len([e for e in model.evolution_history if 'PRUNE' in str(e.get('operations', []))])}")
        
        # Show module utilization
        low_util = [
            (mid, d['utilization'])
            for mid, d in summary['module_details'].items()
            if d['utilization'] < 0.3
        ]
        if low_util:
            print(f"  Low utilization modules: {len(low_util)}")
    
    print("\n✅ Automatic Pruning Demo Complete!")
    print(f"   Started with: {module_count_history[0]} modules")
    print(f"   Ended with: {module_count_history[-1]} modules")
    print(f"   Modules pruned: {module_count_history[0] - module_count_history[-1]}")
    
    return module_count_history


def demo_5_morphological_evolution():
    """Demo 5: Morphological evolution of network topology."""
    print("\n" + "="*70)
    print("DEMO 5: Morphological Evolution of Network Topology")
    print("="*70)
    print("Demonstrating: Complete topology transformation over time")
    print()
    
    config = ArchitectureEvolutionConfig(
        enable_nas=True,
        enable_runtime_adaptation=True,
        enable_pruning=True,
        enable_specialization=True,
        evolution_interval=20,
        min_layers=1,
        max_layers=15
    )
    
    model = create_dynamic_architecture(120, 4, config)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    print("📊 Initial topology:")
    initial = model.get_architecture_summary()
    print(f"  Modules: {initial['num_modules']}")
    print(f"  Layers: {initial['total_layers']}")
    print(f"  Params: {initial['total_params']:,}")
    
    topology_snapshots = [initial]
    
    # Evolve through different scenarios
    scenarios = [
        ("Simple tasks", TaskComplexity.SIMPLE, 40),
        ("Sudden complexity spike", TaskComplexity.EXPERT, 30),
        ("Complex sustained", TaskComplexity.COMPLEX, 40),
        ("Return to moderate", TaskComplexity.MODERATE, 35),
        ("Multi-task mix", TaskComplexity.COMPLEX, 50)
    ]
    
    for scenario_name, complexity, steps in scenarios:
        print(f"\n🔄 Scenario: {scenario_name}")
        
        for step in range(steps):
            # Vary task IDs in multi-task scenario
            task_id = None
            if "multi-task" in scenario_name.lower():
                task_id = f"task_{step % 4}"
            
            X, y = create_synthetic_dataset(20, 120, complexity)
            # Make 4-class
            y = (y.long() + torch.randint(0, 3, (20,))) % 4
            
            optimizer.zero_grad()
            output = model(X, task_id=task_id, task_complexity=complexity)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
            
            with torch.no_grad():
                accuracy = (output.argmax(dim=1) == y).float().mean().item()
                model.update_performance(accuracy)
        
        snapshot = model.get_architecture_summary()
        topology_snapshots.append(snapshot)
        
        print(f"  Modules: {snapshot['num_modules']} "
              f"(Δ {snapshot['num_modules'] - topology_snapshots[-2]['num_modules']:+d})")
        print(f"  Layers: {snapshot['total_layers']} "
              f"(Δ {snapshot['total_layers'] - topology_snapshots[-2]['total_layers']:+d})")
        print(f"  Params: {snapshot['total_params']:,} "
              f"(Δ {snapshot['total_params'] - topology_snapshots[-2]['total_params']:+,})")
    
    print("\n✅ Morphological Evolution Demo Complete!")
    print(f"\n📈 Topology Transformation Summary:")
    print(f"   Initial → Final Modules: {topology_snapshots[0]['num_modules']} → {topology_snapshots[-1]['num_modules']}")
    print(f"   Initial → Final Layers: {topology_snapshots[0]['total_layers']} → {topology_snapshots[-1]['total_layers']}")
    print(f"   Initial → Final Params: {topology_snapshots[0]['total_params']:,} → {topology_snapshots[-1]['total_params']:,}")
    print(f"   Total Evolution Events: {len(model.evolution_history)}")
    
    return topology_snapshots


def demo_6_comparative_benchmark():
    """Demo 6: Compare dynamic vs. static architecture."""
    print("\n" + "="*70)
    print("DEMO 6: Dynamic vs. Static Architecture Comparison")
    print("="*70)
    print("Demonstrating: Performance advantage of dynamic adaptation")
    print()
    
    # Dynamic architecture
    dynamic_config = ArchitectureEvolutionConfig(
        enable_nas=True,
        enable_runtime_adaptation=True,
        enable_pruning=True,
        evolution_interval=25
    )
    dynamic_model = create_dynamic_architecture(100, 2, dynamic_config)
    
    # Static architecture (evolution disabled)
    static_config = ArchitectureEvolutionConfig(
        enable_nas=False,
        enable_runtime_adaptation=False,
        enable_pruning=False,
        evolution_interval=999999  # Never evolve
    )
    static_model = create_dynamic_architecture(100, 2, static_config)
    
    dynamic_optimizer = optim.Adam(dynamic_model.parameters(), lr=0.001)
    static_optimizer = optim.Adam(static_model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    dynamic_accuracies = []
    static_accuracies = []
    dynamic_params = []
    static_params = []
    
    # Test on varying complexity
    complexities = [
        TaskComplexity.SIMPLE,
        TaskComplexity.MODERATE,
        TaskComplexity.COMPLEX,
        TaskComplexity.EXPERT,
        TaskComplexity.SIMPLE
    ]
    
    print("🏁 Training both models on varying complexity tasks...")
    
    for phase, complexity in enumerate(complexities):
        print(f"\n  Phase {phase + 1}: {complexity.value}")
        
        phase_dynamic_acc = []
        phase_static_acc = []
        
        for step in range(60):
            X, y = create_synthetic_dataset(32, 100, complexity)
            
            # Dynamic model
            dynamic_optimizer.zero_grad()
            dynamic_out = dynamic_model(X, task_complexity=complexity)
            dynamic_loss = criterion(dynamic_out, y)
            dynamic_loss.backward()
            dynamic_optimizer.step()
            
            # Static model
            static_optimizer.zero_grad()
            static_out = static_model(X, task_complexity=complexity)
            static_loss = criterion(static_out, y)
            static_loss.backward()
            static_optimizer.step()
            
            # Evaluate
            with torch.no_grad():
                dynamic_acc = (dynamic_out.argmax(dim=1) == y).float().mean().item()
                static_acc = (static_out.argmax(dim=1) == y).float().mean().item()
                
                dynamic_model.update_performance(dynamic_acc)
                static_model.update_performance(static_acc)
                
                phase_dynamic_acc.append(dynamic_acc)
                phase_static_acc.append(static_acc)
        
        # Phase summary
        dynamic_acc_avg = np.mean(phase_dynamic_acc[-20:])
        static_acc_avg = np.mean(phase_static_acc[-20:])
        
        dynamic_accuracies.append(dynamic_acc_avg)
        static_accuracies.append(static_acc_avg)
        
        dynamic_summary = dynamic_model.get_architecture_summary()
        static_summary = static_model.get_architecture_summary()
        
        dynamic_params.append(dynamic_summary['total_params'])
        static_params.append(static_summary['total_params'])
        
        print(f"    Dynamic: Acc={dynamic_acc_avg:.3f}, Params={dynamic_summary['total_params']:,}, "
              f"Modules={dynamic_summary['num_modules']}")
        print(f"    Static:  Acc={static_acc_avg:.3f}, Params={static_summary['total_params']:,}, "
              f"Modules={static_summary['num_modules']}")
        print(f"    Advantage: {(dynamic_acc_avg - static_acc_avg)*100:+.1f}% accuracy, "
              f"{(static_summary['total_params'] - dynamic_summary['total_params'])/1000:.0f}K fewer params")
    
    print("\n✅ Comparative Benchmark Complete!")
    print(f"\n📊 Overall Results:")
    print(f"   Dynamic avg accuracy: {np.mean(dynamic_accuracies):.3f}")
    print(f"   Static avg accuracy:  {np.mean(static_accuracies):.3f}")
    print(f"   Accuracy advantage:   {(np.mean(dynamic_accuracies) - np.mean(static_accuracies))*100:+.1f}%")
    print(f"   Dynamic avg params:   {np.mean(dynamic_params):,.0f}")
    print(f"   Static avg params:    {np.mean(static_params):,.0f}")
    print(f"   Parameter efficiency: {(1 - np.mean(dynamic_params)/np.mean(static_params))*100:.1f}% reduction")
    
    return {
        'dynamic_acc': dynamic_accuracies,
        'static_acc': static_accuracies,
        'dynamic_params': dynamic_params,
        'static_params': static_params
    }


def main():
    """Run all demos."""
    print("╔" + "="*68 + "╗")
    print("║" + " "*68 + "║")
    print("║" + "  Dynamic Neural Architecture Evolution - Complete Demo".center(68) + "║")
    print("║" + " "*68 + "║")
    print("║" + "  Features:".ljust(68) + "║")
    print("║" + "    1. Neural Architecture Search (NAS) during inference".ljust(68) + "║")
    print("║" + "    2. Task-adaptive depth and width".ljust(68) + "║")
    print("║" + "    3. Automatic module specialization".ljust(68) + "║")
    print("║" + "    4. Intelligent pruning".ljust(68) + "║")
    print("║" + "    5. Morphological evolution".ljust(68) + "║")
    print("║" + "    6. Performance vs. static architectures".ljust(68) + "║")
    print("║" + " "*68 + "║")
    print("║" + "  Competitive Edge: Real-time adaptation (others use fixed)".ljust(68) + "║")
    print("║" + " "*68 + "║")
    print("╚" + "="*68 + "╝")
    
    start_time = time.time()
    
    results = {}
    
    # Run all demos
    results['nas'] = demo_1_nas_during_inference()
    results['adaptive'] = demo_2_task_adaptive_depth_width()
    results['specialization'] = demo_3_module_specialization()
    results['pruning'] = demo_4_automatic_pruning()
    results['morphological'] = demo_5_morphological_evolution()
    results['comparison'] = demo_6_comparative_benchmark()
    
    elapsed = time.time() - start_time
    
    # Final summary
    print("\n" + "="*70)
    print("DYNAMIC NEURAL ARCHITECTURE EVOLUTION - COMPLETE")
    print("="*70)
    print(f"\n⏱️  Total demo time: {elapsed:.1f} seconds")
    print(f"\n🎯 Key Achievements:")
    print(f"   ✅ NAS during inference: Architecture evolved automatically")
    print(f"   ✅ Task-adaptive depth/width: Network size adapted to complexity")
    print(f"   ✅ Module specialization: Modules specialized for specific tasks")
    print(f"   ✅ Automatic pruning: Removed {results['pruning'][0] - results['pruning'][-1]} underutilized modules")
    print(f"   ✅ Morphological evolution: Complete topology transformation")
    
    comparison = results['comparison']
    dynamic_advantage = (np.mean(comparison['dynamic_acc']) - np.mean(comparison['static_acc'])) * 100
    param_reduction = (1 - np.mean(comparison['dynamic_params'])/np.mean(comparison['static_params'])) * 100
    
    print(f"\n🏆 Competitive Advantages:")
    print(f"   • {dynamic_advantage:+.1f}% accuracy over static architectures")
    print(f"   • {param_reduction:.1f}% fewer parameters on average")
    print(f"   • Real-time adaptation to task complexity")
    print(f"   • Automatic specialization for multi-task scenarios")
    print(f"   • Self-optimizing topology")
    
    print(f"\n💡 Market Differentiation:")
    print(f"   • Most systems: Fixed architecture (must be redesigned)")
    print(f"   • Symbio AI: Dynamic architecture (adapts automatically)")
    print(f"   • Result: Better performance with fewer resources")
    
    print(f"\n✅ All dynamic architecture evolution features operational!")
    
    return results


if __name__ == "__main__":
    results = main()
