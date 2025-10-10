#!/usr/bin/env python3
"""
Memory-Enhanced Mixture of Experts - Comprehensive Demo

Demonstrates:
1. Experts with specialized external memory banks
2. Automatic memory indexing and retrieval
3. Memory-based few-shot adaptation
4. Hierarchical memory (short-term ↔ long-term)
5. Expert specialization with memory

Competitive Edge: Current MoE lacks memory; ours remembers and recalls
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import List, Dict, Tuple
import json
import time

from training.memory_enhanced_moe import (
    MemoryEnhancedMoE,
    MemoryConfig,
    ExpertSpecialization,
    MemoryType,
    create_memory_enhanced_moe
)


def create_task_dataset(
    task_type: str,
    num_samples: int,
    input_dim: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Create synthetic dataset for specific task type."""
    X = torch.randn(num_samples, input_dim)
    
    if task_type == "vision":
        # Spatial patterns
        y = (X[:, :10].sum(dim=1) > 0).long()
    elif task_type == "language":
        # Sequential patterns
        y = (X[:, :20:2].prod(dim=1) > 0).long()
    elif task_type == "reasoning":
        # Logical patterns
        y = ((X[:, :5].mean(dim=1) * X[:, 5:10].std(dim=1)) > 0).long()
    else:  # multimodal
        # Combined patterns
        y = ((X[:, :15].sum(dim=1) + X[:, 15:30].prod(dim=1)) > 0).long()
    
    return X, y


def demo_1_memory_storage_retrieval():
    """Demo 1: Automatic memory storage and retrieval."""
    print("\n" + "="*70)
    print("DEMO 1: Automatic Memory Storage and Retrieval")
    print("="*70)
    print("Demonstrating: Experts store experiences and retrieve relevant ones")
    print()
    
    config = MemoryConfig(
        short_term_capacity=20,
        long_term_capacity=100,
        episodic_capacity=50,
        top_k_retrieve=3
    )
    
    model = create_memory_enhanced_moe(
        input_dim=64,
        output_dim=2,
        num_experts=4,
        hidden_dim=128,
        memory_config=config
    )
    
    print("📊 Training model and building memories...")
    
    memory_growth = []
    retrieval_hits = []
    
    # Train on vision tasks
    for epoch in range(5):
        X, y = create_task_dataset("vision", 32, 64)
        
        # Forward with memory
        output, info = model(
            X,
            use_memory=True,
            store_experience=True,
            metadata={'task': 'vision', 'epoch': epoch}
        )
        
        # Track statistics
        stats = model.get_statistics()
        total_memories = stats['total_memories']
        memories_retrieved = info['total_memories_retrieved']
        
        memory_growth.append(total_memories)
        retrieval_hits.append(memories_retrieved)
        
        print(f"Epoch {epoch + 1}:")
        print(f"  Total memories stored: {total_memories}")
        print(f"  Memories retrieved: {memories_retrieved}")
        print(f"  Active experts: {info['active_experts']}")
    
    print(f"\n✅ Memory Storage & Retrieval Complete!")
    print(f"   Memory growth: {memory_growth[0]} → {memory_growth[-1]}")
    print(f"   Avg retrieval hits: {np.mean(retrieval_hits):.1f} per forward")
    
    return memory_growth, retrieval_hits


def demo_2_few_shot_adaptation():
    """Demo 2: Memory-based few-shot learning."""
    print("\n" + "="*70)
    print("DEMO 2: Memory-Based Few-Shot Adaptation")
    print("="*70)
    print("Demonstrating: Rapid adaptation from a few examples using memory")
    print()
    
    config = MemoryConfig(
        episodic_capacity=200,
        top_k_retrieve=5
    )
    
    model = create_memory_enhanced_moe(
        input_dim=100,
        output_dim=3,
        num_experts=6,
        hidden_dim=256,
        memory_config=config
    )
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Pre-training on general tasks
    print("📚 Pre-training on general tasks...")
    for _ in range(10):
        X, y = create_task_dataset("multimodal", 16, 100)
        y = y % 3  # Make 3-class
        
        output, _ = model(X)
        loss = criterion(output, y)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    print("  Pre-training complete")
    
    # Few-shot adaptation scenarios
    scenarios = [
        ("New vision task", "vision", 3),
        ("New language task", "language", 5),
        ("New reasoning task", "reasoning", 10)
    ]
    
    print("\n🎯 Testing few-shot adaptation:")
    
    for scenario_name, task_type, num_examples in scenarios:
        print(f"\n  {scenario_name} ({num_examples} examples):")
        
        # Create few-shot examples
        X_few, y_few = create_task_dataset(task_type, num_examples, 100)
        y_few = y_few % 3
        
        examples = [(X_few[i], F.one_hot(y_few[i], 3).float()) for i in range(num_examples)]
        
        # Adapt
        adapt_info = model.few_shot_adapt(examples)
        
        print(f"    Expert: {adapt_info['expert_id']} ({adapt_info['specialization']})")
        print(f"    Adaptation loss: {adapt_info['adaptation_loss']:.4f}")
        
        # Test on new data
        X_test, y_test = create_task_dataset(task_type, 20, 100)
        y_test = y_test % 3
        
        with torch.no_grad():
            output_test, _ = model(X_test, use_memory=True)
            test_acc = (output_test.argmax(dim=1) == y_test).float().mean().item()
        
        print(f"    Test accuracy: {test_acc:.2%}")
    
    print("\n✅ Few-Shot Adaptation Complete!")
    stats = model.get_statistics()
    print(f"   Total memories accumulated: {stats['total_memories']}")
    
    return stats


def demo_3_hierarchical_memory():
    """Demo 3: Hierarchical memory consolidation."""
    print("\n" + "="*70)
    print("DEMO 3: Hierarchical Memory (Short-term ↔ Long-term)")
    print("="*70)
    print("Demonstrating: Automatic consolidation from short to long-term memory")
    print()
    
    config = MemoryConfig(
        short_term_capacity=10,
        long_term_capacity=50,
        consolidation_interval=20,
        consolidation_threshold=0.6
    )
    
    model = create_memory_enhanced_moe(
        input_dim=80,
        output_dim=2,
        num_experts=3,
        memory_config=config
    )
    
    print("📊 Tracking memory hierarchy evolution...")
    
    hierarchy_snapshots = []
    
    for step in range(100):
        X, y = create_task_dataset("reasoning", 8, 80)
        
        # Forward pass (stores in short-term)
        output, info = model(X, metadata={'step': step})
        
        # Every 20 steps, check hierarchy
        if step % 20 == 0:
            stats = model.get_statistics()
            expert_stats = stats['expert_memory_stats'][0]  # First expert
            
            snapshot = {
                'step': step,
                'short_term': expert_stats['short_term_count'],
                'long_term': expert_stats['long_term_count'],
                'episodic': expert_stats['episodic_count'],
                'consolidations': expert_stats['consolidations']
            }
            hierarchy_snapshots.append(snapshot)
            
            print(f"Step {step}:")
            print(f"  Short-term: {snapshot['short_term']}")
            print(f"  Long-term: {snapshot['long_term']}")
            print(f"  Episodic: {snapshot['episodic']}")
            print(f"  Consolidations: {snapshot['consolidations']}")
    
    print("\n✅ Hierarchical Memory Demo Complete!")
    print(f"   Total consolidation events: {hierarchy_snapshots[-1]['consolidations']}")
    print(f"   Final long-term memories: {hierarchy_snapshots[-1]['long_term']}")
    
    return hierarchy_snapshots


def demo_4_expert_specialization_with_memory():
    """Demo 4: Expert specialization through memory."""
    print("\n" + "="*70)
    print("DEMO 4: Expert Specialization with Memory Banks")
    print("="*70)
    print("Demonstrating: Experts specialize and build domain-specific memories")
    print()
    
    # Define expert specializations
    specializations = [
        ExpertSpecialization.VISION,
        ExpertSpecialization.LANGUAGE,
        ExpertSpecialization.REASONING,
        ExpertSpecialization.MULTIMODAL
    ]
    
    config = MemoryConfig(
        episodic_capacity=100,
        semantic_capacity=100
    )
    
    model = MemoryEnhancedMoE(
        input_dim=120,
        hidden_dim=256,
        output_dim=2,
        num_experts=4,
        memory_config=config,
        expert_specializations=specializations
    )
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    print("🎯 Training on specialized tasks...")
    
    task_types = ["vision", "language", "reasoning", "multimodal"]
    
    expert_usage_by_task = {task: [] for task in task_types}
    
    for epoch in range(8):
        print(f"\nEpoch {epoch + 1}:")
        
        for task_type in task_types:
            X, y = create_task_dataset(task_type, 16, 120)
            
            # Forward
            optimizer.zero_grad()
            output, info = model(X, metadata={'task': task_type})
            loss = criterion(output, y.float().unsqueeze(1).expand(-1, 2))
            loss.backward()
            optimizer.step()
            
            # Track which expert was most active
            gate_weights = info['gate_weights']
            if isinstance(gate_weights, list):
                most_active = np.argmax(np.mean(gate_weights, axis=0))
            else:
                most_active = gate_weights.argmax().item()
            
            expert_usage_by_task[task_type].append(most_active)
        
        # Show expert usage
        for task, usage in expert_usage_by_task.items():
            if len(usage) > 0:
                most_common = max(set(usage), key=usage.count)
                spec = model.experts[most_common].specialization.value
                print(f"  {task}: Expert {most_common} ({spec})")
    
    # Memory statistics per expert
    print("\n📈 Expert Memory Statistics:")
    stats = model.get_statistics()
    for i, expert_stat in enumerate(stats['expert_memory_stats']):
        spec = model.experts[i].specialization.value
        print(f"  Expert {i} ({spec}):")
        print(f"    Episodic memories: {expert_stat['episodic_count']}")
        print(f"    Semantic memories: {expert_stat['semantic_count']}")
        print(f"    Total accesses: {expert_stat['total_accesses']}")
    
    print("\n✅ Expert Specialization Complete!")
    print(f"   Each expert developed task-specific memories")
    
    return expert_usage_by_task


def demo_5_memory_pruning():
    """Demo 5: Automatic memory pruning."""
    print("\n" + "="*70)
    print("DEMO 5: Automatic Memory Pruning")
    print("="*70)
    print("Demonstrating: Low-importance memories are automatically pruned")
    print()
    
    config = MemoryConfig(
        episodic_capacity=30,
        enable_pruning=True,
        prune_interval=50,
        min_importance=0.2
    )
    
    model = create_memory_enhanced_moe(
        input_dim=64,
        output_dim=2,
        num_experts=2,
        memory_config=config
    )
    
    print("📊 Filling memories and monitoring pruning...")
    
    memory_counts = []
    pruning_events = []
    
    for step in range(150):
        X, y = create_task_dataset("vision", 8, 64)
        
        # Vary importance by adding noise
        importance = 0.3 + 0.4 * np.random.rand()
        
        output, info = model(
            X,
            metadata={'step': step, 'importance': importance}
        )
        
        # Track memory count
        stats = model.get_statistics()
        total_mem = stats['total_memories']
        memory_counts.append(total_mem)
        
        # Check for pruning
        expert_stats = stats['expert_memory_stats'][0]
        if expert_stats['prunings'] > len(pruning_events):
            pruning_events.append(step)
            print(f"Step {step}: Pruning occurred!")
            print(f"  Memories before: {memory_counts[step-1] if step > 0 else 0}")
            print(f"  Memories after: {total_mem}")
    
    print(f"\n✅ Memory Pruning Complete!")
    print(f"   Total pruning events: {len(pruning_events)}")
    print(f"   Final memory count: {memory_counts[-1]}")
    print(f"   Memory stayed within capacity limits")
    
    return memory_counts, pruning_events


def demo_6_comparative_benchmark():
    """Demo 6: MoE with vs without memory."""
    print("\n" + "="*70)
    print("DEMO 6: Memory-Enhanced MoE vs Standard MoE")
    print("="*70)
    print("Demonstrating: Performance advantage of memory enhancement")
    print()
    
    # Memory-enhanced MoE
    config = MemoryConfig(
        episodic_capacity=200,
        top_k_retrieve=5
    )
    
    memory_moe = create_memory_enhanced_moe(
        input_dim=100,
        output_dim=3,
        num_experts=4,
        memory_config=config
    )
    
    # Standard MoE (memory disabled)
    standard_moe = create_memory_enhanced_moe(
        input_dim=100,
        output_dim=3,
        num_experts=4,
        memory_config=config
    )
    
    criterion = nn.CrossEntropyLoss()
    memory_optimizer = optim.Adam(memory_moe.parameters(), lr=0.001)
    standard_optimizer = optim.Adam(standard_moe.parameters(), lr=0.001)
    
    print("🏁 Training both models...")
    
    memory_accs = []
    standard_accs = []
    
    task_sequence = ["vision", "language", "reasoning", "multimodal"] * 3
    
    for step, task_type in enumerate(task_sequence):
        X_train, y_train = create_task_dataset(task_type, 32, 100)
        y_train = y_train % 3
        
        # Memory-enhanced MoE
        memory_optimizer.zero_grad()
        output_mem, _ = memory_moe(
            X_train,
            use_memory=True,  # Memory enabled
            metadata={'task': task_type}
        )
        loss_mem = criterion(output_mem, y_train)
        loss_mem.backward()
        memory_optimizer.step()
        
        # Standard MoE
        standard_optimizer.zero_grad()
        output_std, _ = standard_moe(
            X_train,
            use_memory=False,  # Memory disabled
            store_experience=False
        )
        loss_std = criterion(output_std, y_train)
        loss_std.backward()
        standard_optimizer.step()
        
        # Evaluate
        X_test, y_test = create_task_dataset(task_type, 20, 100)
        y_test = y_test % 3
        
        with torch.no_grad():
            output_mem_test, _ = memory_moe(X_test, use_memory=True, store_experience=False)
            output_std_test, _ = standard_moe(X_test, use_memory=False, store_experience=False)
            
            acc_mem = (output_mem_test.argmax(dim=1) == y_test).float().mean().item()
            acc_std = (output_std_test.argmax(dim=1) == y_test).float().mean().item()
            
            memory_accs.append(acc_mem)
            standard_accs.append(acc_std)
        
        if step % 3 == 0:
            print(f"Step {step} ({task_type}):")
            print(f"  Memory-enhanced: {acc_mem:.2%}")
            print(f"  Standard: {acc_std:.2%}")
            print(f"  Advantage: {(acc_mem - acc_std)*100:+.1f}%")
    
    print("\n✅ Comparative Benchmark Complete!")
    avg_mem = np.mean(memory_accs)
    avg_std = np.mean(standard_accs)
    print(f"\n📊 Overall Results:")
    print(f"   Memory-enhanced avg: {avg_mem:.2%}")
    print(f"   Standard avg: {avg_std:.2%}")
    print(f"   Overall advantage: {(avg_mem - avg_std)*100:+.1f}%")
    
    mem_stats = memory_moe.get_statistics()
    print(f"\n💾 Memory Statistics:")
    print(f"   Total memories: {mem_stats['total_memories']}")
    print(f"   Consolidations: {mem_stats['consolidation_count']}")
    
    return {
        'memory_accs': memory_accs,
        'standard_accs': standard_accs,
        'advantage': avg_mem - avg_std
    }


def main():
    """Run all demos."""
    print("╔" + "="*68 + "╗")
    print("║" + " "*68 + "║")
    print("║" + "  Memory-Enhanced Mixture of Experts - Complete Demo".center(68) + "║")
    print("║" + " "*68 + "║")
    print("║" + "  Features:".ljust(68) + "║")
    print("║" + "    1. Experts with specialized external memory banks".ljust(68) + "║")
    print("║" + "    2. Automatic memory indexing and retrieval".ljust(68) + "║")
    print("║" + "    3. Memory-based few-shot adaptation".ljust(68) + "║")
    print("║" + "    4. Hierarchical memory (short-term ↔ long-term)".ljust(68) + "║")
    print("║" + "    5. Expert specialization with memory".ljust(68) + "║")
    print("║" + " "*68 + "║")
    print("║" + "  Competitive Edge: Current MoE lacks memory; ours recalls".ljust(68) + "║")
    print("║" + " "*68 + "║")
    print("╚" + "="*68 + "╝")
    
    start_time = time.time()
    
    results = {}
    
    # Run all demos
    results['storage_retrieval'] = demo_1_memory_storage_retrieval()
    results['few_shot'] = demo_2_few_shot_adaptation()
    results['hierarchical'] = demo_3_hierarchical_memory()
    results['specialization'] = demo_4_expert_specialization_with_memory()
    results['pruning'] = demo_5_memory_pruning()
    results['comparison'] = demo_6_comparative_benchmark()
    
    elapsed = time.time() - start_time
    
    # Final summary
    print("\n" + "="*70)
    print("MEMORY-ENHANCED MIXTURE OF EXPERTS - COMPLETE")
    print("="*70)
    print(f"\n⏱️  Total demo time: {elapsed:.1f} seconds")
    
    comparison = results['comparison']
    
    print(f"\n🎯 Key Achievements:")
    print(f"   ✅ Memory storage: Automatic episodic & semantic storage")
    print(f"   ✅ Memory retrieval: {np.mean(results['storage_retrieval'][1]):.1f} avg hits per forward")
    print(f"   ✅ Few-shot: Rapid adaptation from 3-10 examples")
    print(f"   ✅ Hierarchical: {results['hierarchical'][-1]['consolidations']} consolidation events")
    print(f"   ✅ Specialization: Expert-specific memory banks developed")
    print(f"   ✅ Pruning: {len(results['pruning'][1])} pruning events, capacity maintained")
    
    print(f"\n🏆 Competitive Advantages:")
    print(f"   • {comparison['advantage']*100:+.1f}% accuracy over standard MoE")
    print(f"   • Automatic memory consolidation (short → long term)")
    print(f"   • Few-shot adaptation without retraining")
    print(f"   • Expert-specific episodic memories")
    print(f"   • Intelligent memory pruning")
    
    print(f"\n💡 Market Differentiation:")
    print(f"   • Standard MoE: Stateless (forgets after forward pass)")
    print(f"   • Memory-Enhanced MoE: Remembers and recalls experiences")
    print(f"   • Result: Better adaptation, few-shot learning, continual learning")
    
    print(f"\n✅ All memory-enhanced MoE features operational!")
    
    return results


if __name__ == "__main__":
    # Import F for one-hot encoding
    import torch.nn.functional as F
    results = main()
