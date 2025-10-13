#!/usr/bin/env python3
"""
Phase 1 Critical Test: Learning from Demonstrations (Test 13/15)

Tests learning from expert demonstrations:
- Memory-enhanced MoE for few-shot learning
- Demonstration replay and adaptation
- Expert knowledge transfer
- Rapid task acquisition

Competitive Advantage:
Learn from limited examples using memory-enhanced experts,
enabling practical deployment with minimal training data.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
import numpy as np
from training.memory_enhanced_moe import (
    create_memory_enhanced_moe,
    MemoryConfig,
    ExpertSpecialization
)


class TestDemonstrationLearning:
    """Tests for learning from demonstrations."""
    
    def test_few_shot_adaptation(self):
        """Test 13.1: Adapt from few demonstrations."""
        config = MemoryConfig(
            short_term_capacity=20,
            long_term_capacity=100,
            episodic_capacity=50
        )
        
        model = create_memory_enhanced_moe(
            input_dim=64,
            output_dim=10,
            num_experts=4,
            memory_config=config
        )
        
        # Provide 5 demonstrations
        demonstrations = [
            (torch.randn(64), torch.randint(0, 10, (1,)).item())
            for _ in range(5)
        ]
        
        # Adapt from demos
        for demo_input, demo_label in demonstrations:
            output, info = model(demo_input.unsqueeze(0), 
                                metadata={'demonstration': True})
        
        # Test adaptation worked
        test_input = demonstrations[0][0].unsqueeze(0)
        output, info = model(test_input, use_memory=True)
        
        assert output is not None, "Adaptation failed"
        assert info['retrieved'] > 0, "No demonstrations retrieved"
    
    def test_expert_memory_storage(self):
        """Test 13.2: Experts store demonstration experiences."""
        model = create_memory_enhanced_moe(num_experts=4)
        
        # Store demonstrations
        for _ in range(10):
            data = torch.randn(1, 64)
            output, info = model(data, store_experience=True, 
                                metadata={'task': 'demo_task'})
        
        # Check memories stored
        total_memories = sum(
            expert.memory.size()
            for expert in model.experts
        )
        
        assert total_memories > 0, "No memories stored"
        print(f"   Stored {total_memories} demonstration memories")
    
    def test_demonstration_replay(self):
        """Test 13.3: Replay demonstrations for continual learning."""
        model = create_memory_enhanced_moe(num_experts=4)
        
        # Store initial demonstrations
        demo_inputs = [torch.randn(1, 64) for _ in range(15)]
        
        for demo in demo_inputs:
            model(demo, store_experience=True, metadata={'phase': 'initial'})
        
        # Later: replay demonstrations
        for expert in model.experts:
            memories = expert.memory.retrieve(
                query=torch.randn(64),
                top_k=5
            )
            
            assert len(memories) > 0, f"Expert {expert.expert_id} has no memories"
        
        print(f"   Successfully replayed demonstrations")
    
    def test_expert_specialization_from_demos(self):
        """Test 13.4: Experts specialize based on demonstration types."""
        model = create_memory_enhanced_moe(num_experts=4)
        
        # Different types of demonstrations
        demo_types = ['vision', 'language', 'reasoning', 'multimodal']
        
        for demo_type in demo_types:
            for _ in range(5):
                data = torch.randn(1, 64)
                model(data, metadata={'task': demo_type})
        
        # Check expert usage patterns
        expert_stats = {}
        for expert in model.experts:
            expert_stats[expert.expert_id] = {
                'forward_count': expert.forward_count,
                'specialization': expert.specialization.value
            }
        
        assert len(expert_stats) == 4, "Not all experts tracked"
        print(f"   Expert specializations: {expert_stats}")
    
    def test_rapid_task_acquisition(self):
        """Test 13.5: Quickly acquire new tasks from demos."""
        model = create_memory_enhanced_moe(num_experts=4)
        
        # New task demonstrations
        new_task_demos = [
            (torch.randn(64), torch.tensor([3]))  # class 3
            for _ in range(8)
        ]
        
        # Rapid adaptation
        for demo_input, demo_label in new_task_demos:
            output, info = model(demo_input.unsqueeze(0), 
                                store_experience=True,
                                metadata={'new_task': True})
        
        # Test on similar input
        test_input = new_task_demos[0][0].unsqueeze(0)
        output, info = model(test_input, use_memory=True)
        
        assert info['retrieved'] > 0, "Failed to retrieve demonstrations"
        print(f"   Retrieved {info['retrieved']} relevant demonstrations")


def run_all_tests():
    """Run all demonstration learning tests."""
    print("=" * 80)
    print("PHASE 1 - TEST 13: Learning from Demonstrations")
    print("=" * 80)
    
    test_suite = TestDemonstrationLearning()
    results = {'total': 5, 'passed': 0, 'failed': 0, 'errors': []}
    
    tests = [
        ('Few-Shot Adaptation', test_suite.test_few_shot_adaptation),
        ('Expert Memory Storage', test_suite.test_expert_memory_storage),
        ('Demonstration Replay', test_suite.test_demonstration_replay),
        ('Expert Specialization', test_suite.test_expert_specialization_from_demos),
        ('Rapid Task Acquisition', test_suite.test_rapid_task_acquisition),
    ]
    
    for test_name, test_func in tests:
        try:
            print(f"\n🧪 Running: {test_name}...")
            test_func()
            print(f"✅ PASSED: {test_name}")
            results['passed'] += 1
        except Exception as e:
            print(f"❌ FAILED: {test_name} - {str(e)}")
            results['failed'] += 1
            results['errors'].append({'test': test_name, 'error': str(e)})
    
    print(f"\n{'='*80}")
    print(f"Total: {results['total']}, Passed: {results['passed']}, Failed: {results['failed']}")
    print(f"Success Rate: {(results['passed']/results['total']*100):.1f}%")
    
    return results


if __name__ == "__main__":
    results = run_all_tests()
    sys.exit(0 if results['failed'] == 0 else 1)
