#!/usr/bin/env python3
"""
Phase 1 Critical Test: Neural-Symbolic Integration (Test 1/3)

Tests core neural-symbolic architecture capabilities:
- Hybrid neural-symbolic reasoning
- Differentiable logic programming
- Program synthesis from natural language
- Knowledge base with facts and rules
- Proof generation and verification

Competitive Advantage vs SakanaAI:
SakanaAI focuses on pure neural approaches. SymbioAI combines neural learning
with symbolic reasoning for explainable, verifiable AI decisions.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
import pytest
from training.neural_symbolic_architecture import (
    NeuralSymbolicArchitecture,
    SymbolicReasoningAgent,
    DifferentiableLogicNetwork,
    create_neural_symbolic_architecture
)


class TestNeuralSymbolicIntegration:
    """Critical tests for neural-symbolic integration."""
    
    def test_architecture_creation(self):
        """Test 1.1: Neural-symbolic architecture can be created."""
        arch = create_neural_symbolic_architecture(
            input_dim=64,
            hidden_dim=128,
            num_symbols=10,
            num_rules=5
        )
        
        assert arch is not None, "Architecture creation failed"
        assert isinstance(arch, NeuralSymbolicArchitecture), "Wrong type returned"
        assert arch.num_symbols == 10, "Symbol count mismatch"
        assert arch.num_rules == 5, "Rule count mismatch"
    
    def test_symbolic_reasoning_agent(self):
        """Test 1.2: Symbolic reasoning agent can perform logical operations."""
        agent = SymbolicReasoningAgent(
            num_symbols=10,
            num_rules=5,
            logic_dim=64
        )
        
        # Test knowledge base initialization
        assert hasattr(agent, 'knowledge_base'), "Missing knowledge base"
        assert hasattr(agent, 'inference_engine'), "Missing inference engine"
        
        # Test fact addition
        agent.add_fact("sky", "is", "blue")
        assert len(agent.knowledge_base.facts) > 0, "Fact not added"
    
    def test_differentiable_logic_operations(self):
        """Test 1.3: Differentiable logic can perform AND, OR, NOT, IMPLIES."""
        logic_net = DifferentiableLogicNetwork(logic_dim=64)
        
        # Create test propositions
        prop_a = torch.tensor([1.0])  # True
        prop_b = torch.tensor([0.0])  # False
        
        # Test AND operation (fuzzy logic)
        and_result = logic_net.fuzzy_and(prop_a, prop_b)
        assert 0.0 <= and_result.item() <= 1.0, "AND result out of bounds"
        assert and_result.item() < prop_a.item(), "AND should be less than True AND False"
        
        # Test OR operation
        or_result = logic_net.fuzzy_or(prop_a, prop_b)
        assert 0.0 <= or_result.item() <= 1.0, "OR result out of bounds"
        assert or_result.item() > prop_b.item(), "OR should be greater than False"
        
        # Test NOT operation
        not_result = logic_net.fuzzy_not(prop_a)
        assert 0.0 <= not_result.item() <= 1.0, "NOT result out of bounds"
    
    def test_program_synthesis(self):
        """Test 1.4: Can synthesize programs from natural language."""
        arch = create_neural_symbolic_architecture()
        
        # Test program synthesis capability
        task_description = "Sort a list of numbers in ascending order"
        result = arch.synthesize_program(task_description)
        
        assert result is not None, "Program synthesis failed"
        assert 'program' in result or 'code' in result or 'solution' in result, \
            "No program generated"
    
    def test_proof_generation(self):
        """Test 1.5: Can generate logical proofs."""
        agent = SymbolicReasoningAgent(num_symbols=10, num_rules=5, logic_dim=64)
        
        # Add logical facts and rules
        agent.add_fact("socrates", "is", "human")
        agent.add_rule("if X is human then X is mortal")
        
        # Generate proof
        proof = agent.generate_proof(goal="socrates is mortal")
        
        assert proof is not None, "Proof generation failed"
        assert len(proof) > 0, "Empty proof generated"


def run_all_tests():
    """Run all neural-symbolic integration tests."""
    print("=" * 80)
    print("PHASE 1 - TEST 1: Neural-Symbolic Integration")
    print("=" * 80)
    
    test_suite = TestNeuralSymbolicIntegration()
    results = {
        'total': 5,
        'passed': 0,
        'failed': 0,
        'errors': []
    }
    
    tests = [
        ('Architecture Creation', test_suite.test_architecture_creation),
        ('Symbolic Reasoning Agent', test_suite.test_symbolic_reasoning_agent),
        ('Differentiable Logic', test_suite.test_differentiable_logic_operations),
        ('Program Synthesis', test_suite.test_program_synthesis),
        ('Proof Generation', test_suite.test_proof_generation),
    ]
    
    for test_name, test_func in tests:
        try:
            print(f"\n🧪 Running: {test_name}...")
            test_func()
            print(f"✅ PASSED: {test_name}")
            results['passed'] += 1
        except Exception as e:
            print(f"❌ FAILED: {test_name}")
            print(f"   Error: {str(e)}")
            results['failed'] += 1
            results['errors'].append({
                'test': test_name,
                'error': str(e)
            })
    
    # Print summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY - Neural-Symbolic Integration")
    print("=" * 80)
    print(f"Total Tests: {results['total']}")
    print(f"Passed: {results['passed']}")
    print(f"Failed: {results['failed']}")
    print(f"Success Rate: {(results['passed'] / results['total'] * 100):.1f}%")
    
    return results


if __name__ == "__main__":
    results = run_all_tests()
    sys.exit(0 if results['failed'] == 0 else 1)
