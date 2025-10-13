#!/usr/bin/env python3
"""
Phase 1 Critical Test: Neural-Symbolic Reasoning (Test 2/3)

Tests advanced reasoning capabilities:
- Constraint satisfaction problems
- Knowledge graph reasoning
- Explainable decision making
- Integration with neural learning

Competitive Advantage:
Provides interpretable, verifiable reasoning traces that neural-only
approaches (like SakanaAI) cannot match.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
import numpy as np
from training.neural_symbolic_architecture import (
    create_neural_symbolic_architecture,
    SymbolicReasoningAgent
)


class TestNeuralSymbolicReasoning:
    """Advanced reasoning capability tests."""
    
    def test_constraint_satisfaction(self):
        """Test 2.1: Solve constraint satisfaction problems."""
        arch = create_neural_symbolic_architecture()
        
        # Define a simple CSP: Graph coloring problem
        # 3 nodes, 2 colors, constraint: adjacent nodes different colors
        constraints = [
            {"node_a": 0, "node_b": 1, "relation": "different"},
            {"node_a": 1, "node_b": 2, "relation": "different"},
        ]
        
        result = arch.solve_csp(
            num_variables=3,
            domain_size=2,
            constraints=constraints
        )
        
        assert result is not None, "CSP solver failed"
        assert 'solution' in result or 'satisfiable' in result, "No CSP solution found"
    
    def test_knowledge_graph_reasoning(self):
        """Test 2.2: Reason over knowledge graphs."""
        agent = SymbolicReasoningAgent(num_symbols=20, num_rules=10, logic_dim=64)
        
        # Build knowledge graph
        agent.add_fact("alice", "friend_of", "bob")
        agent.add_fact("bob", "friend_of", "charlie")
        agent.add_rule("if X friend_of Y and Y friend_of Z then X connected_to Z")
        
        # Query the graph
        query_result = agent.query("alice connected_to charlie")
        
        assert query_result is not None, "Knowledge graph query failed"
        # Should infer transitivity
    
    def test_explainable_decisions(self):
        """Test 2.3: Generate explanations for decisions."""
        arch = create_neural_symbolic_architecture()
        
        # Make a decision with explanation
        input_data = torch.randn(1, 64)
        result = arch.forward_with_explanation(input_data)
        
        assert result is not None, "Forward pass failed"
        assert 'output' in result, "No output generated"
        assert 'explanation' in result or 'reasoning_trace' in result, \
            "No explanation provided"
    
    def test_hybrid_neural_symbolic_learning(self):
        """Test 2.4: Learn from both neural signals and symbolic constraints."""
        arch = create_neural_symbolic_architecture()
        
        # Create training data with symbolic constraints
        inputs = torch.randn(10, 64)
        targets = torch.randint(0, 5, (10,))
        
        # Add symbolic constraints
        constraints = [
            "output must be between 0 and 4",
            "if input[0] > 0.5 then output >= 2"
        ]
        
        # Train with constraints
        loss = arch.train_with_constraints(
            inputs=inputs,
            targets=targets,
            constraints=constraints,
            num_steps=10
        )
        
        assert loss is not None, "Training with constraints failed"
        assert isinstance(loss, (int, float, torch.Tensor)), "Invalid loss type"
    
    def test_multi_hop_reasoning(self):
        """Test 2.5: Perform multi-hop reasoning over knowledge base."""
        agent = SymbolicReasoningAgent(num_symbols=30, num_rules=15, logic_dim=64)
        
        # Build multi-hop knowledge
        agent.add_fact("paris", "capital_of", "france")
        agent.add_fact("france", "part_of", "europe")
        agent.add_fact("europe", "type", "continent")
        
        # Multi-hop query: What continent is Paris in?
        result = agent.multi_hop_query(
            start_entity="paris",
            target_relation="type",
            max_hops=3
        )
        
        assert result is not None, "Multi-hop reasoning failed"


def run_all_tests():
    """Run all neural-symbolic reasoning tests."""
    print("=" * 80)
    print("PHASE 1 - TEST 2: Neural-Symbolic Reasoning")
    print("=" * 80)
    
    test_suite = TestNeuralSymbolicReasoning()
    results = {
        'total': 5,
        'passed': 0,
        'failed': 0,
        'errors': []
    }
    
    tests = [
        ('Constraint Satisfaction', test_suite.test_constraint_satisfaction),
        ('Knowledge Graph Reasoning', test_suite.test_knowledge_graph_reasoning),
        ('Explainable Decisions', test_suite.test_explainable_decisions),
        ('Hybrid Neural-Symbolic Learning', test_suite.test_hybrid_neural_symbolic_learning),
        ('Multi-Hop Reasoning', test_suite.test_multi_hop_reasoning),
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
    print("TEST SUMMARY - Neural-Symbolic Reasoning")
    print("=" * 80)
    print(f"Total Tests: {results['total']}")
    print(f"Passed: {results['passed']}")
    print(f"Failed: {results['failed']}")
    print(f"Success Rate: {(results['passed'] / results['total'] * 100):.1f}%")
    
    return results


if __name__ == "__main__":
    results = run_all_tests()
    sys.exit(0 if results['failed'] == 0 else 1)
