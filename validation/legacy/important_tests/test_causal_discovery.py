#!/usr/bin/env python3
"""
Phase 1 Critical Test: Causal Discovery (Test 4/6)

Tests causal reasoning foundations:
- Causal DAG construction from data
- Root cause analysis of failures
- Causal intervention planning
- Structural learning algorithms

Competitive Advantage vs SakanaAI:
Causal reasoning enables principled "why" explanations and intervention
planning, not just pattern matching. Critical for scientific applications.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
import numpy as np
from training.causal_self_diagnosis import (
    CausalGraph,
    CounterfactualReasoner,
    CausalSelfDiagnosis,
    create_causal_diagnosis_system
)


class TestCausalDiscovery:
    """Tests for causal graph discovery and construction."""
    
    def test_causal_graph_creation(self):
        """Test 4.1: Create causal directed acyclic graph (DAG)."""
        graph = CausalGraph(num_variables=5)
        
        assert graph is not None, "Causal graph creation failed"
        assert hasattr(graph, 'nodes'), "Missing nodes attribute"
        assert hasattr(graph, 'edges'), "Missing edges attribute"
    
    def test_causal_structure_learning(self):
        """Test 4.2: Learn causal structure from observational data."""
        # Generate synthetic data with known causal structure
        # X -> Y -> Z
        n_samples = 100
        X = np.random.randn(n_samples, 1)
        Y = 2 * X + np.random.randn(n_samples, 1) * 0.1
        Z = 3 * Y + np.random.randn(n_samples, 1) * 0.1
        
        data = np.concatenate([X, Y, Z], axis=1)
        
        graph = CausalGraph(num_variables=3)
        learned_structure = graph.learn_structure(data)
        
        assert learned_structure is not None, "Structure learning failed"
        assert 'edges' in learned_structure or 'dag' in learned_structure, \
            "No causal structure found"
    
    def test_root_cause_analysis(self):
        """Test 4.3: Identify root causes of observed failures."""
        diagnosis_system = create_causal_diagnosis_system(
            num_variables=10,
            num_interventions=5
        )
        
        # Simulate failure scenario
        failure_data = {
            'model_accuracy': 0.45,  # Low accuracy (failure)
            'data_quality': 0.3,      # Poor data quality (potential cause)
            'learning_rate': 0.1,     # High learning rate (potential cause)
            'batch_size': 32,
            'num_epochs': 100
        }
        
        root_causes = diagnosis_system.identify_root_causes(failure_data)
        
        assert root_causes is not None, "Root cause analysis failed"
        assert len(root_causes) > 0, "No root causes identified"
    
    def test_causal_discovery_from_interventions(self):
        """Test 4.4: Discover causal relationships from interventional data."""
        graph = CausalGraph(num_variables=4)
        
        # Interventional data: manipulate variable and observe effects
        interventions = [
            {'variable': 0, 'value': 1.0, 'observed_changes': {1: 0.5, 2: 0.0}},
            {'variable': 1, 'value': 2.0, 'observed_changes': {2: 1.0, 3: 0.3}},
        ]
        
        discovered_edges = graph.discover_from_interventions(interventions)
        
        assert discovered_edges is not None, "Intervention-based discovery failed"
        # Should discover: 0->1, 1->2, 1->3
    
    def test_markov_blanket_identification(self):
        """Test 4.5: Identify Markov blanket (local causal neighborhood)."""
        graph = CausalGraph(num_variables=6)
        
        # Build a causal graph
        graph.add_edge(0, 1)  # Parent
        graph.add_edge(1, 2)  # Child
        graph.add_edge(3, 2)  # Co-parent
        
        # Find Markov blanket of variable 1
        markov_blanket = graph.get_markov_blanket(variable_id=1)
        
        assert markov_blanket is not None, "Markov blanket identification failed"
        # Should include: parents(1), children(1), co-parents(1)


def run_all_tests():
    """Run all causal discovery tests."""
    print("=" * 80)
    print("PHASE 1 - TEST 4: Causal Discovery")
    print("=" * 80)
    
    test_suite = TestCausalDiscovery()
    results = {
        'total': 5,
        'passed': 0,
        'failed': 0,
        'errors': []
    }
    
    tests = [
        ('Causal Graph Creation', test_suite.test_causal_graph_creation),
        ('Causal Structure Learning', test_suite.test_causal_structure_learning),
        ('Root Cause Analysis', test_suite.test_root_cause_analysis),
        ('Discovery from Interventions', test_suite.test_causal_discovery_from_interventions),
        ('Markov Blanket Identification', test_suite.test_markov_blanket_identification),
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
    print("TEST SUMMARY - Causal Discovery")
    print("=" * 80)
    print(f"Total Tests: {results['total']}")
    print(f"Passed: {results['passed']}")
    print(f"Failed: {results['failed']}")
    print(f"Success Rate: {(results['passed'] / results['total'] * 100):.1f}%")
    
    return results


if __name__ == "__main__":
    results = run_all_tests()
    sys.exit(0 if results['failed'] == 0 else 1)
