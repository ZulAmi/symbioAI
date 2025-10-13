#!/usr/bin/env python3
"""
Phase 1 Critical Test: Counterfactual Reasoning (Test 5/6)

Tests "what-if" counterfactual reasoning:
- Counterfactual generation and analysis
- Intervention effect prediction
- Abductive reasoning (finding explanations)
- Causal attribution

Competitive Advantage:
Enables "what would have happened if..." reasoning for debugging,
optimization, and decision making - unique capability vs competitors.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
import numpy as np
from training.causal_self_diagnosis import (
    CounterfactualReasoner,
    CausalGraph,
    create_causal_diagnosis_system
)


class TestCounterfactualReasoning:
    """Tests for counterfactual 'what-if' reasoning."""
    
    def test_counterfactual_generation(self):
        """Test 5.1: Generate counterfactual scenarios."""
        reasoner = CounterfactualReasoner(num_variables=5)
        
        # Actual observation
        actual_scenario = {
            'learning_rate': 0.01,
            'batch_size': 32,
            'accuracy': 0.85
        }
        
        # Generate counterfactual: what if learning_rate was 0.001?
        counterfactual = reasoner.generate_counterfactual(
            actual=actual_scenario,
            intervention={'learning_rate': 0.001}
        )
        
        assert counterfactual is not None, "Counterfactual generation failed"
        assert 'accuracy' in counterfactual, "No outcome prediction"
    
    def test_intervention_effect_prediction(self):
        """Test 5.2: Predict effects of interventions."""
        graph = CausalGraph(num_variables=4)
        
        # Build causal model: hyperparams -> model -> accuracy
        graph.add_edge(0, 2)  # learning_rate -> model_weights
        graph.add_edge(1, 2)  # data_quality -> model_weights
        graph.add_edge(2, 3)  # model_weights -> accuracy
        
        reasoner = CounterfactualReasoner(num_variables=4, causal_graph=graph)
        
        # Predict: what if we improve data_quality from 0.6 to 0.9?
        effect = reasoner.predict_intervention_effect(
            intervention={'variable': 1, 'from': 0.6, 'to': 0.9},
            target_variable=3  # accuracy
        )
        
        assert effect is not None, "Intervention effect prediction failed"
        assert 'expected_change' in effect or 'predicted_outcome' in effect, \
            "No effect estimate"
    
    def test_abductive_reasoning(self):
        """Test 5.3: Find explanations for observed outcomes (abduction)."""
        diagnosis_system = create_causal_diagnosis_system()
        
        # Observed: model performance suddenly dropped
        observation = {
            'accuracy': 0.45,  # Was 0.85, now 0.45
            'loss': 2.5,       # Was 0.3, now 2.5
        }
        
        # Find explanation: what could have caused this?
        explanations = diagnosis_system.find_explanations(observation)
        
        assert explanations is not None, "Abductive reasoning failed"
        assert len(explanations) > 0, "No explanations found"
    
    def test_causal_attribution(self):
        """Test 5.4: Attribute outcomes to specific causal factors."""
        reasoner = CounterfactualReasoner(num_variables=6)
        
        # Outcome to explain
        outcome = {
            'final_accuracy': 0.92
        }
        
        # Potential causes
        factors = {
            'architecture_choice': 'transformer',
            'data_augmentation': True,
            'learning_rate_schedule': 'cosine',
            'batch_size': 64
        }
        
        # Attribute contribution of each factor
        attributions = reasoner.attribute_outcome(
            outcome=outcome,
            factors=factors
        )
        
        assert attributions is not None, "Causal attribution failed"
        # Should quantify: how much did each factor contribute?
    
    def test_counterfactual_fairness(self):
        """Test 5.5: Check counterfactual fairness of decisions."""
        reasoner = CounterfactualReasoner(num_variables=5)
        
        # Decision scenario
        decision = {
            'applicant': 'A',
            'score': 0.75,
            'decision': 'accept',
            'protected_attribute': 'group_1'
        }
        
        # Check: would decision change if protected attribute was different?
        fairness_check = reasoner.check_counterfactual_fairness(
            decision=decision,
            protected_attribute='protected_attribute',
            alternative_value='group_2'
        )
        
        assert fairness_check is not None, "Fairness check failed"
        assert 'is_fair' in fairness_check or 'discriminatory' in fairness_check, \
            "No fairness assessment"


def run_all_tests():
    """Run all counterfactual reasoning tests."""
    print("=" * 80)
    print("PHASE 1 - TEST 5: Counterfactual Reasoning")
    print("=" * 80)
    
    test_suite = TestCounterfactualReasoning()
    results = {
        'total': 5,
        'passed': 0,
        'failed': 0,
        'errors': []
    }
    
    tests = [
        ('Counterfactual Generation', test_suite.test_counterfactual_generation),
        ('Intervention Effect Prediction', test_suite.test_intervention_effect_prediction),
        ('Abductive Reasoning', test_suite.test_abductive_reasoning),
        ('Causal Attribution', test_suite.test_causal_attribution),
        ('Counterfactual Fairness', test_suite.test_counterfactual_fairness),
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
    print("TEST SUMMARY - Counterfactual Reasoning")
    print("=" * 80)
    print(f"Total Tests: {results['total']}")
    print(f"Passed: {results['passed']}")
    print(f"Failed: {results['failed']}")
    print(f"Success Rate: {(results['passed'] / results['total'] * 100):.1f}%")
    
    return results


if __name__ == "__main__":
    results = run_all_tests()
    sys.exit(0 if results['failed'] == 0 else 1)
