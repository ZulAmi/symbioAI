#!/usr/bin/env python3
"""
Phase 1 Critical Test: Causal Self-Diagnosis (Test 6/6)

Tests self-diagnosis and intervention planning:
- Automated failure diagnosis
- Intervention strategy recommendation
- Performance degradation detection
- Self-healing through causal interventions

Competitive Advantage:
Autonomous self-diagnosis and repair - system can fix itself by
understanding causal relationships, not just symptoms.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
import numpy as np
from training.causal_self_diagnosis import (
    CausalSelfDiagnosis,
    InterventionStrategy,
    create_causal_diagnosis_system
)


class TestCausalSelfDiagnosis:
    """Tests for autonomous self-diagnosis capabilities."""
    
    def test_automated_failure_diagnosis(self):
        """Test 6.1: Automatically diagnose performance failures."""
        diagnosis_system = create_causal_diagnosis_system()
        
        # Performance metrics showing failure
        metrics = {
            'train_accuracy': 0.95,
            'val_accuracy': 0.45,  # Overfitting!
            'train_loss': 0.1,
            'val_loss': 2.3,       # High validation loss
            'gradient_norm': 0.001  # Vanishing gradients
        }
        
        diagnosis = diagnosis_system.diagnose_failure(metrics)
        
        assert diagnosis is not None, "Failure diagnosis failed"
        assert 'diagnosis' in diagnosis or 'identified_issues' in diagnosis, \
            "No diagnosis produced"
        assert len(diagnosis.get('identified_issues', [])) > 0 or \
               diagnosis.get('diagnosis') is not None, \
            "No issues identified"
    
    def test_intervention_strategy_recommendation(self):
        """Test 6.2: Recommend intervention strategies to fix issues."""
        diagnosis_system = create_causal_diagnosis_system()
        
        # Diagnosed problem: overfitting
        problem = {
            'type': 'overfitting',
            'severity': 0.8,
            'affected_metrics': ['val_accuracy', 'val_loss']
        }
        
        # Get intervention recommendations
        interventions = diagnosis_system.recommend_interventions(problem)
        
        assert interventions is not None, "Intervention recommendation failed"
        assert len(interventions) > 0, "No interventions recommended"
        
        # Should suggest: regularization, dropout, early stopping, etc.
        recommended_strategies = [i.get('strategy') or i.get('type') 
                                 for i in interventions]
        assert any(s for s in recommended_strategies), "No strategy types found"
    
    def test_performance_degradation_detection(self):
        """Test 6.3: Detect performance degradation over time."""
        diagnosis_system = create_causal_diagnosis_system()
        
        # Historical performance trajectory
        performance_history = [
            {'epoch': 0, 'accuracy': 0.50},
            {'epoch': 10, 'accuracy': 0.75},
            {'epoch': 20, 'accuracy': 0.85},
            {'epoch': 30, 'accuracy': 0.87},
            {'epoch': 40, 'accuracy': 0.85},  # Starting to degrade
            {'epoch': 50, 'accuracy': 0.80},  # Degrading
        ]
        
        degradation = diagnosis_system.detect_degradation(performance_history)
        
        assert degradation is not None, "Degradation detection failed"
        assert 'is_degrading' in degradation or 'degradation_detected' in degradation, \
            "No degradation status"
    
    def test_self_healing_intervention(self):
        """Test 6.4: Execute self-healing interventions."""
        diagnosis_system = create_causal_diagnosis_system()
        
        # Current problematic state
        current_state = {
            'learning_rate': 0.1,     # Too high
            'gradient_norm': 15.0,    # Exploding gradients
            'loss': float('inf')      # Training collapse
        }
        
        # Diagnose and intervene
        diagnosis = diagnosis_system.diagnose_failure(current_state)
        intervention = diagnosis_system.auto_intervene(diagnosis)
        
        assert intervention is not None, "Auto-intervention failed"
        assert 'actions' in intervention or 'applied_fixes' in intervention, \
            "No intervention actions"
        
        # Should propose: reduce learning rate, gradient clipping, etc.
    
    def test_intervention_impact_tracking(self):
        """Test 6.5: Track impact of applied interventions."""
        diagnosis_system = create_causal_diagnosis_system()
        
        # State before intervention
        before = {
            'accuracy': 0.45,
            'loss': 2.5
        }
        
        # Applied intervention
        intervention = {
            'strategy': InterventionStrategy.ADD_REGULARIZATION,
            'parameters': {'l2_weight': 0.01}
        }
        
        # State after intervention
        after = {
            'accuracy': 0.75,
            'loss': 0.8
        }
        
        # Track impact
        impact = diagnosis_system.track_intervention_impact(
            before=before,
            intervention=intervention,
            after=after
        )
        
        assert impact is not None, "Impact tracking failed"
        assert 'improvement' in impact or 'effectiveness' in impact, \
            "No impact assessment"
        
        # Should show improvement metrics


def run_all_tests():
    """Run all causal self-diagnosis tests."""
    print("=" * 80)
    print("PHASE 1 - TEST 6: Causal Self-Diagnosis")
    print("=" * 80)
    
    test_suite = TestCausalSelfDiagnosis()
    results = {
        'total': 5,
        'passed': 0,
        'failed': 0,
        'errors': []
    }
    
    tests = [
        ('Automated Failure Diagnosis', test_suite.test_automated_failure_diagnosis),
        ('Intervention Strategy Recommendation', test_suite.test_intervention_strategy_recommendation),
        ('Performance Degradation Detection', test_suite.test_performance_degradation_detection),
        ('Self-Healing Intervention', test_suite.test_self_healing_intervention),
        ('Intervention Impact Tracking', test_suite.test_intervention_impact_tracking),
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
    print("TEST SUMMARY - Causal Self-Diagnosis")
    print("=" * 80)
    print(f"Total Tests: {results['total']}")
    print(f"Passed: {results['passed']}")
    print(f"Failed: {results['failed']}")
    print(f"Success Rate: {(results['passed'] / results['total'] * 100):.1f}%")
    
    return results


if __name__ == "__main__":
    results = run_all_tests()
    sys.exit(0 if results['failed'] == 0 else 1)
