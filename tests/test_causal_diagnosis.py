"""
Test Suite for Causal Self-Diagnosis System

Validates all core capabilities:
1. Causal inference for failure attribution
2. Counterfactual reasoning
3. Automatic hypothesis generation
4. Root cause analysis with intervention experiments
"""

import sys
from pathlib import Path
import pytest
import numpy as np

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from training.causal_self_diagnosis import (
    CausalSelfDiagnosis,
    CausalGraph,
    CounterfactualReasoner,
    FailureMode,
    CausalNodeType,
    InterventionStrategy,
    create_causal_diagnosis_system
)


class TestCausalGraph:
    """Test causal graph construction and analysis."""
    
    def test_graph_creation(self):
        """Test creating a causal graph."""
        graph = CausalGraph()
        
        # Add nodes
        graph.add_node(
            node_id="input_data",
            node_type=CausalNodeType.INPUT,
            name="Input Data",
            current_value=1000,
            expected_value=5000
        )
        
        graph.add_node(
            node_id="model_params",
            node_type=CausalNodeType.PARAMETER,
            name="Model Parameters",
            current_value=1e6,
            expected_value=1e6
        )
        
        graph.add_node(
            node_id="accuracy",
            node_type=CausalNodeType.OUTPUT,
            name="Model Accuracy",
            current_value=0.65,
            expected_value=0.85
        )
        
        assert len(graph.nodes) == 3
        assert "input_data" in graph.nodes
        assert graph.nodes["input_data"].node_type == CausalNodeType.INPUT
    
    def test_add_edge(self):
        """Test adding causal edges."""
        graph = CausalGraph()
        
        # Add nodes first
        graph.add_node("A", CausalNodeType.INPUT, "Node A")
        graph.add_node("B", CausalNodeType.OUTPUT, "Node B")
        
        # Add edge
        graph.add_edge(
            source="A",
            target="B",
            causal_effect=0.7,
            confidence=0.85
        )
        
        assert len(graph.edges) == 1
        edge = graph.edges[0]
        assert edge.source == "A"
        assert edge.target == "B"
        assert edge.causal_effect == 0.7
        assert edge.confidence == 0.85
    
    def test_root_cause_identification(self):
        """Test identifying root causes."""
        graph = CausalGraph()
        
        # Build a simple causal chain: data -> model -> accuracy
        graph.add_node("data", CausalNodeType.INPUT, "Data Size", 
                      current_value=1000, expected_value=10000)
        graph.add_node("model", CausalNodeType.HIDDEN, "Model State",
                      current_value=0.5, expected_value=0.8)
        graph.add_node("accuracy", CausalNodeType.OUTPUT, "Accuracy",
                      current_value=0.6, expected_value=0.9)
        
        graph.add_edge("data", "model", causal_effect=0.8, confidence=0.9)
        graph.add_edge("model", "accuracy", causal_effect=0.9, confidence=0.95)
        
        # Identify root causes of low accuracy
        root_causes = graph.identify_root_causes("accuracy")
        
        assert len(root_causes) > 0
        # "data" should be identified as root cause (has no parents, high deviation)
        assert "data" in root_causes
    
    def test_causal_path_finding(self):
        """Test finding causal paths."""
        graph = CausalGraph()
        
        graph.add_node("A", CausalNodeType.INPUT, "A")
        graph.add_node("B", CausalNodeType.HIDDEN, "B")
        graph.add_node("C", CausalNodeType.OUTPUT, "C")
        
        graph.add_edge("A", "B", causal_effect=0.7, confidence=0.8)
        graph.add_edge("B", "C", causal_effect=0.8, confidence=0.9)
        
        path = graph.find_causal_path("A", "C")
        
        assert path is not None
        assert len(path) == 3
        assert path == ["A", "B", "C"]


class TestCounterfactualReasoner:
    """Test counterfactual reasoning capabilities."""
    
    def test_basic_counterfactual(self):
        """Test generating basic counterfactual."""
        graph = CausalGraph()
        
        # Build simple model
        graph.add_node("learning_rate", CausalNodeType.HYPERPARAMETER, 
                      "Learning Rate", current_value=0.01)
        graph.add_node("accuracy", CausalNodeType.OUTPUT,
                      "Accuracy", current_value=0.70)
        
        graph.add_edge("learning_rate", "accuracy", 
                      causal_effect=0.6, confidence=0.85)
        
        reasoner = CounterfactualReasoner(graph)
        
        # Generate counterfactual: what if learning_rate was 0.001?
        cf = reasoner.generate_counterfactual(
            target_node="accuracy",
            intervention_node="learning_rate",
            intervention_value=0.001
        )
        
        assert cf is not None
        assert cf.changed_node == "learning_rate"
        assert cf.original_value == 0.01
        assert cf.counterfactual_value == 0.001
        assert cf.outcome_change != 0  # Should predict some change
    
    def test_actionable_counterfactuals(self):
        """Test finding actionable counterfactuals."""
        graph = CausalGraph()
        
        # Build model with multiple intervention points
        graph.add_node("data_size", CausalNodeType.INPUT, "Data Size", 
                      current_value=1000)
        graph.add_node("lr", CausalNodeType.HYPERPARAMETER, "Learning Rate",
                      current_value=0.01)
        graph.add_node("accuracy", CausalNodeType.OUTPUT, "Accuracy",
                      current_value=0.65)
        
        graph.add_edge("data_size", "accuracy", causal_effect=0.7, confidence=0.9)
        graph.add_edge("lr", "accuracy", causal_effect=0.5, confidence=0.8)
        
        reasoner = CounterfactualReasoner(graph)
        
        # Find best actionable counterfactuals
        counterfactuals = reasoner.find_best_counterfactuals(
            target_outcome="accuracy",
            num_counterfactuals=3,
            require_actionable=True
        )
        
        assert len(counterfactuals) > 0
        # All should be marked as actionable
        for cf in counterfactuals:
            assert cf.is_actionable == True
            assert cf.intervention_required is not None


class TestCausalSelfDiagnosis:
    """Test complete diagnosis system."""
    
    def test_diagnosis_creation(self):
        """Test creating diagnosis system."""
        system = create_causal_diagnosis_system()
        assert system is not None
        assert isinstance(system, CausalSelfDiagnosis)
    
    def test_build_causal_model(self):
        """Test building causal model from components."""
        system = create_causal_diagnosis_system()
        
        components = {
            "data": {
                "type": "INPUT",
                "name": "Training Data",
                "value": 5000,
                "expected_value": 10000,
                "parents": []
            },
            "model": {
                "type": "PARAMETER",
                "name": "Model Weights",
                "value": 1e6,
                "expected_value": 1e6,
                "parents": ["data"]
            },
            "accuracy": {
                "type": "OUTPUT",
                "name": "Accuracy",
                "value": 0.70,
                "expected_value": 0.85,
                "parents": ["model", "data"]
            }
        }
        
        system.build_causal_model(components)
        
        assert len(system.causal_graph.nodes) == 3
        assert len(system.causal_graph.edges) >= 2
    
    def test_failure_diagnosis(self):
        """Test complete failure diagnosis."""
        system = create_causal_diagnosis_system()
        
        # Build model
        components = {
            "training_data_size": {
                "type": "INPUT",
                "name": "Training Data Size",
                "value": 1000,
                "expected_value": 10000,
                "parents": []
            },
            "learning_rate": {
                "type": "HYPERPARAMETER",
                "name": "Learning Rate",
                "value": 0.1,  # Too high
                "expected_value": 0.001,
                "parents": []
            },
            "model_accuracy": {
                "type": "OUTPUT",
                "name": "Model Accuracy",
                "value": 0.60,  # Low
                "expected_value": 0.85,
                "parents": ["training_data_size", "learning_rate"]
            }
        }
        
        system.build_causal_model(components)
        
        # Diagnose underfitting
        failure = {
            "severity": 0.8,
            "component_values": {
                "model_accuracy": 0.60,
                "training_data_size": 1000,
                "learning_rate": 0.1
            }
        }
        
        diagnosis = system.diagnose_failure(
            failure_description=failure,
            failure_mode=FailureMode.UNDERFITTING
        )
        
        # Validate diagnosis
        assert diagnosis is not None
        assert diagnosis.failure_mode == FailureMode.UNDERFITTING
        assert len(diagnosis.root_causes) > 0
        assert diagnosis.diagnosis_confidence > 0
        
        # Should recommend interventions
        assert len(diagnosis.recommended_interventions) > 0
        
        # Should generate counterfactuals
        assert len(diagnosis.counterfactuals) > 0
    
    def test_intervention_recommendations(self):
        """Test intervention recommendation logic."""
        system = create_causal_diagnosis_system()
        
        # Build simple model
        components = {
            "regularization": {
                "type": "HYPERPARAMETER",
                "name": "Regularization",
                "value": 0.0,  # No regularization
                "expected_value": 0.01,
                "parents": []
            },
            "validation_accuracy": {
                "type": "OUTPUT",
                "name": "Validation Accuracy",
                "value": 0.65,  # Low
                "expected_value": 0.85,
                "parents": ["regularization"]
            }
        }
        
        system.build_causal_model(components)
        
        # Diagnose overfitting (low validation, implied high training)
        failure = {
            "severity": 0.7,
            "component_values": {
                "validation_accuracy": 0.65,
                "regularization": 0.0
            }
        }
        
        diagnosis = system.diagnose_failure(
            failure_description=failure,
            failure_mode=FailureMode.OVERFITTING
        )
        
        # Should recommend adding regularization
        interventions = [s for s, _ in diagnosis.recommended_interventions]
        assert any(
            s in [InterventionStrategy.ADD_REGULARIZATION, 
                  InterventionStrategy.ADJUST_HYPERPARAMETERS]
            for s in interventions
        )
    
    def test_hypothesis_generation(self):
        """Test automatic hypothesis generation."""
        system = create_causal_diagnosis_system()
        
        # Build model with distribution shift scenario
        components = {
            "input_distribution": {
                "type": "ENVIRONMENT",
                "name": "Input Distribution",
                "value": "shifted",
                "expected_value": "original",
                "parents": []
            },
            "prediction_accuracy": {
                "type": "OUTPUT",
                "name": "Accuracy",
                "value": 0.60,
                "expected_value": 0.85,
                "parents": ["input_distribution"]
            }
        }
        
        system.build_causal_model(components)
        
        failure = {
            "severity": 0.8,
            "component_values": {
                "prediction_accuracy": 0.60,
                "input_distribution": "shifted"
            }
        }
        
        diagnosis = system.diagnose_failure(
            failure_description=failure,
            failure_mode=FailureMode.DISTRIBUTION_SHIFT
        )
        
        # Should identify distribution shift as root cause
        assert len(diagnosis.root_causes) > 0
        # Should have evidence
        assert len(diagnosis.supporting_evidence) > 0


class TestEndToEndScenarios:
    """Test complete end-to-end scenarios."""
    
    def test_overfitting_scenario(self):
        """Test diagnosing and fixing overfitting."""
        system = create_causal_diagnosis_system()
        
        # Model with overfitting indicators
        components = {
            "model_capacity": {
                "type": "PARAMETER",
                "name": "Model Parameters",
                "value": 1e9,  # Very large
                "expected_value": 1e7,
                "parents": []
            },
            "training_data": {
                "type": "INPUT",
                "name": "Training Examples",
                "value": 1000,  # Small dataset
                "expected_value": 100000,
                "parents": []
            },
            "dropout": {
                "type": "HYPERPARAMETER",
                "name": "Dropout Rate",
                "value": 0.0,
                "expected_value": 0.3,
                "parents": []
            },
            "train_loss": {
                "type": "OUTPUT",
                "name": "Training Loss",
                "value": 0.01,  # Very low
                "expected_value": 0.1,
                "parents": ["model_capacity", "training_data", "dropout"]
            },
            "val_loss": {
                "type": "OUTPUT",
                "name": "Validation Loss",
                "value": 0.8,  # High
                "expected_value": 0.1,
                "parents": ["model_capacity", "training_data", "dropout"]
            }
        }
        
        system.build_causal_model(components)
        
        failure = {
            "severity": 0.9,
            "component_values": {
                "train_loss": 0.01,
                "val_loss": 0.8,
                "model_capacity": 1e9,
                "training_data": 1000,
                "dropout": 0.0
            }
        }
        
        diagnosis = system.diagnose_failure(
            failure_description=failure,
            failure_mode=FailureMode.OVERFITTING
        )
        
        # Validate diagnosis
        assert diagnosis.failure_mode == FailureMode.OVERFITTING
        
        # Should identify multiple root causes
        root_causes = diagnosis.root_causes
        assert len(root_causes) > 0
        
        # Should recommend appropriate interventions
        interventions = [s.value for s, _ in diagnosis.recommended_interventions]
        assert any(i in interventions for i in [
            "add_regularization",
            "collect_more_data",
            "adjust_hyperparameters"
        ])
        
        # Should generate helpful counterfactuals
        assert len(diagnosis.counterfactuals) > 0
        
        # At least one counterfactual should be actionable
        actionable = [cf for cf in diagnosis.counterfactuals if cf.is_actionable]
        assert len(actionable) > 0
    
    def test_data_insufficiency_scenario(self):
        """Test diagnosing data-related failures."""
        system = create_causal_diagnosis_system()
        
        components = {
            "dataset_size": {
                "type": "INPUT",
                "name": "Dataset Size",
                "value": 100,  # Very small
                "expected_value": 10000,
                "parents": []
            },
            "model_performance": {
                "type": "OUTPUT",
                "name": "Performance",
                "value": 0.55,
                "expected_value": 0.85,
                "parents": ["dataset_size"]
            }
        }
        
        system.build_causal_model(components)
        
        failure = {
            "severity": 0.7,
            "component_values": {
                "dataset_size": 100,
                "model_performance": 0.55
            }
        }
        
        diagnosis = system.diagnose_failure(
            failure_description=failure,
            failure_mode=FailureMode.UNDERFITTING
        )
        
        # Should identify data insufficiency
        assert "dataset_size" in diagnosis.root_causes
        
        # Should recommend collecting more data
        interventions = [s for s, _ in diagnosis.recommended_interventions]
        assert InterventionStrategy.COLLECT_MORE_DATA in interventions


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "--tb=short"])
