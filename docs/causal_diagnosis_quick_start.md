# Causal Self-Diagnosis System - Quick Start Guide

## 🚀 Getting Started in 5 Minutes

### Installation

```bash
# Install dependencies
pip install -r requirements.txt
```

### Basic Usage

```python
from training.causal_self_diagnosis import (
    create_causal_diagnosis_system,
    FailureMode,
    InterventionStrategy
)

# 1. Create diagnosis system
diagnosis_system = create_causal_diagnosis_system()

# 2. Define your system components
system_components = {
    "training_data_size": {
        "type": "INPUT",
        "name": "Training Data Size",
        "value": 5000,           # Current value
        "expected_value": 50000,  # Expected value
        "parents": []             # No upstream dependencies
    },
    "learning_rate": {
        "type": "HYPERPARAMETER",
        "name": "Learning Rate",
        "value": 0.01,
        "expected_value": 0.001,
        "parents": []
    },
    "model_accuracy": {
        "type": "OUTPUT",
        "name": "Model Accuracy",
        "value": 0.65,           # Current (low)
        "expected_value": 0.85,  # Target
        "parents": ["training_data_size", "learning_rate"]
    }
}

# 3. Build causal model
diagnosis_system.build_causal_model(system_components)

# 4. Diagnose a failure
failure_description = {
    "severity": 0.8,
    "component_values": {
        "model_accuracy": 0.65,
        "training_data_size": 5000,
        "learning_rate": 0.01
    }
}

diagnosis = diagnosis_system.diagnose_failure(
    failure_description=failure_description,
    failure_mode=FailureMode.UNDERFITTING
)

# 5. View results
print(f"Root Causes: {diagnosis.root_causes}")
print(f"Confidence: {diagnosis.diagnosis_confidence:.2%}")

print("\nRecommended Interventions:")
for strategy, score in diagnosis.recommended_interventions[:3]:
    print(f"  • {strategy.value}: {score:.2%}")

print("\nCounterfactual Scenarios:")
for cf in diagnosis.counterfactuals[:3]:
    print(f"  • {cf.description}")
    print(f"    Expected improvement: {cf.outcome_change:.2%}")
```

---

## 📚 Common Use Cases

### Use Case 1: Diagnosing Low Model Accuracy

```python
# Your model has 65% accuracy when you expected 85%

system = create_causal_diagnosis_system()

components = {
    "data_quality": {
        "type": "INPUT",
        "name": "Data Quality Score",
        "value": 0.6,  # Low quality
        "expected_value": 0.9,
        "parents": []
    },
    "data_quantity": {
        "type": "INPUT",
        "name": "Number of Examples",
        "value": 1000,
        "expected_value": 10000,
        "parents": []
    },
    "model_complexity": {
        "type": "PARAMETER",
        "name": "Model Parameters",
        "value": 1e6,
        "expected_value": 1e7,
        "parents": []
    },
    "accuracy": {
        "type": "OUTPUT",
        "name": "Accuracy",
        "value": 0.65,
        "expected_value": 0.85,
        "parents": ["data_quality", "data_quantity", "model_complexity"]
    }
}

system.build_causal_model(components)

diagnosis = system.diagnose_failure(
    failure_description={
        "severity": 0.8,
        "component_values": {
            "accuracy": 0.65,
            "data_quality": 0.6,
            "data_quantity": 1000
        }
    },
    failure_mode=FailureMode.UNDERFITTING
)

# Diagnosis will identify root causes and recommend:
# - Collecting more data
# - Improving data quality
# - Increasing model capacity
```

### Use Case 2: Overfitting Detection

```python
# High training accuracy, low validation accuracy

system = create_causal_diagnosis_system()

components = {
    "regularization": {
        "type": "HYPERPARAMETER",
        "name": "Regularization Strength",
        "value": 0.0,  # No regularization!
        "expected_value": 0.01,
        "parents": []
    },
    "dropout": {
        "type": "HYPERPARAMETER",
        "name": "Dropout Rate",
        "value": 0.0,
        "expected_value": 0.3,
        "parents": []
    },
    "training_accuracy": {
        "type": "OUTPUT",
        "name": "Training Accuracy",
        "value": 0.99,  # Too high
        "expected_value": 0.85,
        "parents": ["regularization", "dropout"]
    },
    "validation_accuracy": {
        "type": "OUTPUT",
        "name": "Validation Accuracy",
        "value": 0.65,  # Too low
        "expected_value": 0.85,
        "parents": ["regularization", "dropout"]
    }
}

system.build_causal_model(components)

diagnosis = system.diagnose_failure(
    failure_description={
        "severity": 0.9,
        "component_values": {
            "training_accuracy": 0.99,
            "validation_accuracy": 0.65,
            "regularization": 0.0,
            "dropout": 0.0
        }
    },
    failure_mode=FailureMode.OVERFITTING
)

# Will recommend adding regularization and dropout
```

### Use Case 3: Counterfactual Analysis

```python
# "What if I had 10x more data?"

system = create_causal_diagnosis_system()
# ... (build model as above)

# Generate counterfactuals
counterfactuals = system.counterfactual_reasoner.find_best_counterfactuals(
    target_outcome="accuracy",
    num_counterfactuals=5,
    require_actionable=True
)

for cf in counterfactuals:
    print(f"\n{'='*60}")
    print(f"Scenario: {cf.description}")
    print(f"Change: {cf.changed_node}")
    print(f"  From: {cf.original_value}")
    print(f"  To:   {cf.counterfactual_value}")
    print(f"Expected Outcome Change: {cf.outcome_change:+.2%}")
    print(f"Plausibility: {cf.plausibility:.2%}")
    print(f"Action Needed: {cf.intervention_required.value}")
    print(f"Actionable: {'✅ Yes' if cf.is_actionable else '❌ No'}")
```

### Use Case 4: Intervention Planning

```python
# Get a complete plan to fix the issue

system = create_causal_diagnosis_system()
# ... (build model and diagnose failure)

diagnosis = system.diagnose_failure(...)

# Access intervention plan
for strategy, params in diagnosis.recommended_interventions[:3]:
    print(f"\nIntervention: {strategy.value}")
    print(f"  Confidence: {params.get('confidence', 0):.2%}")
    print(f"  Expected Improvement: {params.get('improvement', 0):.2%}")
    print(f"  Estimated Cost: {params.get('cost', 0):.2f}")

# Create detailed plan
plan = diagnosis.intervention_plan  # If available

if plan:
    print(f"\n📋 Intervention Plan: {plan.plan_id}")
    print(f"Expected Improvement: {plan.expected_improvement:.1%}")
    print(f"Confidence: {plan.confidence:.2%}")
    print(f"Cost: {plan.estimated_cost:.2f} GPU hours")

    print("\n⚠️  Risks:")
    for risk in plan.risks:
        print(f"  • {risk}")

    print("\n📊 Validation Metrics:")
    for metric in plan.validation_metrics:
        print(f"  • {metric}")
```

---

## 🎯 Failure Modes Reference

```python
from training.causal_self_diagnosis import FailureMode

# Available failure modes:
FailureMode.ACCURACY_DROP          # General accuracy decrease
FailureMode.HALLUCINATION          # Model generating false info
FailureMode.BIAS                   # Unfair predictions
FailureMode.INSTABILITY            # Inconsistent outputs
FailureMode.OVERFITTING            # Train high, val low
FailureMode.UNDERFITTING           # Both train and val low
FailureMode.CATASTROPHIC_FORGETTING # Lost previous knowledge
FailureMode.ADVERSARIAL_VULNERABILITY # Easily fooled
FailureMode.DISTRIBUTION_SHIFT     # Data changed
```

---

## 🔧 Intervention Strategies Reference

```python
from training.causal_self_diagnosis import InterventionStrategy

# Available interventions:
InterventionStrategy.RETRAIN               # Full retraining
InterventionStrategy.FINE_TUNE             # Targeted fine-tuning
InterventionStrategy.ADJUST_HYPERPARAMETERS # Change hyperparams
InterventionStrategy.ADD_REGULARIZATION    # Add dropout, L2, etc.
InterventionStrategy.COLLECT_MORE_DATA     # Gather more examples
InterventionStrategy.CHANGE_ARCHITECTURE   # Modify model structure
InterventionStrategy.APPLY_PATCH           # Quick fix
InterventionStrategy.RESET_COMPONENT       # Reinitialize part
```

---

## 📊 Component Types Reference

```python
from training.causal_self_diagnosis import CausalNodeType

# Node types for system components:
CausalNodeType.INPUT           # Input features, data
CausalNodeType.HIDDEN          # Hidden states, embeddings
CausalNodeType.OUTPUT          # Predictions, metrics
CausalNodeType.PARAMETER       # Model weights
CausalNodeType.HYPERPARAMETER  # Training settings
CausalNodeType.ENVIRONMENT     # External factors
```

---

## 💡 Best Practices

### 1. Define Clear Expected Values

```python
# ✅ Good: Clear expectations
"learning_rate": {
    "value": 0.01,
    "expected_value": 0.001,  # Clear target
}

# ❌ Bad: No expected value
"learning_rate": {
    "value": 0.01,
    "expected_value": None,  # Can't detect deviation
}
```

### 2. Model Complete Causal Chains

```python
# ✅ Good: Complete chain
components = {
    "data": {..., "parents": []},
    "preprocessing": {..., "parents": ["data"]},
    "model": {..., "parents": ["preprocessing"]},
    "output": {..., "parents": ["model"]}
}

# ❌ Bad: Missing links
components = {
    "data": {...},
    "output": {...}  # Missing intermediate steps
}
```

### 3. Use Specific Failure Modes

```python
# ✅ Good: Specific failure mode
diagnosis = system.diagnose_failure(
    failure_description=...,
    failure_mode=FailureMode.OVERFITTING  # Specific
)

# ⚠️  Less optimal: Generic mode
diagnosis = system.diagnose_failure(
    failure_description=...,
    failure_mode=FailureMode.ACCURACY_DROP  # Generic
)
```

### 4. Provide Complete Failure Context

```python
# ✅ Good: Complete context
failure = {
    "severity": 0.8,
    "component_values": {
        "accuracy": 0.65,
        "training_data": 1000,
        "learning_rate": 0.01,
        "regularization": 0.0
    }
}

# ❌ Bad: Minimal context
failure = {
    "severity": 0.5,
    "component_values": {
        "accuracy": 0.65
    }
}
```

---

## 🔍 Advanced Features

### Custom Causal Effect Estimation

```python
from training.causal_self_diagnosis import CausalGraph

graph = CausalGraph()

# Add edge with specific causal effect
graph.add_edge(
    source="learning_rate",
    target="convergence_speed",
    causal_effect=0.85,  # Strong positive effect
    confidence=0.9,      # High confidence
    observational_evidence=0.75,
    interventional_evidence=0.88,
    counterfactual_evidence=0.82
)
```

### Learning from Intervention Outcomes

```python
# After applying an intervention, teach the system

intervention_result = {
    "intervention_id": "int_001",
    "strategy": InterventionStrategy.ADD_REGULARIZATION,
    "actual_improvement": 0.12,  # 12% improvement
    "predicted_improvement": 0.10,
    "success": True
}

# System learns and improves future recommendations
system.learn_from_intervention("int_001", intervention_result)
```

### Custom Counterfactual Plausibility

```python
from training.causal_self_diagnosis import CounterfactualReasoner

reasoner = CounterfactualReasoner(system.causal_graph)

# Generate with custom plausibility threshold
counterfactuals = reasoner.find_best_counterfactuals(
    target_outcome="accuracy",
    num_counterfactuals=5,
    require_actionable=True,
    min_plausibility=0.7  # Only realistic scenarios
)
```

---

## 📖 Examples

### Complete Example: End-to-End Diagnosis

```python
from training.causal_self_diagnosis import *

# Create system
system = create_causal_diagnosis_system()

# Define components
components = {
    "training_epochs": {
        "type": "HYPERPARAMETER",
        "name": "Training Epochs",
        "value": 10,
        "expected_value": 50,
        "parents": []
    },
    "batch_size": {
        "type": "HYPERPARAMETER",
        "name": "Batch Size",
        "value": 32,
        "expected_value": 128,
        "parents": []
    },
    "dataset_size": {
        "type": "INPUT",
        "name": "Dataset Size",
        "value": 5000,
        "expected_value": 50000,
        "parents": []
    },
    "convergence": {
        "type": "OUTPUT",
        "name": "Convergence Quality",
        "value": 0.6,
        "expected_value": 0.9,
        "parents": ["training_epochs", "batch_size", "dataset_size"]
    }
}

# Build model
system.build_causal_model(components)

# Diagnose
failure = {
    "severity": 0.7,
    "component_values": {
        "convergence": 0.6,
        "training_epochs": 10,
        "dataset_size": 5000
    }
}

diagnosis = system.diagnose_failure(
    failure_description=failure,
    failure_mode=FailureMode.UNDERFITTING
)

# Print comprehensive report
print("="*70)
print("CAUSAL DIAGNOSIS REPORT")
print("="*70)
print(f"\nFailure Mode: {diagnosis.failure_mode.value}")
print(f"Confidence: {diagnosis.diagnosis_confidence:.2%}")

print(f"\n📍 Root Causes ({len(diagnosis.root_causes)}):")
for i, cause in enumerate(diagnosis.root_causes, 1):
    node = system.causal_graph.nodes[cause]
    print(f"  {i}. {node.name} ({cause})")
    print(f"     Current: {node.current_value}")
    print(f"     Expected: {node.expected_value}")
    print(f"     Deviation: {node.deviation:.2%}")

print(f"\n🔧 Recommended Interventions:")
for i, (strategy, score) in enumerate(diagnosis.recommended_interventions, 1):
    print(f"  {i}. {strategy.value} (confidence: {score:.2%})")

print(f"\n💭 Counterfactual Scenarios:")
for i, cf in enumerate(diagnosis.counterfactuals, 1):
    print(f"\n  {i}. {cf.description}")
    print(f"     Change: {cf.changed_node}")
    print(f"     Impact: {cf.outcome_change:+.2%}")
    print(f"     Actionable: {'✅' if cf.is_actionable else '❌'}")

print("\n" + "="*70)
```

---

## 🆘 Troubleshooting

### Issue: No root causes identified

**Solution**: Ensure nodes have `expected_value` set and `current_value` differs significantly

### Issue: Low confidence diagnoses

**Solution**: Add more edges to causal graph, provide more observational data

### Issue: No actionable counterfactuals

**Solution**: Lower `min_plausibility` threshold or add more INPUT/HYPERPARAMETER nodes

---

## 📚 Additional Resources

- **Full Documentation**: `docs/metacognitive_causal_systems.md`
- **Implementation Details**: `docs/CAUSAL_SELF_DIAGNOSIS_COMPLETE.md`
- **Demo Script**: `examples/metacognitive_causal_demo.py`
- **Test Suite**: `tests/test_causal_diagnosis.py`

---

## 🎓 Learning Path

1. **Start Here**: Run the basic usage example above
2. **Try Scenarios**: Experiment with different failure modes
3. **Run Demo**: Execute `examples/metacognitive_causal_demo.py`
4. **Read Docs**: Study `docs/metacognitive_causal_systems.md`
5. **Custom Integration**: Build your own causal models

---

**Questions?** Check the full documentation or examine the demo scripts for more examples!
