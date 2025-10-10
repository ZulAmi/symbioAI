# Neural-Symbolic Architecture - Quick Reference

**Status:** ✅ Production Ready | **Version:** 1.0.0 | **Updated:** December 2024

---

## 🚀 Quick Start (30 seconds)

```python
from training.neural_symbolic_architecture import create_neural_symbolic_architecture, ProgramExample

# 1. Create architecture
arch = create_neural_symbolic_architecture()

# 2. Synthesize a program
program = arch.synthesize_program(
    "Sort numbers descending",
    [ProgramExample({"items": [3,1,4]}, [4,3,1])]
)

# 3. Get verified output
output, proof = arch.reason_with_proof([1.0, 2.0, 3.0], generate_proof=True)
verified = arch.verify_output(output, proof)

print(f"Program: {program.code}")
print(f"Output verified: {verified}")
```

---

## 📋 Common Tasks

### Task 1: Program Synthesis

```python
architecture = create_neural_symbolic_architecture()

examples = [
    ProgramExample(inputs={"x": 5, "y": 3}, output=8),
    ProgramExample(inputs={"x": 10, "y": 7}, output=17)
]

program = architecture.synthesize_program("Add two numbers", examples)
print(program.code)
print(f"Correctness: {program.correctness_score:.0%}")
```

### Task 2: Learn Logical Rules

```python
training_data = [
    ({"temp": 22, "humid": 50}, "comfortable"),
    ({"temp": 35, "humid": 80}, "hot"),
    # ... more examples
]

rules = architecture.learn_logic_rules(training_data, num_epochs=100)

for rule in rules:
    print(f"{rule} (confidence: {rule.confidence:.0%})")
```

### Task 3: Add Constraints

```python
from training.neural_symbolic_architecture import Constraint, SymbolicExpression, LogicalOperator

constraint = Constraint(
    constraint_id="must_be_positive",
    expression=SymbolicExpression(LogicalOperator.AND, variable="x > 0"),
    weight=5.0,
    hard=True  # Must be satisfied
)

architecture.add_constraint(constraint)
```

### Task 4: Verified Reasoning

```python
input_data = [1.0, 2.0, 3.0]
output, proof = architecture.reason_with_proof(input_data, generate_proof=True)

# Verify
is_valid = architecture.verify_output(output, proof)

# Explain
explanation = architecture.explain_reasoning(input_data, output, proof)
print(explanation)
```

### Task 5: Agent Integration

```python
from training.neural_symbolic_architecture import create_symbolic_reasoning_agent
import asyncio

async def run():
    agent = create_symbolic_reasoning_agent("my_agent")

    task = {
        "type": "program_synthesis",
        "description": "Reverse a string",
        "examples": [
            {"inputs": {"text": "hello"}, "output": "olleh"}
        ]
    }

    result = await agent.handle_task(task)
    print(result["program"])

asyncio.run(run())
```

---

## 🎯 Feature Cheat Sheet

| Feature             | Class/Function                | Key Method            | Time   |
| ------------------- | ----------------------------- | --------------------- | ------ |
| Program Synthesis   | `ProgramSynthesizer`          | `synthesize()`        | ~150ms |
| Rule Learning       | `DifferentiableLogicNetwork`  | `learn_logic_rules()` | 8-95s  |
| Constraint Checking | `ConstraintSatisfactionLayer` | `add_constraint()`    | <5ms   |
| Proof Generation    | `ProofGenerator`              | `reason_with_proof()` | ~60ms  |
| Agent Tasks         | `SymbolicReasoningAgent`      | `handle_task()`       | varies |

---

## 🔧 Configuration

### Create with Custom Dimensions

```python
architecture = NeuralSymbolicArchitecture(
    input_dim=256,   # Input feature size
    hidden_dim=512,  # Hidden layer size
    output_dim=128   # Output size
)
```

### Add Symbolic Knowledge

```python
from training.neural_symbolic_architecture import SymbolicExpression, LogicalRule, LogicalOperator

facts = [
    SymbolicExpression(LogicalOperator.AND, variable="fact1"),
    SymbolicExpression(LogicalOperator.OR, variable="fact2")
]

rules = [
    LogicalRule(
        rule_id="rule1",
        premises=[SymbolicExpression(LogicalOperator.AND, variable="A")],
        conclusion=SymbolicExpression(LogicalOperator.OR, variable="B"),
        confidence=0.9,
        weight=0.8
    )
]

architecture.add_symbolic_knowledge(facts, rules)
```

---

## 📊 Performance Tips

### For Faster Synthesis

- Use specific descriptions
- Provide more examples (3-5 ideal)
- Leverage pattern-based synthesis

### For Better Rule Learning

- More training data = better rules
- 100+ examples recommended
- Use 50-100 epochs for balance

### For Efficient Proofs

- Reuse proofs when possible
- Cache proof structures
- Batch verify multiple outputs

---

## 🐛 Troubleshooting

### Issue: Low synthesis accuracy

**Solution:** Add more diverse examples, be more specific in description

### Issue: Rules not learning

**Solution:** Increase epochs, check training data quality, ensure sufficient examples

### Issue: Proofs invalid

**Solution:** Check constraints, validate input data, review proof steps

### Issue: Agent tasks failing

**Solution:** Verify task format, check async execution, review error logs

---

## 📚 Key Classes Reference

### NeuralSymbolicArchitecture

Main system class

**Methods:**

- `add_symbolic_knowledge(facts, rules)` - Add knowledge
- `add_constraint(constraint)` - Add constraint
- `synthesize_program(desc, examples, constraints=None)` - Generate code
- `learn_logic_rules(data, epochs=100)` - Learn rules
- `reason_with_proof(input, generate_proof=True)` - Verified reasoning
- `verify_output(output, proof)` - Verify correctness
- `explain_reasoning(input, output, proof)` - Get explanation

### ProgramSynthesizer

Code generation engine

**Methods:**

- `synthesize(description, examples, constraints)` - Main synthesis method

### ProofGenerator

Proof creation system

**Methods:**

- `generate_proof(input, output, reasoning, constraints)` - Create proof

### SymbolicReasoningAgent

Agent wrapper for orchestrator

**Methods:**

- `handle_task(task)` - Process agent tasks (async)

---

## 🎨 Examples

### Example 1: Complete Workflow

```python
from training.neural_symbolic_architecture import *

# Setup
arch = create_neural_symbolic_architecture()

# Add domain knowledge
arch.add_symbolic_knowledge(
    facts=[SymbolicExpression(LogicalOperator.AND, variable="valid_input")],
    rules=[LogicalRule(
        rule_id="validation",
        premises=[SymbolicExpression(LogicalOperator.AND, variable="valid_input")],
        conclusion=SymbolicExpression(LogicalOperator.OR, variable="valid_output"),
        confidence=0.95
    )]
)

# Learn from data
data = [({"x": 1, "y": 2}, 3), ({"x": 3, "y": 4}, 7)]
rules = arch.learn_logic_rules(data, num_epochs=50)

# Add constraint
arch.add_constraint(Constraint(
    constraint_id="positive",
    expression=SymbolicExpression(LogicalOperator.AND, variable="x > 0"),
    weight=5.0,
    hard=True
))

# Reason with proof
output, proof = arch.reason_with_proof([1, 2, 3], generate_proof=True)
verified = arch.verify_output(output, proof)

print(f"Verified: {verified}, Validity: {proof.validity_score:.0%}")
```

### Example 2: Multi-Task Agent

```python
import asyncio
from training.neural_symbolic_architecture import create_symbolic_reasoning_agent

async def multi_task_demo():
    agent = create_symbolic_reasoning_agent("agent1")

    tasks = [
        {"type": "program_synthesis", "description": "Sort list", "examples": [...]},
        {"type": "verified_reasoning", "input": [1, 2, 3]},
        {"type": "rule_learning", "training_data": [...], "num_epochs": 50}
    ]

    results = []
    for task in tasks:
        result = await agent.handle_task(task)
        results.append(result)

    for i, result in enumerate(results):
        print(f"Task {i+1}: {result['status']}")

asyncio.run(multi_task_demo())
```

---

## 🔗 Quick Links

- **Full Documentation:** `docs/neural_symbolic_architecture.md`
- **Demo Script:** `examples/neural_symbolic_demo.py`
- **Implementation:** `training/neural_symbolic_architecture.py`
- **Completion Summary:** `NEURAL_SYMBOLIC_COMPLETE.md`

---

## 💡 Tips & Best Practices

1. ✅ Always generate proofs for critical outputs
2. ✅ Use hard constraints for safety requirements
3. ✅ Provide 3-5 examples for program synthesis
4. ✅ Train with 100+ examples for rule learning
5. ✅ Verify outputs before using in production
6. ✅ Cache learned rules for reuse
7. ✅ Use agent integration for complex workflows
8. ✅ Monitor proof validity scores (>80% is good)

---

## 🎯 Common Patterns

### Pattern: Safe AI System

```python
# 1. Define safety constraints
safety_constraints = [
    Constraint("no_harm", ..., hard=True),
    Constraint("fairness", ..., hard=True)
]

# 2. Add to architecture
for c in safety_constraints:
    architecture.add_constraint(c)

# 3. Always verify
output, proof = architecture.reason_with_proof(input, generate_proof=True)
if not architecture.verify_output(output, proof):
    raise ValueError("Safety verification failed")
```

### Pattern: Explainable Predictions

```python
# 1. Get prediction with proof
output, proof = architecture.reason_with_proof(input_data, generate_proof=True)

# 2. Generate explanation
explanation = architecture.explain_reasoning(input_data, output, proof)

# 3. Present to user
print(f"Prediction: {output}")
print(f"Confidence: {proof.validity_score:.0%}")
print(f"\nExplanation:\n{explanation}")
```

### Pattern: Continuous Learning

```python
# 1. Collect new data
new_data = collect_recent_examples()

# 2. Learn new rules
new_rules = architecture.learn_logic_rules(new_data, num_epochs=50)

# 3. Update knowledge base (rules auto-added)
print(f"Learned {len(new_rules)} new rules")

# 4. Re-verify existing rules
all_rules = architecture.knowledge_base.rules
for rule in all_rules:
    if rule.confidence < 0.7:
        print(f"Rule {rule.rule_id} needs retraining")
```

---

**Last Updated:** December 2024  
**Version:** 1.0.0  
**Status:** Production Ready ✅
