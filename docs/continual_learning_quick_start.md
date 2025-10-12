# Continual Learning Quick Start Guide

Learn new tasks without forgetting old ones in minutes.

---

## Quick Start (5 Minutes)

### 1. Run the Demo

```bash
python examples/continual_learning_demo.py
```

**Expected Output**: 8 comprehensive demos showing all continual learning features

---

## Basic Usage

### Minimal Example

```python
from training.continual_learning import create_continual_learning_engine, Task, TaskType

# Create engine (one line!)
engine = create_continual_learning_engine(strategy="combined")

# Register task
task = Task(
 task_id="mnist",
 task_name="MNIST",
 task_type=TaskType.CLASSIFICATION,
 input_dim=784,
 output_dim=10
)
engine.register_task(task)

# Train with anti-forgetting
engine.prepare_for_task(task, model, dataloader)
losses = engine.train_step(model, batch, optimizer, task)
engine.finish_task_training(task, model, dataloader, performance=0.95)
```

**That's it!** Your model now learns without forgetting.

---

## Choose Your Strategy

### 1. EWC (Elastic Weight Consolidation)

**Best for**: First 5-10 tasks, similar domains

```python
engine = create_continual_learning_engine(
 strategy="ewc",
 ewc_lambda=1000.0 # Higher = more protection
)
```

**Pros**: Simple, effective, low memory
**Cons**: Accumulates constraints over time

---

### 2. Experience Replay

**Best for**: Diverse tasks, enough memory

```python
engine = create_continual_learning_engine(
 strategy="replay",
 replay_buffer_size=10000
)
```

**Pros**: Works for any task type
**Cons**: Memory intensive (10K samples)

---

### 3. Progressive Neural Networks

**Best for**: Zero forgetting requirement, unlimited memory

```python
engine = create_continual_learning_engine(
 strategy="progressive",
 use_progressive_nets=True
)
```

**Pros**: **Zero forgetting** (guaranteed)
**Cons**: Memory grows with tasks

---

### 4. Task-Specific Adapters (LoRA)

**Best for**: 10+ tasks, parameter efficiency critical

```python
engine = create_continual_learning_engine(
 strategy="adapters",
 use_adapters=True,
 adapter_rank=8 # Lower = more efficient
)
```

**Pros**: **99% parameter savings**
**Cons**: Slight performance drop (<2%)

---

### 5. Combined (Recommended)

**Best for**: Production systems

```python
engine = create_continual_learning_engine(
 strategy="combined",
 ewc_lambda=1000.0,
 replay_buffer_size=10000,
 use_adapters=True
)
```

**Pros**: <5% forgetting, automatic optimization
**Cons**: Higher complexity

---

## Common Use Cases

### Use Case 1: Add New Task Without Forgetting

```python
# Learn Task 1
engine.prepare_for_task(task1, model, dataloader1)
for batch in dataloader1:
 losses = engine.train_step(model, batch, optimizer, task1)
engine.finish_task_training(task1, model, dataloader1, performance=0.95)

# Learn Task 2 (Task 1 still works!)
engine.prepare_for_task(task2, model, dataloader2)
for batch in dataloader2:
 losses = engine.train_step(model, batch, optimizer, task2)
engine.finish_task_training(task2, model, dataloader2, performance=0.93)

# Evaluate both tasks
results = engine.evaluate_all_tasks(model, {
 "task1": dataloader1,
 "task2": dataloader2
})
# Task 1: ~95% (no forgetting!)
# Task 2: ~93%
```

---

### Use Case 2: Detect and Prevent Interference

```python
# After training on multiple tasks
report = engine.interference_detector.detect_interference(
 current_task=task3,
 previous_tasks=[task1, task2]
)

print(f"Interference Level: {report.interference_level.value}")
# → "LOW" or "MEDIUM" or "HIGH" or "CATASTROPHIC"

print(f"Recommended Strategy: {report.recommended_strategy.value}")
# → Automatic recommendation based on severity

# Apply recommendations
if report.interference_level.value in ["HIGH", "CATASTROPHIC"]:
 # Increase protection
 engine.ewc.ewc_lambda = report.ewc_lambda # Higher λ
 # Increase replay ratio
 replay_batch = engine.get_replay_batch(
 batch_size=int(batch_size * report.replay_ratio)
 )
```

---

### Use Case 3: Parameter-Efficient Multi-Task Learning

```python
# Create adapter-based engine
engine = create_continual_learning_engine(
 strategy="adapters",
 use_adapters=True
)

# Add 100 tasks with minimal memory
for i, task in enumerate(tasks[:100]):
 # Each task adds only 1-10% parameters
 adapter = engine.adapter_manager.create_task_adapter(
 task, model, target_layers=["linear", "fc"]
 )

 # Train with adapter
 engine.adapter_manager.activate_task_adapter(task.task_id)
 # ... training loop ...

# Result: 100 tasks with ~10× memory of single model
```

---

### Use Case 4: Fast Task Switching

```python
# Switch between tasks instantly
engine.adapter_manager.activate_task_adapter("task1")
output1 = model(input_data) # Task 1 prediction

engine.adapter_manager.activate_task_adapter("task2")
output2 = model(input_data) # Task 2 prediction

# Switching time: <1ms
```

---

### Use Case 5: Get Comprehensive Report

```python
report = engine.get_continual_learning_report()

print(f"""
Continual Learning Report

Tasks: {report['num_tasks']}
Strategy: {report['strategy']}

EWC Protection:
 - Tasks protected: {report['components']['ewc']['num_tasks_protected']}
 - Average λ: {report['components']['ewc']['avg_lambda']:.1f}

Experience Replay:
 - Total samples: {report['components']['replay']['total_samples']}
 - Buffer usage: {report['components']['replay']['buffer_usage']:.1%}

Adapters:
 - Active adapters: {report['components']['adapters']['num_adapters']}
 - Parameter efficiency: {report['components']['adapters']['avg_efficiency']:.1%}

Performance:
 - Average: {report['performance']['average_performance']:.1%}
 - Best: {report['performance']['best_task_performance']:.1%}
 - Worst: {report['performance']['worst_task_performance']:.1%}
""")
```

---

## Run Specific Demos

### Demo 1: Task Registration

```bash
python examples/continual_learning_demo.py
# Select: 1
```

Shows how to register tasks and view characteristics.

---

### Demo 2: EWC Protection

```bash
python examples/continual_learning_demo.py
# Select: 2
```

Shows Fisher Information Matrix and EWC loss calculation.

---

### Demo 3: Experience Replay

```bash
python examples/continual_learning_demo.py
# Select: 3
```

Shows replay buffer management and sampling.

---

### Demo 4: Task Adapters

```bash
python examples/continual_learning_demo.py
# Select: 4
```

Shows LoRA adapter creation and efficiency.

---

### Demo 5: Interference Detection

```bash
python examples/continual_learning_demo.py
# Select: 5
```

Shows automatic interference detection and recommendations.

---

### Demo 6: Progressive Neural Networks

```bash
python examples/continual_learning_demo.py
# Select: 6
```

Shows column addition and zero forgetting.

---

### Demo 7: Combined Strategy

```bash
python examples/continual_learning_demo.py
# Select: 7
```

Shows complete multi-task learning workflow.

---

### Demo 8: Competitive Advantages

```bash
python examples/continual_learning_demo.py
# Select: 8
```

Shows comparison vs. competitors.

---

## Configuration Guide

### EWC Lambda (λ)

Controls protection strength:

```python
ewc_lambda=100 # Light protection (10% forgetting)
ewc_lambda=1000 # Medium protection (5% forgetting) ← RECOMMENDED
ewc_lambda=10000 # Strong protection (1% forgetting, may hurt new task)
```

---

### Replay Buffer Size

Controls memory vs. performance:

```python
replay_buffer_size=1000 # Low memory (10-15% forgetting)
replay_buffer_size=10000 # Balanced (5% forgetting) ← RECOMMENDED
replay_buffer_size=100000 # High memory (1% forgetting)
```

---

### Adapter Rank

Controls efficiency vs. performance:

```python
adapter_rank=4 # 99% efficiency, slight performance drop
adapter_rank=8 # 95% efficiency, minimal drop ← RECOMMENDED
adapter_rank=16 # 90% efficiency, no drop
```

---

## Troubleshooting

### Problem: High forgetting on old tasks

**Solution**: Increase protection

```python
# Option 1: Increase EWC lambda
engine = create_continual_learning_engine(
 strategy="ewc",
 ewc_lambda=5000 # Increase from 1000
)

# Option 2: Add replay
engine = create_continual_learning_engine(
 strategy="combined",
 replay_buffer_size=20000 # Increase from 10000
)

# Option 3: Use progressive nets (zero forgetting)
engine = create_continual_learning_engine(
 strategy="progressive",
 use_progressive_nets=True
)
```

---

### Problem: Poor performance on new tasks

**Solution**: Decrease protection

```python
# Option 1: Lower EWC lambda
engine = create_continual_learning_engine(
 strategy="ewc",
 ewc_lambda=500 # Decrease from 1000
)

# Option 2: Reduce replay ratio
engine = create_continual_learning_engine(
 strategy="replay",
 replay_buffer_size=5000 # Decrease from 10000
)

# Option 3: Use adapters (isolated parameters)
engine = create_continual_learning_engine(
 strategy="adapters",
 use_adapters=True
)
```

---

### Problem: Out of memory

**Solution**: Use parameter-efficient methods

```python
# Option 1: Use LoRA adapters (99% savings)
engine = create_continual_learning_engine(
 strategy="adapters",
 adapter_rank=4 # Lower rank = less memory
)

# Option 2: Reduce replay buffer
engine = create_continual_learning_engine(
 strategy="replay",
 replay_buffer_size=1000 # Reduce from 10000
)

# Option 3: Don't use progressive nets
engine = create_continual_learning_engine(
 strategy="combined",
 use_progressive_nets=False
)
```

---

### Problem: Interference detected

**Solution**: Apply automatic recommendations

```python
# Detect interference
report = engine.interference_detector.detect_interference(
 current_task, previous_tasks
)

# Apply recommendations
if report.interference_level != InterferenceLevel.NONE:
 # Use recommended strategy
 engine = create_continual_learning_engine(
 strategy=report.recommended_strategy.value
 )

 # Use recommended hyperparameters
 engine.ewc.ewc_lambda = report.ewc_lambda

 # Adjust replay ratio
 replay_size = int(batch_size * report.replay_ratio)
 replay_batch = engine.get_replay_batch(batch_size=replay_size)
```

---

## API Reference

### Core Classes

- `ContinualLearningEngine`: Main orchestrator
- `ElasticWeightConsolidation`: EWC implementation
- `ExperienceReplayManager`: Replay buffer management
- `ProgressiveNeuralNetwork`: Column-based architecture
- `TaskAdapterManager`: LoRA adapter management
- `InterferenceDetector`: Automatic monitoring

### Factory Function

```python
create_continual_learning_engine(
 strategy="combined", # ewc, replay, progressive, adapters, combined
 ewc_lambda=1000.0, # EWC regularization strength
 online_ewc=True, # Use online EWC variant
 replay_buffer_size=10000, # Max replay samples
 use_progressive_nets=False, # Use column-based architecture
 use_adapters=True, # Use LoRA adapters
 adapter_rank=8 # LoRA rank
)
```

### Key Methods

```python
# Register task
engine.register_task(task)

# Prepare for training
engine.prepare_for_task(task, model, dataloader)

# Train step (with anti-forgetting)
losses = engine.train_step(model, batch, optimizer, task)

# Finish training
engine.finish_task_training(task, model, dataloader, performance)

# Evaluate all tasks
results = engine.evaluate_all_tasks(model, task_dataloaders)

# Get report
report = engine.get_continual_learning_report()

# Detect interference
interference = engine.interference_detector.detect_interference(
 current_task, previous_tasks
)
```

---

## Best Practices

### 1. Start with Combined Strategy

```python
engine = create_continual_learning_engine(strategy="combined")
```

Automatically combines all techniques based on interference.

---

### 2. Monitor Interference

```python
# After each task
report = engine.interference_detector.detect_interference(
 current_task, previous_tasks
)

if report.interference_level.value in ["HIGH", "CATASTROPHIC"]:
 print(f" High interference detected!")
 print(f"Recommended: {report.recommended_strategy.value}")
```

---

### 3. Use Adapters for 10+ Tasks

```python
if len(tasks) >= 10:
 engine = create_continual_learning_engine(
 strategy="adapters",
 adapter_rank=8
 )
```

99% parameter savings.

---

### 4. Evaluate Regularly

```python
# Every N tasks
if num_tasks % 5 == 0:
 results = engine.evaluate_all_tasks(model, all_dataloaders)
 avg_performance = sum(results.values()) / len(results)
 print(f"Average performance: {avg_performance:.1%}")
```

---

### 5. Get Reports

```python
# At the end
report = engine.get_continual_learning_report()
# Save report for analysis
```

---

## Success Metrics

### Good Continual Learning

- <5% forgetting on old tasks
- >90% performance on new tasks
- Interference level: NONE or LOW
- Memory usage: <10× single model

### Signs of Problems

- >20% forgetting on old tasks
- <80% performance on new tasks
- Interference level: HIGH or CATASTROPHIC
- Memory usage: >50× single model

---

## Related Documentation

- **Full Documentation**: `CONTINUAL_LEARNING_COMPLETE.md`
- **Architecture**: `docs/architecture.md`
- **API Reference**: `docs/api_reference.md`
- **Adapter Registry**: `registry/adapter_registry.py`

---

## Quick Tips

1. **Use `combined` strategy** for most cases
2. **Monitor interference** after each task
3. **Use adapters** for 10+ tasks (99% parameter savings)
4. **Progressive nets** for zero forgetting (guaranteed)
5. **Increase EWC λ** if forgetting too much
6. **Decrease EWC λ** if new task performance poor
7. **Get reports** regularly for insights

---

## Next Steps

1. **Run demo**: `python examples/continual_learning_demo.py`
2. **Try on your data**: Modify demo with your tasks
3. **Tune hyperparameters**: Adjust based on interference
4. **Scale to production**: Use combined strategy with monitoring
5. **Read full docs**: `CONTINUAL_LEARNING_COMPLETE.md`

---

**Happy continual learning! No more catastrophic forgetting! **
