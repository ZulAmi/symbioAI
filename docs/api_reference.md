# Symbio AI - API Reference

## Overview

Symbio AI provides a comprehensive Python API for building, training, and deploying modular AI systems. This reference covers all major components and their interfaces.

## Core System

### SymbioAI

Main system coordinator class.

```python
from main import SymbioAI

# Initialize system
system = SymbioAI(config_path="config/custom.yaml")

# Run system
await system.run()
```

#### Methods

**`__init__(config_path: Optional[str] = None)`**

- Initialize the Symbio AI system
- `config_path`: Path to configuration file (optional)

**`async run() -> None`**

- Start the main system loop
- Initializes all subsystems and begins orchestration

## Configuration Management

### Settings

```python
from config.settings import load_config, save_config, SymbioConfig

# Load configuration
config = load_config("config/my_config.yaml")

# Modify configuration
config.training.population_size = 100

# Save configuration
save_config(config, "config/updated_config.yaml")
```

#### Configuration Classes

**`SymbioConfig`**

- Main configuration container
- Contains all subsystem configurations

**`DataConfig`**

- Data management settings
- `base_path`: Data directory path
- `batch_size`: Default batch size
- `cache_enabled`: Enable/disable caching

**`TrainingConfig`**

- Evolutionary training parameters
- `population_size`: Evolution population size
- `mutation_rate`: Mutation probability
- `crossover_rate`: Crossover probability

## Data Management

### DataManager

Central data management system.

```python
from data.loader import DataManager
from config.settings import DataConfig

# Initialize
config = DataConfig()
data_manager = DataManager(config)
await data_manager.initialize()

# Load dataset
data = await data_manager.load_dataset("my_dataset")

# Add new dataset
await data_manager.add_dataset("new_data", my_dataframe, "csv")

# Stream large dataset
async for batch in data_manager.stream_dataset("large_dataset"):
    process_batch(batch)
```

#### Methods

**`async initialize() -> None`**

- Initialize the data management system
- Scans for existing datasets

**`async load_dataset(name: str, use_cache: bool = True) -> Any`**

- Load a dataset by name
- Returns the loaded data

**`async add_dataset(name: str, data: Any, format_type: str) -> None`**

- Add a new dataset to the system
- `format_type`: "json", "csv", or "parquet"

**`async preprocess_dataset(name: str, processors: List[str]) -> Any`**

- Apply preprocessing to a dataset
- `processors`: List of preprocessing operations

**`async stream_dataset(name: str, batch_size: int = None) -> AsyncGenerator`**

- Stream dataset in batches
- Memory-efficient for large datasets

## Model Management

### ModelRegistry

Central model registry and management.

```python
from models.registry import ModelRegistry, ModelFramework
from config.settings import ModelConfig

# Initialize
config = ModelConfig()
registry = ModelRegistry(config)
await registry.initialize()

# Create base model
model_id = await registry.create_base_model(
    name="my_transformer",
    framework=ModelFramework.PYTORCH,
    architecture="transformer",
    parameters=1000000
)

# Get model
model = await registry.get_model(model_id)

# Merge models
merged_id = await registry.merge_models([model_id1, model_id2])

# Distill model
distilled_id = await registry.distill_model(model_id, "smaller_arch", 0.5)
```

#### Methods

**`async create_base_model(name: str, framework: ModelFramework, architecture: str, **kwargs) -> str`\*\*

- Create a new base model
- Returns model ID

**`async get_model(model_id: str, load_if_needed: bool = True) -> Optional[BaseModel]`**

- Retrieve a model by ID
- Loads from storage if needed

**`async merge_models(model_ids: List[str], strategy: str = "weighted_average") -> str`**

- Merge multiple models
- Returns merged model ID

**`async distill_model(teacher_id: str, student_architecture: str, compression_ratio: float = 0.5) -> str`**

- Distill a model to create compressed version
- Returns distilled model ID

**`list_models(model_type: Optional[str] = None) -> List[ModelMetadata]`**

- List all models or filter by type

**`search_models(query: str) -> List[ModelMetadata]`**

- Search models by name, tags, or description

## Agent Orchestration

### AgentOrchestrator

Coordinate intelligent agents for task execution.

```python
from agents.orchestrator import AgentOrchestrator, Task, TaskType, MessagePriority

# Initialize
orchestrator = AgentOrchestrator(config.agents)
await orchestrator.initialize()

# Create and submit task
task = Task(
    id="inference_task_1",
    task_type=TaskType.INFERENCE,
    priority=MessagePriority.HIGH,
    payload={"model_id": "my_model", "inputs": data}
)
task_id = await orchestrator.submit_task(task)

# Get task result
result = await orchestrator.get_task_result(task_id)

# Get system status
status = orchestrator.get_system_status()
```

#### Task Types

- `TaskType.INFERENCE`: Model inference tasks
- `TaskType.TRAINING`: Model training tasks
- `TaskType.EVALUATION`: Model evaluation tasks
- `TaskType.DATA_PROCESSING`: Data preprocessing tasks
- `TaskType.MODEL_MERGE`: Model merging operations
- `TaskType.COORDINATION`: System coordination tasks

#### Methods

**`async submit_task(task: Task) -> str`**

- Submit a task for execution
- Returns task ID

**`async get_task_result(task_id: str) -> Optional[Task]`**

- Get result of completed task

**`get_system_status() -> Dict[str, Any]`**

- Get current system status and metrics

## Training Management

### TrainingManager

Evolutionary training coordination.

```python
from training.manager import TrainingManager, EvolutionaryConfig

# Initialize
manager = TrainingManager(config.training)
await manager.initialize()

# Start evolutionary training
training_config = EvolutionaryConfig(
    population_size=50,
    max_generations=100,
    mutation_rate=0.1
)

training_id = await manager.start_evolutionary_training(
    "experiment_1",
    model_registry,
    config=training_config
)

# Check training status
status = await manager.get_training_status(training_id)

# Get training history
history = await manager.get_training_history(training_id)
```

#### Methods

**`async start_evolutionary_training(training_id: str, model_registry, config: Optional[EvolutionaryConfig] = None) -> str`**

- Start new evolutionary training session
- Returns training session ID

**`async get_training_status(training_id: str) -> Optional[Dict[str, Any]]`**

- Get current training status

**`async stop_training(training_id: str) -> bool`**

- Stop a training session

**`async pause_training(training_id: str) -> bool`**

- Pause a training session

**`async resume_training(training_id: str) -> bool`**

- Resume a paused training session

## Evaluation and Benchmarking

### BenchmarkSuiteCollection

Comprehensive model evaluation system.

```python
from evaluation.benchmarks import BenchmarkSuiteCollection

# Initialize
benchmark_system = BenchmarkSuiteCollection(config.evaluation)
await benchmark_system.initialize()

# Evaluate single model
results = await benchmark_system.evaluate_model(
    model,
    suite_names=["standard", "adversarial"]
)

# Compare multiple models
comparison = await benchmark_system.compare_models(
    [model1, model2, model3],
    suite_names=["comprehensive"]
)

# Run ablation study
ablation = await benchmark_system.run_ablation_study(
    base_model,
    [variant1, variant2, variant3]
)
```

#### Benchmark Suites

- `"standard"`: Basic performance benchmarks
- `"adversarial"`: Robustness and security tests
- `"efficiency"`: Speed and resource optimization
- `"comprehensive"`: All benchmark types

#### Methods

**`async evaluate_model(model, suite_names: List[str] = None) -> Dict[str, List[BenchmarkResult]]`**

- Evaluate model with specified benchmark suites

**`async compare_models(models: List[Any], suite_names: List[str] = None) -> Dict[str, Dict[str, Any]]`**

- Compare multiple models and generate rankings

**`async run_ablation_study(base_model, model_variants: List[Any]) -> Dict[str, Any]`**

- Run ablation study comparing variants

## Error Handling

All async operations may raise exceptions. Wrap in try-catch blocks:

```python
try:
    result = await data_manager.load_dataset("nonexistent")
except ValueError as e:
    print(f"Dataset not found: {e}")
except Exception as e:
    print(f"Unexpected error: {e}")
```

## Logging

Configure logging for debugging:

```python
import logging

# Set logging level
logging.getLogger("symbio_ai").setLevel(logging.DEBUG)

# Custom logger
logger = logging.getLogger("my_app")
logger.info("Application started")
```

## Examples

### Complete Workflow

```python
import asyncio
from main import SymbioAI
from models.registry import ModelFramework
from training.manager import EvolutionaryConfig
from agents.orchestrator import Task, TaskType, MessagePriority

async def main():
    # Initialize system
    system = SymbioAI()

    # Create and register a model
    model_id = await system.model_registry.create_base_model(
        name="my_classifier",
        framework=ModelFramework.PYTORCH,
        architecture="resnet50"
    )

    # Start evolutionary training
    training_id = await system.training_manager.start_evolutionary_training(
        "my_experiment",
        system.model_registry,
        config=EvolutionaryConfig(population_size=20)
    )

    # Submit inference task
    task = Task(
        id="test_inference",
        task_type=TaskType.INFERENCE,
        priority=MessagePriority.NORMAL,
        payload={"model_id": model_id, "inputs": test_data}
    )
    await system.orchestrator.submit_task(task)

    # Evaluate model
    model = await system.model_registry.get_model(model_id)
    results = await system.benchmark_suite.evaluate_model(model)

    print(f"Evaluation results: {results}")

if __name__ == "__main__":
    asyncio.run(main())
```

### Custom Agent

```python
from agents.orchestrator import Agent, AgentCapabilities, TaskType

class CustomAgent(Agent):
    def __init__(self, agent_id: str):
        capabilities = AgentCapabilities(
            supported_tasks=[TaskType.INFERENCE],
            max_concurrent_tasks=3,
            performance_metrics={"speed": 95.0},
            resource_requirements={"memory": "2GB"},
            specializations=["custom_models"]
        )
        super().__init__(agent_id, capabilities)

    async def execute_task(self, task):
        # Custom task execution logic
        return {"result": "custom_processing_complete"}

# Register with orchestrator
custom_agent = CustomAgent("my_custom_agent")
await orchestrator._register_agent(custom_agent)
```

### Custom Benchmark

```python
from evaluation.benchmarks import Benchmark, BenchmarkType, BenchmarkResult

class CustomBenchmark(Benchmark):
    def __init__(self):
        super().__init__("custom_test", "My Custom Test", BenchmarkType.ACCURACY)

    async def run(self, model, test_data):
        # Custom benchmark logic
        score = await self.evaluate_custom_metric(model, test_data)

        return BenchmarkResult(
            benchmark_id=self.benchmark_id,
            model_id=model.metadata.id,
            benchmark_type=self.benchmark_type,
            metric_type=MetricType.CLASSIFICATION_ACCURACY,
            score=score,
            metadata={},
            execution_time=0.1,
            timestamp=datetime.now().isoformat(),
            passed=score > 0.8
        )

    def validate_model(self, model):
        return True

# Register with benchmark system
benchmark_system.runner.register_benchmark(CustomBenchmark())
```

For more examples and advanced usage, see the `/docs/examples/` directory.
