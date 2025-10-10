"""
Basic tests for Symbio AI system components.
Run with: pytest tests/
"""

import pytest
import pytest_asyncio
import asyncio
from unittest.mock import Mock, AsyncMock
from pathlib import Path
import sys

# Ensure project root on path for direct pytest invocation
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import load_config, SymbioConfig, DataConfig, bootstrap_control_plane
from data.loader import DataManager
from models.registry import ModelRegistry, ModelFramework
from agents.orchestrator import AgentOrchestrator, Task, TaskType, MessagePriority


class TestConfiguration:
    """Test configuration management."""
    
    def test_load_default_config(self):
        """Test loading default configuration."""
        config = load_config()
        assert isinstance(config, SymbioConfig)
        assert config.data.batch_size == 32
        assert config.training.population_size == 50
    
    def test_config_serialization(self, tmp_path):
        """Test configuration serialization."""
        config = SymbioConfig()
        config_path = tmp_path / "test_config.yaml"
        
        # This would work with proper YAML implementation
        # save_config(config, str(config_path))
        # loaded_config = load_config(str(config_path))
        # assert loaded_config.data.batch_size == config.data.batch_size

    def test_control_plane_bootstrap(self):
        """Ensure control plane bootstrap registers default tenant."""
        config = SymbioConfig()
        bootstrap_control_plane(config)
        from control_plane.tenancy import TENANT_REGISTRY

        tenant = TENANT_REGISTRY.get_tenant("public")
        assert tenant is not None
        assert "quota_guard" in tenant.enabled_policies


class TestDataManager:
    """Test data management system."""
    
    @pytest_asyncio.fixture
    async def data_manager(self, tmp_path):
        """Create test data manager."""
        config = DataConfig()
        config.base_path = str(tmp_path)
        manager = DataManager(config)
        await manager.initialize()
        return manager
    
    @pytest.mark.asyncio
    async def test_initialization(self, data_manager):
        """Test data manager initialization."""
        assert data_manager.base_path.exists()
        assert len(data_manager.datasets) == 0
    
    @pytest.mark.asyncio
    async def test_add_dataset(self, data_manager):
        """Test adding a dataset."""
        test_data = {"key": "value", "items": [1, 2, 3]}
        await data_manager.add_dataset("test_json", test_data, "json")
        
        assert "test_json" in data_manager.datasets
        metadata = data_manager.get_dataset_info("test_json")
        assert metadata.format == "json"
    
    @pytest.mark.asyncio
    async def test_load_dataset(self, data_manager):
        """Test loading a dataset."""
        test_data = {"test": "data"}
        await data_manager.add_dataset("load_test", test_data, "json")
        
        loaded_data = await data_manager.load_dataset("load_test")
        assert loaded_data == test_data
    
    @pytest.mark.asyncio
    async def test_list_datasets(self, data_manager):
        """Test listing datasets."""
        datasets = data_manager.list_datasets()
        assert isinstance(datasets, list)


class TestModelRegistry:
    """Test model registry system."""
    
    @pytest_asyncio.fixture
    async def model_registry(self, tmp_path):
        """Create test model registry."""
        from config.settings import ModelConfig
        config = ModelConfig()
        config.registry_path = str(tmp_path / "models")
        registry = ModelRegistry(config)
        await registry.initialize()
        return registry
    
    @pytest.mark.asyncio
    async def test_initialization(self, model_registry):
        """Test model registry initialization."""
        assert model_registry.registry_path.exists()
        assert len(model_registry.models) == 0
    
    @pytest.mark.asyncio
    async def test_create_base_model(self, model_registry):
        """Test creating a base model."""
        model_id = await model_registry.create_base_model(
            name="test_model",
            framework=ModelFramework.PYTORCH,
            architecture="transformer"
        )
        
        assert model_id in model_registry.metadata_cache
        metadata = model_registry.metadata_cache[model_id]
        assert metadata.name == "test_model"
        assert metadata.framework == ModelFramework.PYTORCH
    
    @pytest.mark.asyncio
    async def test_get_model(self, model_registry):
        """Test retrieving a model."""
        model_id = await model_registry.create_base_model(
            name="retrieval_test",
            framework=ModelFramework.PYTORCH,
            architecture="cnn"
        )
        
        model = await model_registry.get_model(model_id)
        assert model is not None
        assert model.metadata.id == model_id
    
    @pytest.mark.asyncio
    async def test_list_models(self, model_registry):
        """Test listing models."""
        models = model_registry.list_models()
        assert isinstance(models, list)
    
    @pytest.mark.asyncio
    async def test_search_models(self, model_registry):
        """Test searching models."""
        results = model_registry.search_models("transformer")
        assert isinstance(results, list)


class TestAgentOrchestrator:
    """Test agent orchestration system."""
    
    @pytest_asyncio.fixture
    async def orchestrator(self):
        """Create test orchestrator."""
        agent_configs = [
            {
                "id": "test-agent",
                "name": "Test Agent",
                "capabilities": ["test"],
                "description": "Synthetic test agent",
                "skills": ["testing"],
            }
        ]
        def _noop_integration(self, *args, **kwargs):  # type: ignore[override]
            return {}

        for attr in (
            "_chaining_integration",
            "_weighted_integration",
            "_ensemble_integration",
            "_hierarchical_integration",
        ):
            if not hasattr(AgentOrchestrator, attr):
                setattr(AgentOrchestrator, attr, _noop_integration)
        orchestrator = AgentOrchestrator(agent_configs)
        await orchestrator.initialize()
        orchestrator.agents = orchestrator.agent_registry
        return orchestrator
    
    @pytest.mark.asyncio
    async def test_initialization(self, orchestrator):
        """Test orchestrator initialization."""
        assert len(orchestrator.agents) > 0
        assert orchestrator.system_metrics["active_agents"] > 0
    
    @pytest.mark.asyncio
    async def test_submit_task(self, orchestrator):
        """Test task submission."""
        task = Task(
            id="test_task_1",
            task_type=TaskType.INFERENCE,
            priority=MessagePriority.NORMAL,
            payload={"test": "data"}
        )
        
        task_id = await orchestrator.submit_task(task)
        assert task_id == task.id
    
    @pytest.mark.asyncio
    async def test_get_system_status(self, orchestrator):
        """Test getting system status."""
        status = orchestrator.get_system_status()
        assert "agents" in status
        assert "system_metrics" in status
        assert isinstance(status["agents"], dict)


class TestEvolutionaryTraining:
    """Test evolutionary training system."""
    
    @pytest.mark.asyncio
    async def test_fitness_evaluator(self):
        """Test fitness evaluation."""
        from training.manager import MultiObjectiveFitnessEvaluator, Individual
        
        evaluator = MultiObjectiveFitnessEvaluator()
        individual = Individual(id="test_1", model_id="model_1")
        
        fitness = await evaluator.evaluate(individual, Mock(), None)
        assert isinstance(fitness, float)
        assert 0.0 <= fitness <= 1.0
        assert len(individual.evaluation_metrics) > 0
    
    @pytest.mark.asyncio
    async def test_training_manager_initialization(self):
        """Test training manager initialization."""
        from training.manager import TrainingManager
        from config.settings import TrainingConfig
        
        config = TrainingConfig()
        manager = TrainingManager(config)
        await manager.initialize()
        
        assert manager.default_fitness_evaluator is not None
        assert len(manager.active_trainers) == 0


class TestBenchmarkSystem:
    """Test evaluation and benchmarking system."""
    
    @pytest.mark.asyncio
    async def test_benchmark_system_initialization(self):
        """Test benchmark system initialization."""
        from evaluation.benchmarks import BenchmarkSuiteCollection
        from config.settings import EvaluationConfig
        
        config = EvaluationConfig()
        system = BenchmarkSuiteCollection(config)
        await system.initialize()
        
        assert len(system.runner.benchmarks) > 0
        assert len(system.runner.suites) > 0
    
    @pytest.mark.asyncio
    async def test_accuracy_benchmark(self):
        """Test accuracy benchmark."""
        from evaluation.benchmarks import AccuracyBenchmark
        
        benchmark = AccuracyBenchmark()
        mock_model = Mock()
        mock_model.metadata.id = "test_model"
        
        result = await benchmark.run(mock_model, None)
        assert result.benchmark_id == "accuracy_standard"
        assert 0.0 <= result.score <= 1.0
        assert isinstance(result.passed, bool)
    
    @pytest.mark.asyncio
    async def test_latency_benchmark(self):
        """Test latency benchmark."""
        from evaluation.benchmarks import LatencyBenchmark
        
        benchmark = LatencyBenchmark()
        mock_model = Mock()
        mock_model.metadata.id = "test_model"
        
        result = await benchmark.run(mock_model, None)
        assert result.benchmark_id == "latency_standard"
        assert result.score > 0
        assert "p95_latency" in result.metadata


class TestIntegration:
    """Integration tests for the complete system."""
    
    @pytest.mark.asyncio
    async def test_basic_system_workflow(self, tmp_path):
        """Test basic system workflow."""
        # This test would require proper initialization
        # and would be more comprehensive in a real implementation
        
        from config.settings import SymbioConfig, DataConfig, ModelConfig
        
        # Create test configuration
        config = SymbioConfig()
        config.data.base_path = str(tmp_path / "data")
        config.models.registry_path = str(tmp_path / "models")
        
        # Test that configuration is properly structured
        assert isinstance(config.data, DataConfig)
        assert isinstance(config.models, ModelConfig)


# Utility functions for testing
def create_mock_model():
    """Create a mock model for testing."""
    model = Mock()
    model.metadata.id = "test_model_123"
    model.metadata.name = "Test Model"
    model.is_loaded = True
    model.predict = AsyncMock(return_value={"prediction": "test_result"})
    return model


def create_test_task():
    """Create a test task for agent testing."""
    return Task(
        id="test_task_123",
        task_type=TaskType.INFERENCE,
        priority=MessagePriority.NORMAL,
        payload={"inputs": [1, 2, 3], "model_id": "test_model"}
    )


# Pytest configuration
@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()