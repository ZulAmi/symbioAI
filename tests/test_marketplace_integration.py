"""
Integration Test Suite for Marketplace-Integrated Self-Healing System

Comprehensive tests covering the integration between marketplace patch discovery,
failure monitoring, and auto-surgery systems.
"""

import pytest
import asyncio
import json
import tempfile
import shutil
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Any

from marketplace.patch_marketplace import (
    PatchMarketplace, PatchManifest, PatchType, PatchStatus, SecurityLevel
)
from marketplace.healing_integration import (
    MarketplaceIntegratedHealing, AutoPatchContext, PatchEvaluator
)
from training.auto_surgery import SurgeryConfig, AutoModelSurgery
from monitoring.failure_monitor import FailureMonitor, FailureContext
from config.settings import SETTINGS


@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
def surgery_config(temp_dir):
    """Create surgery configuration for tests."""
    return SurgeryConfig(
        output_dir=temp_dir,
        max_training_steps=10,  # Reduced for testing
        learning_rate=5e-4,
        min_accuracy_improvement=0.01,
        max_training_time_minutes=5
    )


@pytest.fixture
def failure_monitor():
    """Create failure monitor for tests."""
    return FailureMonitor(max_failure_history=100)


@pytest.fixture
def sample_failures():
    """Create sample failure data for testing."""
    return [
        {
            "input": "What is the capital of France?",
            "expected_output": "Paris",
            "actual_output": "Berlin",
            "error_type": "accuracy_drop",
            "severity": "medium",
            "timestamp": datetime.utcnow().isoformat(),
            "metadata": {"confidence": 0.3}
        },
        {
            "input": "Translate 'hello' to Spanish",
            "expected_output": "hola",
            "actual_output": "bonjour",
            "error_type": "wrong_language",
            "severity": "high",
            "timestamp": datetime.utcnow().isoformat(),
            "metadata": {"confidence": 0.8}
        },
        {
            "input": "What is 2 + 2?",
            "expected_output": "4",
            "actual_output": "5",
            "error_type": "calculation_error",
            "severity": "medium",
            "timestamp": datetime.utcnow().isoformat(),
            "metadata": {"confidence": 0.9}
        }
    ]


class TestMarketplaceIntegration:
    """Test marketplace integration functionality."""
    
    @pytest.mark.asyncio
    async def test_patch_marketplace_initialization(self):
        """Test marketplace initialization."""
        marketplace = PatchMarketplace()
        
        # Should initialize without errors
        await marketplace.initialize()
        
        # Should have basic configuration
        assert marketplace.cache_dir is not None
        assert marketplace.hub_api is not None
    
    def test_patch_manifest_creation(self):
        """Test patch manifest creation and validation."""
        manifest = PatchManifest(
            patch_id="test-patch-123",
            name="Test Patch",
            version="1.0.0",
            patch_type=PatchType.LORA_ADAPTER,
            status=PatchStatus.PUBLISHED,
            security_level=SecurityLevel.COMMUNITY_VERIFIED,
            description="Test patch for integration testing",
            author="test-author",
            organization="test-org",
            base_models=["microsoft/DialoGPT-medium"],
            framework_versions={"transformers": ">=4.20.0"},
            benchmark_scores={"accuracy": 0.85},
            performance_delta={"accuracy": 0.05},
            license="apache-2.0"
        )
        
        # Should serialize and deserialize correctly
        manifest_dict = manifest.to_dict()
        assert manifest_dict["patch_id"] == "test-patch-123"
        assert manifest_dict["patch_type"] == "lora_adapter"
        
        # Should validate required fields
        assert manifest.patch_id
        assert manifest.name
        assert manifest.version
    
    @pytest.mark.asyncio
    async def test_patch_search_simulation(self):
        """Test patch search functionality (simulated)."""
        marketplace = PatchMarketplace()
        await marketplace.initialize()
        
        # This will use simulated search since we don't have real hub access
        from marketplace.patch_marketplace import PatchSearchQuery
        
        query = PatchSearchQuery(
            query="accuracy improvement",
            base_model="microsoft/DialoGPT-medium",
            min_score=0.6,
            limit=5
        )
        
        patches = await marketplace.search_patches(query)
        
        # Should return simulated results
        assert isinstance(patches, list)
        # In simulation mode, should return at least one patch
        if patches:
            assert patches[0].patch_id
            assert patches[0].name


class TestFailureMonitoring:
    """Test failure monitoring functionality."""
    
    def test_failure_recording(self, failure_monitor):
        """Test recording and retrieving failures."""
        failure = FailureContext(
            model_id="test-model",
            task_type="text_generation",
            domain="general",
            error_type="accuracy_drop",
            input_sample="Test input",
            expected_output="Expected",
            actual_output="Actual",
            severity="medium"
        )
        
        failure_id = failure_monitor.record_failure(failure)
        
        assert failure_id.startswith("failure_")
        assert len(failure_monitor.failure_history) == 1
        
        # Should update model health
        health = failure_monitor.get_model_health_summary("test-model")
        assert health["model_id"] == "test-model"
        assert health["total_failures"] == 1
    
    def test_failure_pattern_detection(self, failure_monitor):
        """Test failure pattern detection."""
        # Record multiple similar failures
        for i in range(5):
            failure = FailureContext(
                model_id="test-model",
                task_type="text_generation",
                domain="general",
                error_type="accuracy_drop",
                input_sample=f"Test input {i}",
                expected_output=f"Expected {i}",
                actual_output=f"Wrong {i}",
                severity="medium"
            )
            failure_monitor.record_failure(failure, auto_analyze=False)
        
        # Trigger pattern analysis
        failure_monitor._analyze_failure_patterns()
        
        # Should detect pattern
        patterns = failure_monitor.get_failure_patterns(min_frequency=3)
        assert len(patterns) > 0
        
        pattern = patterns[0]
        assert "accuracy_drop" in pattern.error_types
        assert "test-model" in pattern.affected_models
        assert pattern.frequency >= 3
    
    def test_healing_sample_extraction(self, failure_monitor, sample_failures):
        """Test extraction of failure samples for healing."""
        # Record sample failures
        for sample in sample_failures:
            failure = FailureContext(
                model_id="test-model",
                task_type="text_generation",
                domain="general",
                error_type=sample["error_type"],
                input_sample=sample["input"],
                expected_output=sample["expected_output"],
                actual_output=sample["actual_output"],
                severity=sample["severity"]
            )
            failure_monitor.record_failure(failure)
        
        # Extract samples for healing
        healing_samples = failure_monitor.get_failure_samples_for_healing(
            model_id="test-model",
            max_samples=10
        )
        
        assert len(healing_samples) == len(sample_failures)
        assert all("input" in sample for sample in healing_samples)
        assert all("expected_output" in sample for sample in healing_samples)


class TestAutoSurgery:
    """Test auto-surgery functionality."""
    
    def test_surgery_config_creation(self, temp_dir):
        """Test surgery configuration."""
        config = SurgeryConfig(
            output_dir=temp_dir,
            max_training_steps=50,
            learning_rate=1e-4
        )
        
        assert config.output_dir == temp_dir
        assert config.max_training_steps == 50
        assert config.learning_rate == 1e-4
        assert config.target_modules  # Should have default modules
    
    def test_surgery_system_initialization(self, surgery_config):
        """Test auto-surgery system initialization."""
        surgery = AutoModelSurgery(surgery_config)
        
        assert surgery.config == surgery_config
        assert Path(surgery.config.output_dir).exists()
        assert len(surgery.surgery_history) == 0
        assert len(surgery.active_adapters) == 0
    
    def test_training_data_preparation(self, surgery_config, sample_failures):
        """Test training data preparation from failures."""
        surgery = AutoModelSurgery(surgery_config)
        
        # This would normally require actual model training
        # For testing, we'll just verify the data preparation structure
        dataset = surgery._prepare_training_data(sample_failures)
        
        assert len(dataset) == len(sample_failures)
        
        # Test dataset item structure
        item = dataset[0]
        required_keys = {"input_ids", "attention_mask", "labels"}
        assert all(key in item for key in required_keys)


class TestIntegratedHealing:
    """Test integrated healing functionality."""
    
    @pytest.mark.asyncio
    async def test_auto_patch_context_creation(self):
        """Test auto patch context creation."""
        context = AutoPatchContext(
            model_id="test-model",
            task_type="text_generation",
            domain="general",
            failure_patterns=["accuracy_drop", "wrong_output"],
            performance_requirements={"accuracy": 0.03, "robustness": 0.02},
            tenant_id="test-tenant",
            security_constraints=SecurityLevel.COMMUNITY_VERIFIED
        )
        
        assert context.model_id == "test-model"
        assert context.task_type == "text_generation"
        assert len(context.failure_patterns) == 2
        assert context.performance_requirements["accuracy"] == 0.03
    
    @pytest.mark.asyncio
    async def test_patch_evaluator(self, sample_failures):
        """Test patch evaluation functionality."""
        evaluator = PatchEvaluator()
        
        # Create test patch
        patch = PatchManifest(
            patch_id="test-eval-patch",
            name="Test Evaluation Patch",
            version="1.0.0",
            patch_type=PatchType.LORA_ADAPTER,
            status=PatchStatus.PUBLISHED,
            security_level=SecurityLevel.COMMUNITY_VERIFIED,
            description="Test patch for evaluation",
            author="test-author",
            organization="test-org",
            base_models=["test-model"],
            performance_delta={"accuracy": 0.05}
        )
        
        # Create test context
        context = AutoPatchContext(
            model_id="test-model",
            task_type="text_generation",
            domain="general",
            failure_patterns=["accuracy_drop"],
            performance_requirements={"accuracy": 0.02}
        )
        
        # Evaluate patch
        results = await evaluator.evaluate_patch_on_failures(
            patch, sample_failures, context
        )
        
        assert "success_rate" in results
        assert "samples_passed" in results
        assert "samples_total" in results
        assert "performance_delta" in results
        assert results["samples_total"] == len(sample_failures)
    
    @pytest.mark.asyncio
    async def test_integrated_healing_system(self, surgery_config, sample_failures):
        """Test the complete integrated healing system."""
        healing_system = MarketplaceIntegratedHealing(surgery_config)
        
        context = AutoPatchContext(
            model_id="test-model",
            task_type="text_generation",
            domain="general",
            failure_patterns=["accuracy_drop"],
            performance_requirements={"accuracy": 0.02},
            max_patches=2,
            evaluation_required=False  # Skip evaluation for testing
        )
        
        # This test simulates the healing process
        # In a real environment, this would interact with actual models
        result = await healing_system.heal_with_marketplace(context, sample_failures)
        
        assert "approach" in result
        assert "success" in result
        assert "patches_tried" in result
        
        # Should attempt marketplace first, then fall back to surgery
        assert result["approach"] in ["marketplace_patch", "auto_surgery"]


class TestEndToEndIntegration:
    """End-to-end integration tests."""
    
    @pytest.mark.asyncio
    async def test_complete_healing_workflow(
        self, 
        surgery_config, 
        failure_monitor, 
        sample_failures
    ):
        """Test complete healing workflow from failure to resolution."""
        
        # 1. Record failures
        for sample in sample_failures:
            failure = FailureContext(
                model_id="integration-test-model",
                task_type="text_generation",
                domain="general",
                error_type=sample["error_type"],
                input_sample=sample["input"],
                expected_output=sample["expected_output"],
                actual_output=sample["actual_output"],
                severity=sample["severity"]
            )
            failure_monitor.record_failure(failure)
        
        # 2. Check model health
        health = failure_monitor.get_model_health_summary("integration-test-model")
        assert health["healing_recommended"] == True
        
        # 3. Extract healing samples
        healing_samples = failure_monitor.get_failure_samples_for_healing(
            "integration-test-model"
        )
        assert len(healing_samples) > 0
        
        # 4. Attempt integrated healing
        from marketplace.healing_integration import heal_with_marketplace_integration
        
        result = await heal_with_marketplace_integration(
            model_id="integration-test-model",
            task_type="text_generation",
            domain="general",
            failure_samples=healing_samples,
            surgery_config=surgery_config
        )
        
        # 5. Verify healing attempted
        assert "approach" in result
        assert result["approach"] in ["marketplace_patch", "auto_surgery"]
        
        # 6. Record success to update health
        failure_monitor.record_success(
            "integration-test-model",
            "text_generation", 
            "general"
        )
        
        # 7. Verify health improvement
        updated_health = failure_monitor.get_model_health_summary("integration-test-model")
        # Success rate should improve after recording success
        assert updated_health["success_rate"] > health["success_rate"]
    
    def test_system_statistics_aggregation(self, failure_monitor):
        """Test system-wide statistics aggregation."""
        
        # Record various failures and successes
        models = ["model-1", "model-2", "model-3"]
        
        for model in models:
            # Record some failures
            for i in range(3):
                failure = FailureContext(
                    model_id=model,
                    task_type="text_generation",
                    domain="general",
                    error_type="accuracy_drop",
                    input_sample=f"Test {i}",
                    severity="medium"
                )
                failure_monitor.record_failure(failure)
            
            # Record some successes
            for i in range(7):
                failure_monitor.record_success(model, "text_generation", "general")
        
        # Get system stats
        stats = failure_monitor.get_monitoring_stats()
        
        assert stats["monitored_models"] == len(models)
        assert stats["total_failures_recorded"] >= len(models) * 3
        
        # Check individual model health
        for model in models:
            health = failure_monitor.get_model_health_summary(model)
            assert health["model_id"] == model
            assert health["total_failures"] >= 3


def test_configuration_integration():
    """Test configuration system integration."""
    
    # Test that settings are accessible
    assert SETTINGS is not None
    
    # Test configuration structure
    config_dict = SETTINGS.to_dict()
    assert isinstance(config_dict, dict)
    
    # Should contain marketplace-related config
    # (These would be expanded based on actual config structure)
    assert "database" in config_dict
    assert "observability" in config_dict


if __name__ == "__main__":
    # Run specific test if called directly
    pytest.main([__file__, "-v"])