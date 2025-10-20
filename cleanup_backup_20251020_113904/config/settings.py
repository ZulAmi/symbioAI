"""
Configuration management for Symbio AI system.
Handles loading and validation of system configuration.
"""

import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, field, fields, is_dataclass
from enum import Enum


class ModelType(Enum):
    """Supported model types."""
    BASE = "base"
    MERGED = "merged"
    DISTILLED = "distilled"
    ENSEMBLE = "ensemble"


class TrainingStrategy(Enum):
    """Training strategies."""
    EVOLUTIONARY = "evolutionary"
    GRADIENT_BASED = "gradient_based"
    REINFORCEMENT = "reinforcement"
    HYBRID = "hybrid"


@dataclass
class LoggingConfig:
    """Logging configuration."""
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file: str = "logs/symbio_ai.log"


@dataclass
class DataConfig:
    """Data management configuration."""
    base_path: str = "data/"
    preprocessing_workers: int = 4
    batch_size: int = 32
    cache_enabled: bool = True
    formats: list = field(default_factory=lambda: ["json", "csv", "parquet"])


@dataclass
class ModelConfig:
    """Model configuration."""
    registry_path: str = "models/registry/"
    default_type: ModelType = ModelType.BASE
    auto_optimization: bool = True
    quantization_enabled: bool = False
    supported_frameworks: list = field(default_factory=lambda: ["pytorch", "tensorflow", "jax"])


@dataclass
class AgentConfig:
    """Agent orchestration configuration."""
    max_concurrent_agents: int = 10
    communication_protocol: str = "async_message_passing"
    coordination_strategy: str = "hierarchical"
    failure_recovery: bool = True


@dataclass
class TrainingConfig:
    """Training configuration."""
    strategy: TrainingStrategy = TrainingStrategy.EVOLUTIONARY
    population_size: int = 50
    generations: int = 100
    mutation_rate: float = 0.1
    crossover_rate: float = 0.7
    selection_pressure: float = 1.2
    early_stopping: bool = True
    checkpoint_interval: int = 10


@dataclass
class EvaluationConfig:
    """Evaluation and benchmarking configuration."""
    benchmark_suites: list = field(default_factory=lambda: ["standard", "adversarial", "efficiency"])
    metrics: list = field(default_factory=lambda: ["accuracy", "latency", "memory", "robustness"])
    validation_split: float = 0.2
    test_split: float = 0.1
    cross_validation_folds: int = 5


@dataclass
class PolicyEngineConfig:
    """Policy engine and governance configuration."""

    enabled: bool = True
    retry_after_seconds: int = 60
    enforce_adapter_capabilities: bool = True
    default_policies: list = field(default_factory=lambda: ["quota_guard"])


@dataclass
class TelemetryConfig:
    """Telemetry and observability configuration."""

    log_observer: bool = False
    enable_default_dashboard: bool = True


@dataclass
class TenantBootstrapConfig:
    """Bootstrap tenants for control plane."""

    tenants: list = field(
        default_factory=lambda: [
            {
                "tenant_id": "public",
                "name": "Public Sandbox",
                "contact_email": "ops@symbio.ai",
                "enabled_policies": ["quota_guard"],
                "metadata": {"bootstrap": True},
            }
        ]
    )


@dataclass
class ObservabilityConfig:
    """Observability dashboard configuration."""

    dashboard_path: str = "monitoring/dashboards/default.json"


@dataclass
class SelfHealingConfig:
    """Self-healing and automatic model surgery configuration."""
    enabled: bool = True
    window_size: int = 5000
    confidence_threshold: float = 0.65
    entropy_threshold: float = 2.2
    failure_ratio_trigger: float = 0.2
    min_failures_trigger: int = 20
    surgery_cooldown_minutes: int = 30
    base_model: str = "gpt2"  # adapter base for LoRA
    max_steps: int = 200
    lora_r: int = 8
    lora_alpha: int = 32
    lora_dropout: float = 0.05


@dataclass
class SymbioConfig:
    """Main Symbio AI configuration."""
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    models: ModelConfig = field(default_factory=ModelConfig)
    agents: AgentConfig = field(default_factory=AgentConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    self_healing: SelfHealingConfig = field(default_factory=SelfHealingConfig)
    policies: PolicyEngineConfig = field(default_factory=PolicyEngineConfig)
    telemetry: TelemetryConfig = field(default_factory=TelemetryConfig)
    tenants: TenantBootstrapConfig = field(default_factory=TenantBootstrapConfig)
    observability: ObservabilityConfig = field(default_factory=ObservabilityConfig)
    
    # System-wide settings
    debug_mode: bool = False
    distributed_computing: bool = False
    gpu_enabled: bool = True
    random_seed: int = 42


def load_config(config_path: Optional[str] = None) -> SymbioConfig:
    """
    Load configuration from file or return default configuration.
    
    Args:
        config_path: Path to configuration file (YAML format)
        
    Returns:
        SymbioConfig instance
    """
    if config_path is None:
        config_path = "config/default.yaml"
    
    config_file = Path(config_path)
    
    if config_file.exists():
        with open(config_file, 'r') as f:
            config_dict = yaml.safe_load(f) or {}
            config = SymbioConfig()
            _merge_dataclass(config, config_dict)
            return config
    else:
        # Return default configuration and save it
        default_config = SymbioConfig()
        save_config(default_config, config_path)
        return default_config

def bootstrap_control_plane(config: SymbioConfig) -> None:
    """Bootstrap control plane primitives (tenants, policies, telemetry)."""

    from control_plane.tenancy import TENANT_REGISTRY, TenantSettings
    from registry.adapter_registry import AdapterMetadata, ADAPTER_REGISTRY
    from control_plane.telemetry import register_default_observers
    from monitoring.observability import OBSERVABILITY

    if config.telemetry.log_observer:
        register_default_observers()

    for tenant_cfg in config.tenants.tenants:
        tenant_id = tenant_cfg.get("tenant_id")
        if not tenant_id:
            continue
        if TENANT_REGISTRY.get_tenant(tenant_id):
            tenant = TENANT_REGISTRY.get_tenant(tenant_id)
        else:
            tenant = TenantSettings(
                tenant_id=tenant_id,
                name=tenant_cfg.get("name", tenant_id.title()),
                contact_email=tenant_cfg.get("contact_email", "ops@symbio.ai"),
                metadata=tenant_cfg.get("metadata", {}),
            )
            TENANT_REGISTRY.register_tenant(tenant)

        for policy_id in tenant_cfg.get("enabled_policies", []):
            TENANT_REGISTRY.enable_policy(tenant_id, policy_id)

        default_adapter = tenant_cfg.get("default_adapter")
        if default_adapter:
            TENANT_REGISTRY.assign_adapter(tenant_id, default_adapter)

    if config.observability.dashboard_path:
        try:
            OBSERVABILITY.load_dashboard_config()
        except FileNotFoundError:
            pass

    # Optionally register adapters defined in configuration
    for adapter_cfg in getattr(config.models, "preloaded_adapters", []):  # type: ignore[attr-defined]
        adapter_id = adapter_cfg.get("adapter_id")
        if not adapter_id:
            continue
        metadata = AdapterMetadata(
            adapter_id=adapter_id,
            name=adapter_cfg.get("name", adapter_id),
            version=adapter_cfg.get("version", "1.0.0"),
            capabilities=set(adapter_cfg.get("capabilities", [])),
            owner=adapter_cfg.get("owner"),
            lineage=adapter_cfg.get("lineage"),
            config={k: str(v) for k, v in adapter_cfg.get("config", {}).items()},
        )
        ADAPTER_REGISTRY.register_adapter(metadata)

def save_config(config: SymbioConfig, config_path: str) -> None:
    """
    Save configuration to file.
    
    Args:
        config: SymbioConfig instance to save
        config_path: Path where to save the configuration
    """
    config_file = Path(config_path)
    config_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert dataclass to dictionary for YAML serialization
    config_dict = _dataclass_to_dict(config)
    
    with open(config_file, 'w') as f:
        yaml.dump(config_dict, f, default_flow_style=False, indent=2)


def _dataclass_to_dict(obj) -> Dict[str, Any]:
    """Convert dataclass to dictionary, handling nested dataclasses and enums."""
    if hasattr(obj, '__dataclass_fields__'):
        result = {}
        for field_name, field_def in obj.__dataclass_fields__.items():
            value = getattr(obj, field_name)
            if hasattr(value, '__dataclass_fields__'):
                result[field_name] = _dataclass_to_dict(value)
            elif isinstance(value, Enum):
                result[field_name] = value.value
            else:
                result[field_name] = value
        return result
    else:
        return obj


def _merge_dataclass(instance: Any, updates: Dict[str, Any]) -> None:
    """Recursively merge dictionary values into a dataclass instance."""

    if not isinstance(updates, dict):
        return

    for field_obj in fields(instance):
        key = field_obj.name
        if key not in updates:
            continue
        value = updates[key]
        current = getattr(instance, key)

        if is_dataclass(current):
            _merge_dataclass(current, value)
        elif isinstance(current, Enum):
            setattr(instance, key, type(current)(value))
        else:
            setattr(instance, key, value)