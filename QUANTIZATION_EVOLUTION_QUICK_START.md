# Quantization-Aware Evolution - Quick Start Guide

## 5-Minute Quick Start

Deploy massive models on edge devices in minutes!

### Installation

```bash
# Already included in Symbio AI requirements
pip install -r requirements.txt
```

### Basic Usage

```python
import asyncio
from training.quantization_aware_evolution import (
    QuantizationEvolutionEngine,
    QuantizationEvolutionConfig,
    HardwareTarget
)

async def main():
    # 1. Define your model layers
    layer_names = [
        "embedding",
        "transformer_layer_0",
        "transformer_layer_1",
        "output"
    ]

    # 2. Configure evolution
    config = QuantizationEvolutionConfig(
        population_size=20,
        max_generations=30,
        hardware_target=HardwareTarget.ARM_NEON,
        allow_mixed_precision=True
    )

    # 3. Run evolution
    engine = QuantizationEvolutionEngine(config)
    results = await engine.run_evolution(layer_names)

    # 4. Get results
    best = engine.get_best_genome()
    print(f"✓ {best.compression_ratio:.1f}x compression @ {best.accuracy:.1%} accuracy")

    # 5. Export config
    engine.export_quantization_config(best, "quantization_config.json")

asyncio.run(main())
```

## Common Scenarios

### Mobile Deployment (Latency-Critical)

```python
config = QuantizationEvolutionConfig(
    hardware_target=HardwareTarget.ARM_NEON,
    target_latency_ms=50.0,
    latency_weight=0.5,
    accuracy_weight=0.3
)
```

### IoT Device (Energy-Critical)

```python
config = QuantizationEvolutionConfig(
    hardware_target=HardwareTarget.QUALCOMM_HEXAGON,
    target_energy_mj=0.3,
    energy_weight=0.6,
    compression_weight=0.3
)
```

### Extreme Compression (INT4/INT2)

```python
config = QuantizationEvolutionConfig(
    min_bits=2,  # Allow INT2
    max_bits=4,  # Max INT4
    compression_weight=0.5,
    accuracy_weight=0.3
)
```

## Run the Demo

```bash
cd "Symbio AI"
python examples/quantization_evolution_demo.py
```

You'll see:

- ✓ 8 comprehensive demos
- ✓ Hardware comparison
- ✓ Mixed-precision search
- ✓ Pareto front analysis
- ✓ Real-world deployment scenarios

## Configuration Cheatsheet

### Hardware Targets

```python
HardwareTarget.ARM_NEON              # Most mobile CPUs
HardwareTarget.ARM_DOT               # Modern ARM (ARMv8.2+)
HardwareTarget.QUALCOMM_HEXAGON      # Qualcomm Snapdragon DSP
HardwareTarget.APPLE_NEURAL_ENGINE   # Apple A/M-series
HardwareTarget.NVIDIA_TENSORRT       # NVIDIA GPUs
HardwareTarget.INTEL_VNNI            # Intel CPUs
```

### Quantization Precision

```python
# Conservative (high accuracy)
min_bits=6, max_bits=8

# Balanced (good trade-off)
min_bits=4, max_bits=8

# Aggressive (high compression)
min_bits=2, max_bits=4
```

### Objective Weights

```python
# Accuracy-first
accuracy_weight=0.6, compression_weight=0.2, latency_weight=0.2

# Compression-first
accuracy_weight=0.3, compression_weight=0.5, latency_weight=0.2

# Speed-first
accuracy_weight=0.3, compression_weight=0.2, latency_weight=0.5

# Energy-first
accuracy_weight=0.3, compression_weight=0.2, energy_weight=0.5
```

## Understanding Results

### Fitness Metrics

```python
best = engine.get_best_genome()

# Key metrics
best.accuracy               # 0.95 = 95% accuracy
best.compression_ratio      # 4.2 = 4.2x smaller
best.inference_latency_ms   # 45.2 = 45.2ms latency
best.memory_footprint_mb    # 125.3 = 125.3MB memory
best.energy_per_inference_mj # 0.42 = 0.42mJ per inference
```

### Per-Layer Configuration

```python
for layer_name, config in best.layer_configs.items():
    print(f"{layer_name}:")
    print(f"  Weights: {config.weight_bits}-bit")
    print(f"  Activations: {config.activation_bits}-bit")
    print(f"  Scheme: {config.weight_scheme.value}")
```

## Next Steps

1. **Explore full documentation**: `docs/quantization_aware_evolution.md`
2. **Run all demos**: `examples/quantization_evolution_demo.py`
3. **Integrate with your model**: Use exported config
4. **Deploy to hardware**: Test on target device

## Troubleshooting

### Evolution taking too long?

```python
# Reduce population or generations
config = QuantizationEvolutionConfig(
    population_size=10,      # Down from 20
    max_generations=20,      # Down from 50
    parallel_evaluations=8   # Use more cores
)
```

### Accuracy loss too high?

```python
# Increase minimum bits
config = QuantizationEvolutionConfig(
    min_bits=6,             # Up from 4
    accuracy_weight=0.6     # Prioritize accuracy
)
```

### Latency target not met?

```python
# Choose faster hardware or relax constraint
config = QuantizationEvolutionConfig(
    hardware_target=HardwareTarget.APPLE_NEURAL_ENGINE,  # Faster
    target_latency_ms=75.0  # Relax from 50ms
)
```

## Performance Expectations

| Precision | Compression | Accuracy Loss | Latency Speedup |
| --------- | ----------- | ------------- | --------------- |
| INT8      | 4x          | <0.5%         | 2-3x            |
| INT4      | 8x          | 1-2%          | 3-4x            |
| Mixed 4/8 | 5-6x        | <1%           | 2.5-3.5x        |
| INT2/4    | 12x+        | 2-4%          | 4-5x            |

## Getting Help

- Full docs: `docs/quantization_aware_evolution.md`
- Examples: `examples/quantization_evolution_demo.py`
- Issues: Check error messages, adjust config
- Support: See main README.md

---

**Ready to deploy massive models on edge devices? Let's go! 🚀**
