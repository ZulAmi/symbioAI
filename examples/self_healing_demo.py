#!/usr/bin/env python3
"""
Self-Healing Demo: Automatic Model Surgery pipeline

This demo simulates multiple fusion predictions, intentionally creating
low-confidence outcomes to trigger the self-healing pipeline. It shows:
- Failure detection and dataset curation
- LoRA adapter fine-tuning (if deps installed)
- Publishing of the adapter artifact
"""

import asyncio
import time
from models.adaptive_fusion import AdaptiveFusionEngine, ModelCapability
from models.inference_engine import InferenceRequest
from models.inference_engine import MockInferenceEngine, ModelConfig as InferModelConfig


async def main():
    fusion_engine = AdaptiveFusionEngine()

    # Register 2 mock models with different capabilities
    caps = [
        ModelCapability(
            model_id="generalist",
            strengths=["general"],
            weaknesses=["math"],
            accuracy_by_domain={"general": 0.75, "code": 0.6, "math": 0.55},
            latency_ms=120,
            throughput_qps=12,
            memory_usage_mb=1200,
            specialization_score=0.7,
            reliability_score=0.75,
            cost_per_request=0.02,
            supported_languages=["en"],
        ),
        ModelCapability(
            model_id="analyst",
            strengths=["analysis"],
            weaknesses=["creative"],
            accuracy_by_domain={"analysis": 0.78, "general": 0.7, "code": 0.65},
            latency_ms=140,
            throughput_qps=10,
            memory_usage_mb=1400,
            specialization_score=0.75,
            reliability_score=0.72,
            cost_per_request=0.025,
            supported_languages=["en"],
        ),
    ]

    for c in caps:
        mc = InferModelConfig(model_id=c.model_id, model_path=f"/tmp/{c.model_id}", max_batch_size=8)
        eng = MockInferenceEngine(mc)
        await eng.initialize()
        fusion_engine.register_model(c.model_id, eng, c)

    # Generate a series of requests likely to be low-confidence to trigger failures
    for i in range(60):
        req = InferenceRequest(
            id=f"req_{i}",
            inputs={"text": "Complex adversarial analysis question with conflicting requirements"},
            model_id="fusion",
            metadata={"task_type": "analysis", "quality_requirement": 0.95, "language": "en"},
        )
        await fusion_engine.fuse_predict(req)
        await asyncio.sleep(0.01)

    # Print a small snapshot
    stats = fusion_engine.get_fusion_stats()
    print("\nSelf-healing demo completed.")
    print(f"Failures recorded (approx): {len(fusion_engine.failure_monitor.get_recent_failures('generalist'))}")
    print(f"Strategy perf keys: {list(stats['strategy_performance'].keys())[:3]} ...")


if __name__ == "__main__":
    asyncio.run(main())
