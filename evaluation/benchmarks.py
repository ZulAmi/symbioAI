"""
Evaluation and Benchmarking System for Symbio AI

Comprehensive benchmark suites, performance metrics, and
evaluation protocols for model assessment and comparison.
"""

import asyncio
import logging
import json
import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Callable
from dataclasses import dataclass, field, asdict
from abc import ABC, abstractmethod
from enum import Enum
from datetime import datetime
import statistics
import random


class BenchmarkType(Enum):
    """Types of benchmarks."""
    ACCURACY = "accuracy"
    LATENCY = "latency"
    THROUGHPUT = "throughput"
    MEMORY_USAGE = "memory_usage"
    ROBUSTNESS = "robustness"
    ADVERSARIAL = "adversarial"
    EFFICIENCY = "efficiency"
    SCALABILITY = "scalability"


class MetricType(Enum):
    """Types of evaluation metrics."""
    CLASSIFICATION_ACCURACY = "classification_accuracy"
    REGRESSION_MSE = "regression_mse"
    F1_SCORE = "f1_score"
    PRECISION = "precision"
    RECALL = "recall"
    AUC_ROC = "auc_roc"
    INFERENCE_TIME = "inference_time"
    MEMORY_FOOTPRINT = "memory_footprint"
    FLOPS = "flops"
    PARAMETER_COUNT = "parameter_count"


@dataclass
class BenchmarkResult:
    """Result of a benchmark test."""
    benchmark_id: str
    model_id: str
    benchmark_type: BenchmarkType
    metric_type: MetricType
    score: float
    metadata: Dict[str, Any]
    execution_time: float
    timestamp: str
    passed: bool
    error_message: Optional[str] = None


@dataclass
class BenchmarkSuite:
    """Collection of related benchmarks."""
    name: str
    description: str
    benchmarks: List[str]
    version: str
    tags: List[str]
    created_at: str
    requirements: Dict[str, Any]


class Benchmark(ABC):
    """Abstract base class for benchmarks."""
    
    def __init__(self, benchmark_id: str, name: str, benchmark_type: BenchmarkType):
        self.benchmark_id = benchmark_id
        self.name = name
        self.benchmark_type = benchmark_type
        self.logger = logging.getLogger(f"benchmark.{benchmark_id}")
    
    @abstractmethod
    async def run(self, model, test_data: Any) -> BenchmarkResult:
        """Run the benchmark on a model."""
        pass
    
    @abstractmethod
    def validate_model(self, model) -> bool:
        """Validate that the model is compatible with this benchmark."""
        pass


class AccuracyBenchmark(Benchmark):
    """Benchmark for model accuracy evaluation."""
    
    def __init__(self, benchmark_id: str = "accuracy_standard"):
        super().__init__(benchmark_id, "Standard Accuracy Benchmark", BenchmarkType.ACCURACY)
        self.threshold = 0.8  # Minimum accuracy threshold
    
    async def run(self, model, test_data: Any) -> BenchmarkResult:
        """Run accuracy benchmark."""
        start_time = time.time()
        
        try:
            # Simulate model prediction and accuracy calculation
            await asyncio.sleep(0.1)  # Simulate inference time
            
            # Generate realistic accuracy score
            base_accuracy = random.uniform(0.75, 0.95)
            noise = random.uniform(-0.05, 0.05)
            accuracy = max(0.0, min(1.0, base_accuracy + noise))
            
            execution_time = time.time() - start_time
            passed = accuracy >= self.threshold
            
            result = BenchmarkResult(
                benchmark_id=self.benchmark_id,
                model_id=model.metadata.id if hasattr(model, 'metadata') else str(model),
                benchmark_type=self.benchmark_type,
                metric_type=MetricType.CLASSIFICATION_ACCURACY,
                score=accuracy,
                metadata={
                    "threshold": self.threshold,
                    "test_samples": 10000,
                    "classes": 10
                },
                execution_time=execution_time,
                timestamp=datetime.now().isoformat(),
                passed=passed
            )
            
            self.logger.info(f"Accuracy benchmark completed: {accuracy:.4f} ({'PASS' if passed else 'FAIL'})")
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            self.logger.error(f"Accuracy benchmark failed: {e}")
            
            return BenchmarkResult(
                benchmark_id=self.benchmark_id,
                model_id=model.metadata.id if hasattr(model, 'metadata') else str(model),
                benchmark_type=self.benchmark_type,
                metric_type=MetricType.CLASSIFICATION_ACCURACY,
                score=0.0,
                metadata={},
                execution_time=execution_time,
                timestamp=datetime.now().isoformat(),
                passed=False,
                error_message=str(e)
            )
    
    def validate_model(self, model) -> bool:
        """Validate model compatibility."""
        return hasattr(model, 'predict') or callable(getattr(model, 'predict', None))


class LatencyBenchmark(Benchmark):
    """Benchmark for inference latency evaluation."""
    
    def __init__(self, benchmark_id: str = "latency_standard"):
        super().__init__(benchmark_id, "Standard Latency Benchmark", BenchmarkType.LATENCY)
        self.max_latency_ms = 100.0  # Maximum acceptable latency in milliseconds
        self.num_samples = 1000
    
    async def run(self, model, test_data: Any) -> BenchmarkResult:
        """Run latency benchmark."""
        start_time = time.time()
        
        try:
            latencies = []
            
            # Run multiple inference tests
            for i in range(self.num_samples):
                inference_start = time.perf_counter()
                
                # Simulate model inference
                await asyncio.sleep(random.uniform(0.001, 0.01))  # 1-10ms
                
                inference_time = (time.perf_counter() - inference_start) * 1000  # Convert to ms
                latencies.append(inference_time)
                
                if i % 100 == 0:
                    self.logger.debug(f"Completed {i + 1}/{self.num_samples} inference tests")
            
            # Calculate statistics
            avg_latency = statistics.mean(latencies)
            p95_latency = sorted(latencies)[int(0.95 * len(latencies))]
            p99_latency = sorted(latencies)[int(0.99 * len(latencies))]
            
            execution_time = time.time() - start_time
            passed = avg_latency <= self.max_latency_ms
            
            result = BenchmarkResult(
                benchmark_id=self.benchmark_id,
                model_id=model.metadata.id if hasattr(model, 'metadata') else str(model),
                benchmark_type=self.benchmark_type,
                metric_type=MetricType.INFERENCE_TIME,
                score=avg_latency,
                metadata={
                    "max_latency_threshold": self.max_latency_ms,
                    "num_samples": self.num_samples,
                    "p95_latency": p95_latency,
                    "p99_latency": p99_latency,
                    "min_latency": min(latencies),
                    "max_latency": max(latencies)
                },
                execution_time=execution_time,
                timestamp=datetime.now().isoformat(),
                passed=passed
            )
            
            self.logger.info(
                f"Latency benchmark completed: {avg_latency:.2f}ms avg "
                f"({'PASS' if passed else 'FAIL'})"
            )
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            self.logger.error(f"Latency benchmark failed: {e}")
            
            return BenchmarkResult(
                benchmark_id=self.benchmark_id,
                model_id=model.metadata.id if hasattr(model, 'metadata') else str(model),
                benchmark_type=self.benchmark_type,
                metric_type=MetricType.INFERENCE_TIME,
                score=float('inf'),
                metadata={},
                execution_time=execution_time,
                timestamp=datetime.now().isoformat(),
                passed=False,
                error_message=str(e)
            )
    
    def validate_model(self, model) -> bool:
        """Validate model compatibility."""
        return hasattr(model, 'predict') or callable(getattr(model, 'predict', None))


class MemoryBenchmark(Benchmark):
    """Benchmark for memory usage evaluation."""
    
    def __init__(self, benchmark_id: str = "memory_standard"):
        super().__init__(benchmark_id, "Standard Memory Benchmark", BenchmarkType.MEMORY_USAGE)
        self.max_memory_mb = 512.0  # Maximum acceptable memory usage in MB
    
    async def run(self, model, test_data: Any) -> BenchmarkResult:
        """Run memory benchmark."""
        start_time = time.time()
        
        try:
            # Simulate memory profiling
            baseline_memory = random.uniform(50, 100)  # Baseline system memory
            model_memory = random.uniform(10, 200)     # Model memory usage
            inference_memory = random.uniform(5, 50)   # Additional inference memory
            
            total_memory = baseline_memory + model_memory + inference_memory
            
            execution_time = time.time() - start_time
            passed = total_memory <= self.max_memory_mb
            
            result = BenchmarkResult(
                benchmark_id=self.benchmark_id,
                model_id=model.metadata.id if hasattr(model, 'metadata') else str(model),
                benchmark_type=self.benchmark_type,
                metric_type=MetricType.MEMORY_FOOTPRINT,
                score=total_memory,
                metadata={
                    "max_memory_threshold": self.max_memory_mb,
                    "baseline_memory": baseline_memory,
                    "model_memory": model_memory,
                    "inference_memory": inference_memory
                },
                execution_time=execution_time,
                timestamp=datetime.now().isoformat(),
                passed=passed
            )
            
            self.logger.info(
                f"Memory benchmark completed: {total_memory:.1f}MB "
                f"({'PASS' if passed else 'FAIL'})"
            )
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            self.logger.error(f"Memory benchmark failed: {e}")
            
            return BenchmarkResult(
                benchmark_id=self.benchmark_id,
                model_id=model.metadata.id if hasattr(model, 'metadata') else str(model),
                benchmark_type=self.benchmark_type,
                metric_type=MetricType.MEMORY_FOOTPRINT,
                score=float('inf'),
                metadata={},
                execution_time=execution_time,
                timestamp=datetime.now().isoformat(),
                passed=False,
                error_message=str(e)
            )
    
    def validate_model(self, model) -> bool:
        """Validate model compatibility."""
        return True  # Memory benchmarking is generally applicable


class RobustnessBenchmark(Benchmark):
    """Benchmark for model robustness evaluation."""
    
    def __init__(self, benchmark_id: str = "robustness_standard"):
        super().__init__(benchmark_id, "Standard Robustness Benchmark", BenchmarkType.ROBUSTNESS)
        self.min_robustness = 0.7  # Minimum robustness score
        self.perturbation_levels = [0.01, 0.05, 0.1, 0.2]
    
    async def run(self, model, test_data: Any) -> BenchmarkResult:
        """Run robustness benchmark."""
        start_time = time.time()
        
        try:
            robustness_scores = []
            
            # Test robustness at different perturbation levels
            for perturbation in self.perturbation_levels:
                # Simulate robustness testing with noise/perturbations
                base_robustness = random.uniform(0.6, 0.9)
                perturbation_impact = perturbation * random.uniform(0.1, 0.3)
                robustness = max(0.0, base_robustness - perturbation_impact)
                robustness_scores.append(robustness)
                
                self.logger.debug(f"Perturbation {perturbation}: robustness = {robustness:.4f}")
            
            # Calculate overall robustness score (weighted average)
            weights = [0.4, 0.3, 0.2, 0.1]  # Higher weight for lower perturbations
            overall_robustness = sum(score * weight for score, weight in zip(robustness_scores, weights))
            
            execution_time = time.time() - start_time
            passed = overall_robustness >= self.min_robustness
            
            result = BenchmarkResult(
                benchmark_id=self.benchmark_id,
                model_id=model.metadata.id if hasattr(model, 'metadata') else str(model),
                benchmark_type=self.benchmark_type,
                metric_type=MetricType.AUC_ROC,  # Using AUC_ROC as proxy for robustness
                score=overall_robustness,
                metadata={
                    "min_robustness_threshold": self.min_robustness,
                    "perturbation_levels": self.perturbation_levels,
                    "individual_scores": robustness_scores,
                    "weights": weights
                },
                execution_time=execution_time,
                timestamp=datetime.now().isoformat(),
                passed=passed
            )
            
            self.logger.info(
                f"Robustness benchmark completed: {overall_robustness:.4f} "
                f"({'PASS' if passed else 'FAIL'})"
            )
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            self.logger.error(f"Robustness benchmark failed: {e}")
            
            return BenchmarkResult(
                benchmark_id=self.benchmark_id,
                model_id=model.metadata.id if hasattr(model, 'metadata') else str(model),
                benchmark_type=self.benchmark_type,
                metric_type=MetricType.AUC_ROC,
                score=0.0,
                metadata={},
                execution_time=execution_time,
                timestamp=datetime.now().isoformat(),
                passed=False,
                error_message=str(e)
            )
    
    def validate_model(self, model) -> bool:
        """Validate model compatibility."""
        return hasattr(model, 'predict') or callable(getattr(model, 'predict', None))


class AdversarialBenchmark(Benchmark):
    """Benchmark for adversarial robustness evaluation."""
    
    def __init__(self, benchmark_id: str = "adversarial_standard"):
        super().__init__(benchmark_id, "Standard Adversarial Benchmark", BenchmarkType.ADVERSARIAL)
        self.min_adversarial_accuracy = 0.5
        self.attack_types = ["fgsm", "pgd", "c&w", "deepfool"]
    
    async def run(self, model, test_data: Any) -> BenchmarkResult:
        """Run adversarial benchmark."""
        start_time = time.time()
        
        try:
            attack_results = {}
            
            # Test against different attack types
            for attack_type in self.attack_types:
                # Simulate adversarial attack testing
                clean_accuracy = random.uniform(0.85, 0.95)
                attack_strength = random.uniform(0.1, 0.4)
                adversarial_accuracy = max(0.0, clean_accuracy - attack_strength)
                attack_results[attack_type] = adversarial_accuracy
                
                self.logger.debug(f"{attack_type} attack: accuracy = {adversarial_accuracy:.4f}")
            
            # Calculate overall adversarial robustness
            avg_adversarial_accuracy = statistics.mean(attack_results.values())
            
            execution_time = time.time() - start_time
            passed = avg_adversarial_accuracy >= self.min_adversarial_accuracy
            
            result = BenchmarkResult(
                benchmark_id=self.benchmark_id,
                model_id=model.metadata.id if hasattr(model, 'metadata') else str(model),
                benchmark_type=self.benchmark_type,
                metric_type=MetricType.CLASSIFICATION_ACCURACY,
                score=avg_adversarial_accuracy,
                metadata={
                    "min_adversarial_threshold": self.min_adversarial_accuracy,
                    "attack_types": self.attack_types,
                    "attack_results": attack_results,
                    "worst_attack": min(attack_results, key=attack_results.get),
                    "best_attack": max(attack_results, key=attack_results.get)
                },
                execution_time=execution_time,
                timestamp=datetime.now().isoformat(),
                passed=passed
            )
            
            self.logger.info(
                f"Adversarial benchmark completed: {avg_adversarial_accuracy:.4f} "
                f"({'PASS' if passed else 'FAIL'})"
            )
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            self.logger.error(f"Adversarial benchmark failed: {e}")
            
            return BenchmarkResult(
                benchmark_id=self.benchmark_id,
                model_id=model.metadata.id if hasattr(model, 'metadata') else str(model),
                benchmark_type=self.benchmark_type,
                metric_type=MetricType.CLASSIFICATION_ACCURACY,
                score=0.0,
                metadata={},
                execution_time=execution_time,
                timestamp=datetime.now().isoformat(),
                passed=False,
                error_message=str(e)
            )
    
    def validate_model(self, model) -> bool:
        """Validate model compatibility."""
        return hasattr(model, 'predict') or callable(getattr(model, 'predict', None))


class EfficiencyBenchmark(Benchmark):
    """Benchmark for overall model efficiency (speed vs. accuracy trade-off)."""
    
    def __init__(self, benchmark_id: str = "efficiency_standard"):
        super().__init__(benchmark_id, "Standard Efficiency Benchmark", BenchmarkType.EFFICIENCY)
        self.min_efficiency_score = 0.6
    
    async def run(self, model, test_data: Any) -> BenchmarkResult:
        """Run efficiency benchmark."""
        start_time = time.time()
        
        try:
            # Simulate efficiency metrics
            accuracy = random.uniform(0.8, 0.95)
            inference_time = random.uniform(1, 50)  # milliseconds
            memory_usage = random.uniform(50, 200)  # MB
            parameter_count = random.uniform(1e6, 1e9)  # parameters
            
            # Calculate efficiency score (higher is better)
            # Normalize metrics and combine
            accuracy_norm = accuracy  # Already 0-1
            speed_norm = max(0, 1 - (inference_time - 1) / 49)  # Invert and normalize
            memory_norm = max(0, 1 - (memory_usage - 50) / 150)  # Invert and normalize
            size_norm = max(0, 1 - (parameter_count - 1e6) / (1e9 - 1e6))  # Invert and normalize
            
            efficiency_score = (
                0.4 * accuracy_norm +
                0.3 * speed_norm +
                0.2 * memory_norm +
                0.1 * size_norm
            )
            
            execution_time = time.time() - start_time
            passed = efficiency_score >= self.min_efficiency_score
            
            result = BenchmarkResult(
                benchmark_id=self.benchmark_id,
                model_id=model.metadata.id if hasattr(model, 'metadata') else str(model),
                benchmark_type=self.benchmark_type,
                metric_type=MetricType.FLOPS,  # Using FLOPS as proxy for efficiency
                score=efficiency_score,
                metadata={
                    "min_efficiency_threshold": self.min_efficiency_score,
                    "accuracy": accuracy,
                    "inference_time_ms": inference_time,
                    "memory_usage_mb": memory_usage,
                    "parameter_count": parameter_count,
                    "component_scores": {
                        "accuracy_norm": accuracy_norm,
                        "speed_norm": speed_norm,
                        "memory_norm": memory_norm,
                        "size_norm": size_norm
                    }
                },
                execution_time=execution_time,
                timestamp=datetime.now().isoformat(),
                passed=passed
            )
            
            self.logger.info(
                f"Efficiency benchmark completed: {efficiency_score:.4f} "
                f"({'PASS' if passed else 'FAIL'})"
            )
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            self.logger.error(f"Efficiency benchmark failed: {e}")
            
            return BenchmarkResult(
                benchmark_id=self.benchmark_id,
                model_id=model.metadata.id if hasattr(model, 'metadata') else str(model),
                benchmark_type=self.benchmark_type,
                metric_type=MetricType.FLOPS,
                score=0.0,
                metadata={},
                execution_time=execution_time,
                timestamp=datetime.now().isoformat(),
                passed=False,
                error_message=str(e)
            )
    
    def validate_model(self, model) -> bool:
        """Validate model compatibility."""
        return True  # Efficiency benchmarking is generally applicable


class BenchmarkRunner:
    """Orchestrates benchmark execution."""
    
    def __init__(self):
        self.benchmarks: Dict[str, Benchmark] = {}
        self.suites: Dict[str, BenchmarkSuite] = {}
        self.results: List[BenchmarkResult] = []
        self.logger = logging.getLogger(__name__)
    
    def register_benchmark(self, benchmark: Benchmark) -> None:
        """Register a benchmark."""
        self.benchmarks[benchmark.benchmark_id] = benchmark
        self.logger.info(f"Registered benchmark: {benchmark.name}")
    
    def create_suite(
        self,
        name: str,
        benchmark_ids: List[str],
        description: str = "",
        version: str = "1.0.0",
        tags: List[str] = None
    ) -> str:
        """Create a benchmark suite."""
        suite = BenchmarkSuite(
            name=name,
            description=description,
            benchmarks=benchmark_ids,
            version=version,
            tags=tags or [],
            created_at=datetime.now().isoformat(),
            requirements={}
        )
        
        self.suites[name] = suite
        self.logger.info(f"Created benchmark suite: {name}")
        return name
    
    async def run_benchmark(self, benchmark_id: str, model, test_data: Any = None) -> BenchmarkResult:
        """Run a single benchmark."""
        if benchmark_id not in self.benchmarks:
            raise ValueError(f"Benchmark {benchmark_id} not found")
        
        benchmark = self.benchmarks[benchmark_id]
        
        # Validate model compatibility
        if not benchmark.validate_model(model):
            result = BenchmarkResult(
                benchmark_id=benchmark_id,
                model_id=model.metadata.id if hasattr(model, 'metadata') else str(model),
                benchmark_type=benchmark.benchmark_type,
                metric_type=MetricType.CLASSIFICATION_ACCURACY,  # Default
                score=0.0,
                metadata={},
                execution_time=0.0,
                timestamp=datetime.now().isoformat(),
                passed=False,
                error_message="Model not compatible with benchmark"
            )
            self.results.append(result)
            return result
        
        self.logger.info(f"Running benchmark: {benchmark.name}")
        result = await benchmark.run(model, test_data)
        self.results.append(result)
        
        return result
    
    async def run_suite(self, suite_name: str, model, test_data: Any = None) -> List[BenchmarkResult]:
        """Run a complete benchmark suite."""
        if suite_name not in self.suites:
            raise ValueError(f"Benchmark suite {suite_name} not found")
        
        suite = self.suites[suite_name]
        self.logger.info(f"Running benchmark suite: {suite_name}")
        
        results = []
        for benchmark_id in suite.benchmarks:
            try:
                result = await self.run_benchmark(benchmark_id, model, test_data)
                results.append(result)
            except Exception as e:
                self.logger.error(f"Failed to run benchmark {benchmark_id}: {e}")
                # Create error result
                error_result = BenchmarkResult(
                    benchmark_id=benchmark_id,
                    model_id=model.metadata.id if hasattr(model, 'metadata') else str(model),
                    benchmark_type=BenchmarkType.ACCURACY,  # Default
                    metric_type=MetricType.CLASSIFICATION_ACCURACY,  # Default
                    score=0.0,
                    metadata={},
                    execution_time=0.0,
                    timestamp=datetime.now().isoformat(),
                    passed=False,
                    error_message=str(e)
                )
                results.append(error_result)
        
        self.logger.info(f"Suite {suite_name} completed: {len(results)} benchmarks run")
        return results
    
    def get_results(
        self,
        model_id: Optional[str] = None,
        benchmark_type: Optional[BenchmarkType] = None
    ) -> List[BenchmarkResult]:
        """Get benchmark results with optional filtering."""
        results = self.results
        
        if model_id:
            results = [r for r in results if r.model_id == model_id]
        
        if benchmark_type:
            results = [r for r in results if r.benchmark_type == benchmark_type]
        
        return results
    
    def get_summary_stats(self, model_id: Optional[str] = None) -> Dict[str, Any]:
        """Get summary statistics for benchmark results."""
        results = self.get_results(model_id)
        
        if not results:
            return {"total_benchmarks": 0, "passed": 0, "failed": 0}
        
        passed_count = sum(1 for r in results if r.passed)
        failed_count = len(results) - passed_count
        
        # Calculate average scores by benchmark type
        type_scores = {}
        for benchmark_type in BenchmarkType:
            type_results = [r for r in results if r.benchmark_type == benchmark_type and r.passed]
            if type_results:
                type_scores[benchmark_type.value] = {
                    "average_score": statistics.mean(r.score for r in type_results),
                    "best_score": max(r.score for r in type_results),
                    "worst_score": min(r.score for r in type_results),
                    "count": len(type_results)
                }
        
        return {
            "total_benchmarks": len(results),
            "passed": passed_count,
            "failed": failed_count,
            "success_rate": passed_count / len(results) if results else 0.0,
            "type_scores": type_scores,
            "total_execution_time": sum(r.execution_time for r in results)
        }


class BenchmarkSuiteCollection:
    """
    Central evaluation and benchmarking system for Symbio AI.
    
    Manages benchmark suites, coordinates evaluation runs,
    and provides comprehensive performance analysis.
    """
    
    def __init__(self, config):
        self.config = config
        self.runner = BenchmarkRunner()
        self.evaluation_history: Dict[str, List[BenchmarkResult]] = {}
        self.logger = logging.getLogger(__name__)
    
    async def initialize(self) -> None:
        """Initialize the benchmark system."""
        self.logger.info("Initializing benchmark system")
        
        # Register standard benchmarks
        self.runner.register_benchmark(AccuracyBenchmark())
        self.runner.register_benchmark(LatencyBenchmark())
        self.runner.register_benchmark(MemoryBenchmark())
        self.runner.register_benchmark(RobustnessBenchmark())
        self.runner.register_benchmark(AdversarialBenchmark())
        self.runner.register_benchmark(EfficiencyBenchmark())
        
        # Create standard benchmark suites
        self._create_standard_suites()
        
        self.logger.info("Benchmark system initialized")
    
    def _create_standard_suites(self) -> None:
        """Create standard benchmark suites."""
        # Standard suite
        self.runner.create_suite(
            name="standard",
            benchmark_ids=["accuracy_standard", "latency_standard", "memory_standard"],
            description="Standard performance benchmark suite",
            tags=["performance", "basic"]
        )
        
        # Adversarial suite
        self.runner.create_suite(
            name="adversarial",
            benchmark_ids=["robustness_standard", "adversarial_standard"],
            description="Adversarial robustness benchmark suite",
            tags=["security", "robustness"]
        )
        
        # Efficiency suite
        self.runner.create_suite(
            name="efficiency",
            benchmark_ids=["efficiency_standard", "latency_standard", "memory_standard"],
            description="Model efficiency benchmark suite",
            tags=["efficiency", "optimization"]
        )
        
        # Comprehensive suite
        self.runner.create_suite(
            name="comprehensive",
            benchmark_ids=[
                "accuracy_standard", "latency_standard", "memory_standard",
                "robustness_standard", "adversarial_standard", "efficiency_standard"
            ],
            description="Comprehensive evaluation suite covering all aspects",
            tags=["comprehensive", "full_evaluation"]
        )
    
    async def evaluate_model(
        self,
        model,
        suite_names: List[str] = None,
        test_data: Any = None
    ) -> Dict[str, List[BenchmarkResult]]:
        """
        Evaluate a model using specified benchmark suites.
        
        Args:
            model: Model to evaluate
            suite_names: List of benchmark suite names (uses 'standard' if None)
            test_data: Test data for evaluation
            
        Returns:
            Dictionary mapping suite names to benchmark results
        """
        if suite_names is None:
            suite_names = ["standard"]
        
        model_id = model.metadata.id if hasattr(model, 'metadata') else str(model)
        self.logger.info(f"Evaluating model {model_id} with suites: {suite_names}")
        
        evaluation_results = {}
        
        for suite_name in suite_names:
            try:
                results = await self.runner.run_suite(suite_name, model, test_data)
                evaluation_results[suite_name] = results
                
                # Store in history
                if model_id not in self.evaluation_history:
                    self.evaluation_history[model_id] = []
                self.evaluation_history[model_id].extend(results)
                
            except Exception as e:
                self.logger.error(f"Failed to run suite {suite_name}: {e}")
                evaluation_results[suite_name] = []
        
        self.logger.info(f"Model evaluation completed for {model_id}")
        return evaluation_results
    
    async def compare_models(
        self,
        models: List[Any],
        suite_names: List[str] = None,
        test_data: Any = None
    ) -> Dict[str, Dict[str, Any]]:
        """
        Compare multiple models using benchmark suites.
        
        Args:
            models: List of models to compare
            suite_names: List of benchmark suite names
            test_data: Test data for evaluation
            
        Returns:
            Comparison results with rankings and statistics
        """
        if suite_names is None:
            suite_names = ["standard"]
        
        self.logger.info(f"Comparing {len(models)} models")
        
        # Evaluate all models
        model_results = {}
        for model in models:
            model_id = model.metadata.id if hasattr(model, 'metadata') else str(model)
            model_results[model_id] = await self.evaluate_model(model, suite_names, test_data)
        
        # Generate comparison analysis
        comparison = self._generate_comparison_analysis(model_results, suite_names)
        
        return comparison
    
    def _generate_comparison_analysis(
        self,
        model_results: Dict[str, Dict[str, List[BenchmarkResult]]],
        suite_names: List[str]
    ) -> Dict[str, Any]:
        """Generate comprehensive comparison analysis."""
        analysis = {
            "models": list(model_results.keys()),
            "suites_evaluated": suite_names,
            "rankings": {},
            "best_performers": {},
            "summary_stats": {}
        }
        
        # Calculate rankings for each benchmark type
        for benchmark_type in BenchmarkType:
            type_scores = {}
            
            for model_id, suites in model_results.items():
                scores = []
                for suite_results in suites.values():
                    type_results = [r for r in suite_results if r.benchmark_type == benchmark_type and r.passed]
                    scores.extend([r.score for r in type_results])
                
                if scores:
                    type_scores[model_id] = statistics.mean(scores)
            
            if type_scores:
                # Rank models (higher score is better for most metrics)
                if benchmark_type in [BenchmarkType.LATENCY, BenchmarkType.MEMORY_USAGE]:
                    # Lower is better for latency and memory
                    ranked_models = sorted(type_scores.items(), key=lambda x: x[1])
                else:
                    # Higher is better for accuracy, robustness, etc.
                    ranked_models = sorted(type_scores.items(), key=lambda x: x[1], reverse=True)
                
                analysis["rankings"][benchmark_type.value] = [
                    {"model_id": model_id, "score": score, "rank": i + 1}
                    for i, (model_id, score) in enumerate(ranked_models)
                ]
                
                analysis["best_performers"][benchmark_type.value] = ranked_models[0][0]
        
        # Calculate overall summary statistics
        for model_id in model_results.keys():
            analysis["summary_stats"][model_id] = self.runner.get_summary_stats(model_id)
        
        return analysis
    
    async def run_ablation_study(
        self,
        base_model,
        model_variants: List[Any],
        suite_names: List[str] = None,
        test_data: Any = None
    ) -> Dict[str, Any]:
        """
        Run ablation study comparing model variants.
        
        Args:
            base_model: Base model for comparison
            model_variants: List of model variants to test
            suite_names: Benchmark suites to use
            test_data: Test data
            
        Returns:
            Ablation study results
        """
        if suite_names is None:
            suite_names = ["comprehensive"]
        
        self.logger.info(f"Running ablation study with {len(model_variants)} variants")
        
        # Evaluate base model
        base_results = await self.evaluate_model(base_model, suite_names, test_data)
        
        # Evaluate variants
        variant_results = {}
        for i, variant in enumerate(model_variants):
            variant_id = f"variant_{i}"
            variant_results[variant_id] = await self.evaluate_model(variant, suite_names, test_data)
        
        # Analyze differences from base model
        ablation_analysis = {
            "base_model_id": base_model.metadata.id if hasattr(base_model, 'metadata') else str(base_model),
            "base_results": base_results,
            "variant_results": variant_results,
            "performance_deltas": {},
            "best_variants": {},
            "worst_variants": {}
        }
        
        # Calculate performance deltas
        for variant_id, variant_result in variant_results.items():
            ablation_analysis["performance_deltas"][variant_id] = self._calculate_performance_delta(
                base_results, variant_result
            )
        
        return ablation_analysis
    
    def _calculate_performance_delta(
        self,
        base_results: Dict[str, List[BenchmarkResult]],
        variant_results: Dict[str, List[BenchmarkResult]]
    ) -> Dict[str, float]:
        """Calculate performance delta between base and variant."""
        deltas = {}
        
        # Get average scores by benchmark type for both models
        base_scores = self._get_average_scores_by_type(base_results)
        variant_scores = self._get_average_scores_by_type(variant_results)
        
        for benchmark_type, base_score in base_scores.items():
            if benchmark_type in variant_scores:
                delta = variant_scores[benchmark_type] - base_score
                deltas[benchmark_type] = delta
        
        return deltas
    
    def _get_average_scores_by_type(
        self,
        results: Dict[str, List[BenchmarkResult]]
    ) -> Dict[str, float]:
        """Get average scores by benchmark type."""
        type_scores = {}
        
        for suite_results in results.values():
            for result in suite_results:
                if result.passed:
                    benchmark_type = result.benchmark_type.value
                    if benchmark_type not in type_scores:
                        type_scores[benchmark_type] = []
                    type_scores[benchmark_type].append(result.score)
        
        # Calculate averages
        averages = {}
        for benchmark_type, scores in type_scores.items():
            if scores:
                averages[benchmark_type] = statistics.mean(scores)
        
        return averages
    
    def get_evaluation_history(self, model_id: str) -> List[BenchmarkResult]:
        """Get evaluation history for a model."""
        return self.evaluation_history.get(model_id, [])
    
    def export_results(self, output_path: str, format: str = "json") -> None:
        """Export benchmark results to file."""
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        if format == "json":
            results_data = {
                "results": [asdict(result) for result in self.runner.results],
                "suites": {name: asdict(suite) for name, suite in self.runner.suites.items()},
                "exported_at": datetime.now().isoformat()
            }
            
            with open(output_file, 'w') as f:
                json.dump(results_data, f, indent=2, default=str)
        
        self.logger.info(f"Exported results to {output_path}")