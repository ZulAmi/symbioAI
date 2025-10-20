#!/usr/bin/env python3
"""
Tier 1 Comprehensive Benchmarking Suite
=======================================

This script runs comprehensive benchmarks for Tier 1 continual learning validation,
including all requested datasets and standard continual learning metrics.

Benchmarks include:
1. Standard Continual Learning Metrics (Average Accuracy, Forgetting Measure, etc.)
2. Baseline Comparisons (Fine-tuning, EWC, Multi-task Upper Bound)
3. Resource Scaling Analysis (Time, Memory, Parameters)
4. Task Similarity Analysis
5. Statistical Significance Testing

Datasets validated:
- TinyImageNet: 200 classes, bridges CIFAR and ImageNet
- SVHN: Real-world digits, domain shift testing
- Omniglot: Few-shot continual learning
- Fashion-MNIST: Harder MNIST variant
- EMNIST: Extended MNIST with letters
- ImageNet-Subset: Realistic continual learning

Usage:
    python validation/tier1_continual_learning/benchmarks.py --dataset fashion_mnist --quick
    python validation/tier1_continual_learning/benchmarks.py --comprehensive
    python validation/tier1_continual_learning/benchmarks.py --all-datasets
"""

import argparse
import sys
from pathlib import Path
import time
import json
from datetime import datetime
from typing import Dict, List, Any
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Add project to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from validation.tier1_continual_learning.validation import Tier1Validator, Tier1ValidationResult
from validation.tier1_continual_learning.datasets import load_tier1_dataset


class Tier1BenchmarkSuite:
    """
    Comprehensive benchmarking suite for Tier 1 continual learning validation.
    
    Runs standardized benchmarks across multiple datasets and methods,
    following established continual learning research protocols.
    """
    
    # Standard datasets for Tier 1 validation
    TIER1_DATASETS = {
        'fashion_mnist': {
            'name': 'Fashion-MNIST',
            'description': 'Clothing items, harder MNIST variant',
            'classes': 10,
            'quick_tasks': 2,
            'full_tasks': 5,
            'recommended_epochs': 3
        },
        'emnist': {
            'name': 'EMNIST',
            'description': 'Extended MNIST with letters',
            'classes': 47, 
            'quick_tasks': 3,
            'full_tasks': 8,
            'recommended_epochs': 3
        },
        'svhn': {
            'name': 'SVHN',
            'description': 'Street View House Numbers, real-world digits',
            'classes': 10,
            'quick_tasks': 2,
            'full_tasks': 5,
            'recommended_epochs': 5
        },
        'tiny_imagenet': {
            'name': 'TinyImageNet',
            'description': '200 classes, bridges CIFAR and ImageNet',
            'classes': 200,
            'quick_tasks': 5,
            'full_tasks': 10,
            'recommended_epochs': 5
        },
        'omniglot': {
            'name': 'Omniglot',
            'description': 'Few-shot continual learning, 1600+ classes',
            'classes': 1600,
            'quick_tasks': 10,
            'full_tasks': 20,
            'recommended_epochs': 2
        },
        'imagenet_subset': {
            'name': 'ImageNet-Subset',
            'description': 'Realistic continual learning under visual shift',
            'classes': 100,
            'quick_tasks': 5,
            'full_tasks': 10,
            'recommended_epochs': 8
        }
    }
    
    def __init__(self, results_dir: str = "validation/results/tier1_benchmarks"):
        """Initialize benchmark suite."""
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        self.validator = Tier1Validator()
        self.benchmark_results = {}
        
        print(f"🏆 Tier 1 Benchmark Suite initialized")
        print(f"📁 Results will be saved to: {self.results_dir}")
    
    def run_single_dataset_benchmark(
        self, 
        dataset_name: str,
        mode: str = 'quick',
        include_baselines: bool = True,
        task_types: List[str] = ['class_incremental']
    ) -> Dict[str, Tier1ValidationResult]:
        """
        Run comprehensive benchmark on a single dataset.
        
        Args:
            dataset_name: Name of dataset to benchmark
            mode: 'quick' or 'comprehensive'
            include_baselines: Whether to include baseline comparisons
            task_types: Types of continual learning to test
            
        Returns:
            Dictionary of results per task type
        """
        if dataset_name not in self.TIER1_DATASETS:
            raise ValueError(f"Unknown dataset: {dataset_name}. Available: {list(self.TIER1_DATASETS.keys())}")
        
        dataset_info = self.TIER1_DATASETS[dataset_name]
        
        print(f"\n{'='*80}")
        print(f"🏆 TIER 1 BENCHMARK: {dataset_info['name']}")
        print(f"{'='*80}")
        print(f"📊 {dataset_info['description']}")
        print(f"🔄 Mode: {mode.upper()}")
        print(f"🎯 Task types: {', '.join(task_types)}")
        
        results = {}
        
        for task_type in task_types:
            print(f"\n{'─'*60}")
            print(f"🧠 Running {task_type} continual learning...")
            print(f"{'─'*60}")
            
            # Determine parameters based on mode
            if mode == 'quick':
                num_tasks = dataset_info['quick_tasks']
                epochs_per_task = max(1, dataset_info['recommended_epochs'] // 2)
            else:  # comprehensive
                num_tasks = dataset_info['full_tasks'] 
                epochs_per_task = dataset_info['recommended_epochs']
            
            try:
                # Run validation
                result = self.validator.validate_continual_learning(
                    dataset_name=dataset_name,
                    num_tasks=num_tasks,
                    task_type=task_type,
                    epochs_per_task=epochs_per_task,
                    include_baselines=include_baselines
                )
                
                results[task_type] = result
                
                print(f"✅ {task_type} benchmark complete:")
                print(f"   Success Level: {result.success_level}")
                print(f"   Benchmark Score: {result.get_benchmark_score():.4f}")
                print(f"   Average Accuracy: {result.metrics.average_accuracy:.4f}")
                print(f"   Forgetting Measure: {result.metrics.forgetting_measure:.4f}")
                
            except Exception as e:
                print(f"❌ {task_type} benchmark failed: {e}")
                continue
        
        # Store results
        self.benchmark_results[dataset_name] = results
        
        # Generate dataset-specific report
        self._generate_dataset_report(dataset_name, results)
        
        return results
    
    def run_comprehensive_benchmark(
        self,
        datasets: List[str] = None,
        mode: str = 'comprehensive',
        include_statistical_analysis: bool = True
    ) -> Dict[str, Dict[str, Tier1ValidationResult]]:
        """
        Run comprehensive benchmark across multiple datasets.
        
        Args:
            datasets: List of datasets to benchmark (None = all)
            mode: 'quick' or 'comprehensive'
            include_statistical_analysis: Whether to run statistical analysis
            
        Returns:
            Dictionary of results per dataset and task type
        """
        if datasets is None:
            # Select datasets based on mode
            if mode == 'quick':
                datasets = ['fashion_mnist', 'svhn']  # Fast datasets
            else:
                datasets = list(self.TIER1_DATASETS.keys())
        
        print(f"\n{'='*80}")
        print(f"🏆 TIER 1 COMPREHENSIVE BENCHMARK SUITE")
        print(f"{'='*80}")
        print(f"📊 Datasets: {', '.join(datasets)}")
        print(f"🔄 Mode: {mode.upper()}")
        print(f"📈 Statistical analysis: {include_statistical_analysis}")
        
        start_time = time.time()
        all_results = {}
        
        # Run benchmarks on each dataset
        for i, dataset_name in enumerate(datasets, 1):
            print(f"\n{'='*80}")
            print(f"PROGRESS: Dataset {i}/{len(datasets)} - {dataset_name.upper()}")
            print(f"{'='*80}")
            
            try:
                dataset_results = self.run_single_dataset_benchmark(
                    dataset_name=dataset_name,
                    mode=mode,
                    include_baselines=True,
                    task_types=['class_incremental']
                )
                
                all_results[dataset_name] = dataset_results
                
            except Exception as e:
                print(f"❌ Dataset {dataset_name} failed: {e}")
                continue
        
        # Calculate total duration
        total_duration = (time.time() - start_time) / 60
        
        # Generate comprehensive report
        self._generate_comprehensive_report(all_results, mode, total_duration)
        
        # Run statistical analysis if requested
        if include_statistical_analysis and len(all_results) > 1:
            self._run_statistical_analysis(all_results)
        
        print(f"\n{'='*80}")
        print(f"🎉 COMPREHENSIVE BENCHMARK COMPLETE!")
        print(f"{'='*80}")
        print(f"⏱️  Total Duration: {total_duration:.2f} minutes")
        print(f"📊 Datasets Completed: {len(all_results)}/{len(datasets)}")
        print(f"📁 Reports saved to: {self.results_dir}")
        
        return all_results
    
    def compare_datasets(
        self,
        results: Dict[str, Dict[str, Tier1ValidationResult]] = None
    ) -> Dict[str, Any]:
        """
        Compare performance across different datasets.
        
        Args:
            results: Benchmark results (uses stored results if None)
            
        Returns:
            Comparison analysis
        """
        if results is None:
            results = self.benchmark_results
        
        if not results:
            print("❌ No benchmark results available for comparison")
            return {}
        
        print(f"\n{'='*60}")
        print(f"📊 CROSS-DATASET COMPARISON ANALYSIS")
        print(f"{'='*60}")
        
        comparison = {
            'dataset_rankings': {},
            'metric_analysis': {},
            'difficulty_assessment': {},
            'resource_requirements': {}
        }
        
        # Extract metrics for comparison
        datasets = []
        avg_accuracies = []
        forgetting_measures = []
        benchmark_scores = []
        training_times = []
        
        for dataset_name, dataset_results in results.items():
            if 'class_incremental' in dataset_results:
                result = dataset_results['class_incremental']
                
                datasets.append(dataset_name)
                avg_accuracies.append(result.metrics.average_accuracy)
                forgetting_measures.append(result.metrics.forgetting_measure)
                benchmark_scores.append(result.get_benchmark_score())
                
                if result.metrics.training_times:
                    training_times.append(np.sum(result.metrics.training_times))
                else:
                    training_times.append(0)
        
        if not datasets:
            print("❌ No valid results for comparison")
            return comparison
        
        # Rank datasets by different metrics
        accuracy_ranking = sorted(zip(datasets, avg_accuracies), key=lambda x: x[1], reverse=True)
        forgetting_ranking = sorted(zip(datasets, forgetting_measures), key=lambda x: x[1])  # Lower is better
        score_ranking = sorted(zip(datasets, benchmark_scores), key=lambda x: x[1], reverse=True)
        
        comparison['dataset_rankings'] = {
            'by_accuracy': accuracy_ranking,
            'by_forgetting_resistance': forgetting_ranking,
            'by_overall_score': score_ranking
        }
        
        # Print comparison
        print(f"\n🏆 Dataset Rankings:")
        print(f"\n📊 By Average Accuracy:")
        for i, (dataset, acc) in enumerate(accuracy_ranking, 1):
            print(f"   {i}. {self.TIER1_DATASETS[dataset]['name']}: {acc:.4f}")
        
        print(f"\n🧠 By Forgetting Resistance (lower forgetting is better):")
        for i, (dataset, forget) in enumerate(forgetting_ranking, 1):
            print(f"   {i}. {self.TIER1_DATASETS[dataset]['name']}: {forget:.4f}")
        
        print(f"\n🎯 By Overall Benchmark Score:")
        for i, (dataset, score) in enumerate(score_ranking, 1):
            print(f"   {i}. {self.TIER1_DATASETS[dataset]['name']}: {score:.4f}")
        
        # Difficulty assessment
        comparison['difficulty_assessment'] = {
            'easiest_dataset': accuracy_ranking[0][0],
            'hardest_dataset': accuracy_ranking[-1][0],
            'most_forgetting_prone': forgetting_ranking[-1][0],
            'least_forgetting_prone': forgetting_ranking[0][0]
        }
        
        print(f"\n📈 Difficulty Assessment:")
        print(f"   Easiest: {self.TIER1_DATASETS[comparison['difficulty_assessment']['easiest_dataset']]['name']}")
        print(f"   Hardest: {self.TIER1_DATASETS[comparison['difficulty_assessment']['hardest_dataset']]['name']}")
        
        return comparison
    
    def _generate_dataset_report(self, dataset_name: str, results: Dict[str, Tier1ValidationResult]):
        """Generate report for a single dataset."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_file = self.results_dir / f"dataset_report_{dataset_name}_{timestamp}.md"
        
        dataset_info = self.TIER1_DATASETS[dataset_name]
        
        doc = []
        doc.append(f"# Tier 1 Benchmark Report: {dataset_info['name']}")
        doc.append(f"\n**Dataset:** {dataset_name}")
        doc.append(f"**Description:** {dataset_info['description']}")
        doc.append(f"**Generated:** {datetime.now().isoformat()}")
        doc.append("\n---\n")
        
        for task_type, result in results.items():
            doc.append(f"## {task_type.replace('_', ' ').title()} Results\n")
            
            doc.append(f"**Success Level:** {result.success_level}")
            doc.append(f"**Benchmark Score:** {result.get_benchmark_score():.4f}")
            doc.append(f"**Duration:** {result.duration_minutes:.2f} minutes\n")
            
            doc.append("### Core Metrics")
            doc.append(f"- Average Accuracy: {result.metrics.average_accuracy:.4f}")
            doc.append(f"- Forgetting Measure: {result.metrics.forgetting_measure:.4f}")
            doc.append(f"- Forward Transfer: {result.metrics.forward_transfer:.4f}")
            doc.append(f"- Backward Transfer: {result.metrics.backward_transfer:.4f}")
            
            doc.append("\n### Task Performance")
            if result.metrics.final_accuracies:
                for i, acc in enumerate(result.metrics.final_accuracies):
                    doc.append(f"- Task {i+1}: {acc:.4f}")
            
            if result.baseline_comparison:
                doc.append("\n### Baseline Comparison")
                for method, scores in result.baseline_comparison.items():
                    doc.append(f"- {method}: Acc={scores['average_accuracy']:.4f}, Forget={scores['forgetting_measure']:.4f}")
            
            doc.append("")
        
        doc.append("---")
        doc.append(f"*Generated by Tier 1 Benchmark Suite*")
        
        with open(report_file, 'w') as f:
            f.write('\n'.join(doc))
        
        print(f"📄 Dataset report saved: {report_file}")
    
    def _generate_comprehensive_report(
        self, 
        all_results: Dict[str, Dict[str, Tier1ValidationResult]], 
        mode: str, 
        duration: float
    ):
        """Generate comprehensive benchmark report."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_file = self.results_dir / f"tier1_comprehensive_benchmark_{timestamp}.md"
        
        doc = []
        doc.append("# Tier 1 Comprehensive Benchmark Report")
        doc.append(f"\n**Generated:** {datetime.now().isoformat()}")
        doc.append(f"**Mode:** {mode}")
        doc.append(f"**Duration:** {duration:.2f} minutes")
        doc.append(f"**Datasets:** {len(all_results)}")
        doc.append("\n---\n")
        
        doc.append("## 🎯 Executive Summary\n")
        
        # Calculate summary statistics
        total_tests = sum(len(dataset_results) for dataset_results in all_results.values())
        excellent_count = 0
        good_count = 0
        needs_work_count = 0
        
        for dataset_results in all_results.values():
            for result in dataset_results.values():
                if result.success_level == "EXCELLENT":
                    excellent_count += 1
                elif result.success_level == "GOOD":
                    good_count += 1
                else:
                    needs_work_count += 1
        
        doc.append(f"- **Total Tests:** {total_tests}")
        doc.append(f"- **Success Levels:** {excellent_count} excellent, {good_count} good, {needs_work_count} need work")
        doc.append(f"- **Overall Success Rate:** {(excellent_count + good_count) / total_tests * 100:.1f}%")
        
        # Dataset-by-dataset results
        doc.append("\n## 📊 Dataset Results\n")
        
        for dataset_name, dataset_results in all_results.items():
            dataset_info = self.TIER1_DATASETS[dataset_name]
            doc.append(f"### {dataset_info['name']}\n")
            
            for task_type, result in dataset_results.items():
                doc.append(f"**{task_type.replace('_', ' ').title()}:**")
                doc.append(f"- Success Level: {result.success_level}")
                doc.append(f"- Benchmark Score: {result.get_benchmark_score():.4f}")
                doc.append(f"- Average Accuracy: {result.metrics.average_accuracy:.4f}")
                doc.append(f"- Forgetting Measure: {result.metrics.forgetting_measure:.4f}")
                doc.append("")
        
        # Comparative analysis
        comparison = self.compare_datasets(all_results)
        if comparison and 'dataset_rankings' in comparison:
            doc.append("## 🏆 Comparative Analysis\n")
            
            doc.append("### Top Performers by Accuracy")
            for i, (dataset, acc) in enumerate(comparison['dataset_rankings']['by_accuracy'][:3], 1):
                dataset_info = self.TIER1_DATASETS[dataset]
                doc.append(f"{i}. **{dataset_info['name']}**: {acc:.4f}")
            
            doc.append("\n### Best Forgetting Resistance")
            for i, (dataset, forget) in enumerate(comparison['dataset_rankings']['by_forgetting_resistance'][:3], 1):
                dataset_info = self.TIER1_DATASETS[dataset]
                doc.append(f"{i}. **{dataset_info['name']}**: {forget:.4f}")
        
        doc.append("\n## 📈 Recommendations\n")
        
        if excellent_count >= len(all_results) // 2:
            doc.append("✅ **Strong Performance**: Multiple datasets showing excellent results")
            doc.append("- Ready for academic publication")
            doc.append("- Suitable for funding proposals")
            doc.append("- Focus on scaling to more complex datasets")
        elif good_count + excellent_count >= len(all_results) // 2:
            doc.append("⚠️  **Moderate Performance**: Core capabilities working")
            doc.append("- Continue development and optimization")
            doc.append("- Focus on reducing forgetting")
            doc.append("- Improve forward transfer capabilities")
        else:
            doc.append("❌ **Needs Improvement**: Foundational work required")
            doc.append("- Focus on basic continual learning algorithms")
            doc.append("- Improve model architecture and training")
            doc.append("- Consider different regularization strategies")
        
        doc.append("\n---")
        doc.append("*Generated by Tier 1 Comprehensive Benchmark Suite*")
        
        with open(report_file, 'w') as f:
            f.write('\n'.join(doc))
        
        print(f"📄 Comprehensive report saved: {report_file}")
    
    def _run_statistical_analysis(self, results: Dict[str, Dict[str, Tier1ValidationResult]]):
        """Run statistical analysis on benchmark results."""
        print(f"\n📈 Running statistical analysis...")
        
        # TODO: Implement statistical significance testing
        # - Compare methods across datasets
        # - Confidence intervals
        # - Effect sizes
        
        print(f"⚠️  Statistical analysis not yet implemented")


def main():
    """Main benchmarking entry point."""
    parser = argparse.ArgumentParser(description='Tier 1 Continual Learning Benchmarks')
    
    parser.add_argument('--dataset', type=str, 
                       choices=list(Tier1BenchmarkSuite.TIER1_DATASETS.keys()),
                       help='Single dataset to benchmark')
    parser.add_argument('--all-datasets', action='store_true',
                       help='Benchmark all Tier 1 datasets')
    parser.add_argument('--quick', action='store_true',
                       help='Quick validation mode (faster, fewer tasks)')
    parser.add_argument('--comprehensive', action='store_true',
                       help='Comprehensive validation mode (slower, more thorough)')
    parser.add_argument('--no-baselines', action='store_true',
                       help='Skip baseline comparisons (faster)')
    
    args = parser.parse_args()
    
    # Determine mode
    if args.quick:
        mode = 'quick'
    elif args.comprehensive:
        mode = 'comprehensive'
    else:
        mode = 'quick'  # Default
    
    print(f"🏆 TIER 1 CONTINUAL LEARNING BENCHMARKS")
    print(f"🔄 Mode: {mode.upper()}")
    print(f"⚡ Include baselines: {not args.no_baselines}")
    
    # Create benchmark suite
    suite = Tier1BenchmarkSuite()
    
    if args.dataset:
        # Single dataset benchmark
        print(f"📊 Benchmarking single dataset: {args.dataset}")
        
        results = suite.run_single_dataset_benchmark(
            dataset_name=args.dataset,
            mode=mode,
            include_baselines=not args.no_baselines
        )
        
        print(f"\n✅ Single dataset benchmark complete!")
        
    elif args.all_datasets:
        # All datasets comprehensive benchmark
        print(f"📊 Benchmarking ALL Tier 1 datasets")
        
        results = suite.run_comprehensive_benchmark(
            datasets=None,  # All datasets
            mode=mode,
            include_statistical_analysis=True
        )
        
        print(f"\n✅ Comprehensive benchmark complete!")
        
    else:
        # Default: benchmark core datasets
        core_datasets = ['fashion_mnist', 'svhn'] if mode == 'quick' else ['fashion_mnist', 'emnist', 'svhn']
        
        print(f"📊 Benchmarking core datasets: {', '.join(core_datasets)}")
        
        results = suite.run_comprehensive_benchmark(
            datasets=core_datasets,
            mode=mode,
            include_statistical_analysis=True
        )
        
        print(f"\n✅ Core benchmark complete!")


if __name__ == '__main__':
    main()