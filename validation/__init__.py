"""
Real Validation Module for SymbioAI
====================================

This module provides actual validation infrastructure for real experiments.

Unlike simulation-based testing frameworks, this module:
- Runs actual training on real datasets
- Measures real performance metrics
- Compares with real baseline methods
- Produces verifiable experimental results
"""

from .real_validation_framework import (
    RealValidationFramework,
    ValidationResult,
    CompetitiveComparison,
    BenchmarkReport
)

__all__ = [
    'RealValidationFramework',
    'ValidationResult',
    'CompetitiveComparison',
    'BenchmarkReport'
]

__version__ = '1.0.0'
