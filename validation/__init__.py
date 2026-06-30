"""
Real Validation Module for SymbioAI
====================================

This package provides actual validation infrastructure for real experiments.

Note: We avoid importing heavy modules at package-import time to prevent
side-effects (e.g., printing warnings) when `python -m validation.something`
implicitly imports this package. We use lazy attribute loading (PEP 562).
"""

__all__ = [
    'RealValidationFramework',
    'ValidationResult',
    'CompetitiveComparison',
    'BenchmarkReport'
]

__version__ = '1.0.0'

def __getattr__(name):
    if name in __all__:
        from . import real_validation_framework as _rvf
        return getattr(_rvf, name)
    raise AttributeError(f"module 'validation' has no attribute {name!r}")
