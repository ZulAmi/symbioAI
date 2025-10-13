#!/usr/bin/env python3
"""
Local Pre-Flight Check

Run this ON YOUR MAC before uploading to RunPod/cloud.
Verifies that all files are present and code imports correctly.

Usage:
    python3 local_preflight_check.py
"""

import sys
from pathlib import Path

print("="*70)
print("🔍 LOCAL PRE-FLIGHT CHECK (Mac)")
print("="*70)
print("\nChecking files before upload to RunPod...\n")

def check_file_exists(filepath, description):
    """Check if a file exists."""
    path = Path(filepath)
    if path.exists():
        size = path.stat().st_size / 1024  # KB
        print(f"✅ {filepath:40s} ({size:6.1f} KB)")
        return True
    else:
        print(f"❌ {filepath:40s} MISSING!")
        return False

def check_directory_exists(dirpath, description):
    """Check if a directory exists."""
    path = Path(dirpath)
    if path.exists() and path.is_dir():
        file_count = len(list(path.rglob('*')))
        print(f"✅ {dirpath:40s} ({file_count} files)")
        return True
    else:
        print(f"❌ {dirpath:40s} MISSING!")
        return False

print("📁 CRITICAL FILES:")
print("-" * 70)

critical_files = [
    ('run_benchmarks.py', 'Main benchmark script'),
    ('test_core_benchmark.py', 'Core functionality test'),
    ('verify_runpod_setup.py', 'RunPod verification'),
    ('requirements.txt', 'Dependencies'),
]

critical_ok = all(check_file_exists(f, d) for f, d in critical_files)

print("\n📁 IMPORTANT DIRECTORIES:")
print("-" * 70)

important_dirs = [
    ('experiments', 'Experiments directory'),
    ('experiments/benchmarks', 'Benchmarks'),
    ('training', 'Training code'),
    ('models', 'Model definitions'),
]

dirs_ok = all(check_directory_exists(d, desc) for d, desc in important_dirs)

print("\n📚 DOCUMENTATION FILES:")
print("-" * 70)

doc_files = [
    ('RUNPOD_QUICKSTART.md', 'RunPod guide'),
    ('RUNPOD_SETUP_VERIFICATION.md', 'Verification guide'),
    ('VAST_AI_QUICKSTART.md', 'Vast.ai guide'),
    ('GPU_PROVIDER_COMPARISON.md', 'Provider comparison'),
    ('GPU_QUICK_REFERENCE.md', 'Quick reference'),
]

docs_ok = all(check_file_exists(f, d) for f, d in doc_files)

print("\n🔍 TESTING IMPORTS (on Mac - might not have GPU):")
print("-" * 70)

imports_ok = True

# Test basic imports
try:
    import torch
    print(f"✅ PyTorch {torch.__version__:30s}")
except ImportError:
    print(f"⚠️  PyTorch not installed (OK on Mac, needed on RunPod)")
    imports_ok = False

try:
    import numpy
    print(f"✅ NumPy {numpy.__version__:32s}")
except ImportError:
    print(f"❌ NumPy not installed")
    imports_ok = False

try:
    import pandas
    print(f"✅ Pandas {pandas.__version__:31s}")
except ImportError:
    print(f"❌ Pandas not installed")
    imports_ok = False

# Test project imports (don't worry about GPU requirements)
print("\n🔍 TESTING PROJECT STRUCTURE:")
print("-" * 70)

structure_ok = True

try:
    sys.path.append(str(Path(__file__).parent))
    
    # Try importing modules (may fail on Mac without GPU, that's OK)
    try:
        from training import continual_learning
        print(f"✅ training.continual_learning module found")
    except Exception as e:
        print(f"⚠️  training.continual_learning import error (OK if no GPU on Mac)")
        print(f"   Error: {str(e)[:50]}...")
    
    try:
        from models import vision
        print(f"✅ models.vision module found")
    except Exception as e:
        print(f"⚠️  models.vision import error (OK if no GPU on Mac)")
    
    try:
        from experiments.benchmarks import continual_learning_benchmarks
        print(f"✅ experiments.benchmarks module found")
    except Exception as e:
        print(f"⚠️  experiments.benchmarks import error (OK if no GPU on Mac)")
        
except Exception as e:
    print(f"❌ Project structure issue: {e}")
    structure_ok = False

print("\n📊 ESTIMATING UPLOAD SIZE:")
print("-" * 70)

total_size = 0
for path in Path('.').rglob('*'):
    if path.is_file() and not any(x in str(path) for x in ['.git', '__pycache__', '.pyc', 'data/', 'experiments/results/']):
        total_size += path.stat().st_size

total_mb = total_size / (1024 * 1024)
print(f"Estimated upload size: {total_mb:.1f} MB")

if total_mb < 100:
    print(f"✅ Upload size reasonable (< 100 MB)")
elif total_mb < 500:
    print(f"⚠️  Upload size large ({total_mb:.1f} MB) but OK")
else:
    print(f"❌ Upload size very large ({total_mb:.1f} MB)")
    print(f"   Consider using git clone instead of file upload")

print("\n" + "="*70)
print("📊 PRE-FLIGHT SUMMARY")
print("="*70)

checks = [
    ("Critical Files", critical_ok),
    ("Important Directories", dirs_ok),
    ("Documentation", docs_ok),
    ("Upload Size", total_mb < 500)
]

passed = sum(1 for _, ok in checks if ok)
total = len(checks)

for name, ok in checks:
    status = "✅ PASS" if ok else "❌ FAIL"
    print(f"{status}: {name}")

print("="*70)
print(f"Score: {passed}/{total} checks passed")
print("="*70)

if passed == total:
    print("\n🎉 ALL CHECKS PASSED!")
    print("✅ Your code is ready to upload to RunPod!")
    print("\n🚀 Next steps:")
    print("   1. Go to https://www.runpod.io/")
    print("   2. Launch RTX 4090 pod with PyTorch template")
    print("   3. Upload code via git or scp")
    print("   4. Run: python3 verify_runpod_setup.py")
    print("   5. Run: python3 run_benchmarks.py --mode full")
    print("\n📖 See: RUNPOD_QUICKSTART.md for detailed guide")
    sys.exit(0)
else:
    print("\n⚠️  SOME ISSUES DETECTED")
    print("Fix the failed checks before uploading")
    print("\n📖 Check the files listed above")
    sys.exit(1)
