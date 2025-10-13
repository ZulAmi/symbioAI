#!/usr/bin/env python3
"""
RunPod Environment Verification Script

Checks that your RunPod instance has everything needed for benchmarks.
Run this FIRST before starting benchmarks!

Usage:
    python3 verify_runpod_setup.py
"""

import sys
import subprocess
from pathlib import Path

print("="*70)
print("🔍 RUNPOD ENVIRONMENT VERIFICATION")
print("="*70)

def check_python():
    """Check Python version."""
    print("\n1️⃣ Checking Python version...")
    version = sys.version_info
    print(f"   Python {version.major}.{version.minor}.{version.micro}")
    
    if version.major == 3 and version.minor >= 8:
        print("   ✅ Python version OK (3.8+)")
        return True
    else:
        print(f"   ⚠️  Python {version.major}.{version.minor} detected, recommend 3.8+")
        return False

def check_gpu():
    """Check GPU availability."""
    print("\n2️⃣ Checking GPU...")
    
    try:
        import torch
        
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            
            print(f"   ✅ GPU detected: {gpu_name}")
            print(f"   ✅ GPU count: {gpu_count}")
            print(f"   ✅ GPU memory: {gpu_memory:.1f} GB")
            
            # Test GPU computation
            x = torch.randn(100, 100).cuda()
            y = torch.randn(100, 100).cuda()
            z = torch.mm(x, y)
            
            print(f"   ✅ GPU computation test: PASSED")
            return True
        else:
            print("   ❌ No GPU detected!")
            print("   ⚠️  Benchmarks will be VERY slow on CPU")
            return False
            
    except ImportError:
        print("   ❌ PyTorch not installed!")
        return False
    except Exception as e:
        print(f"   ❌ GPU test failed: {e}")
        return False

def check_packages():
    """Check required packages."""
    print("\n3️⃣ Checking required packages...")
    
    required = [
        'torch',
        'torchvision', 
        'numpy',
        'pandas',
        'matplotlib',
        'tqdm',
        'scikit-learn'
    ]
    
    missing = []
    
    for package in required:
        try:
            __import__(package)
            print(f"   ✅ {package}")
        except ImportError:
            print(f"   ❌ {package} - MISSING!")
            missing.append(package)
    
    if missing:
        print(f"\n   ⚠️  Missing packages: {', '.join(missing)}")
        print(f"   Run: pip install {' '.join(missing)}")
        return False
    else:
        print(f"   ✅ All required packages installed")
        return True

def check_disk_space():
    """Check available disk space."""
    print("\n4️⃣ Checking disk space...")
    
    try:
        import shutil
        
        total, used, free = shutil.disk_usage("/")
        
        free_gb = free / (1024**3)
        total_gb = total / (1024**3)
        
        print(f"   Total: {total_gb:.1f} GB")
        print(f"   Free: {free_gb:.1f} GB")
        
        if free_gb > 10:
            print(f"   ✅ Disk space OK ({free_gb:.1f} GB free)")
            return True
        else:
            print(f"   ⚠️  Low disk space ({free_gb:.1f} GB free)")
            print(f"   Recommend at least 10 GB free")
            return False
            
    except Exception as e:
        print(f"   ⚠️  Could not check disk space: {e}")
        return True

def check_workspace():
    """Check workspace structure."""
    print("\n5️⃣ Checking workspace structure...")
    
    required_dirs = [
        'experiments',
        'experiments/benchmarks',
        'training',
        'models'
    ]
    
    required_files = [
        'run_benchmarks.py',
        'test_core_benchmark.py',
        'requirements.txt'
    ]
    
    all_ok = True
    
    for dir_path in required_dirs:
        if Path(dir_path).exists():
            print(f"   ✅ {dir_path}/")
        else:
            print(f"   ⚠️  {dir_path}/ - missing (will be created)")
            Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    for file_path in required_files:
        if Path(file_path).exists():
            print(f"   ✅ {file_path}")
        else:
            print(f"   ❌ {file_path} - MISSING!")
            all_ok = False
    
    return all_ok

def check_runpod_specific():
    """Check RunPod-specific features."""
    print("\n6️⃣ Checking RunPod-specific features...")
    
    # Check if we're running on RunPod
    workspace = Path('/workspace')
    
    if workspace.exists():
        print(f"   ✅ RunPod workspace detected: /workspace")
        print(f"   ✅ You're running on RunPod!")
    else:
        print(f"   ℹ️  Not on RunPod (running on Lambda/Vast.ai/local)")
        print(f"   ℹ️  This is fine - code works everywhere!")
    
    # Check screen/tmux availability
    try:
        subprocess.run(['which', 'screen'], capture_output=True, check=True)
        print(f"   ✅ screen available (for background jobs)")
    except:
        try:
            subprocess.run(['which', 'tmux'], capture_output=True, check=True)
            print(f"   ✅ tmux available (for background jobs)")
        except:
            print(f"   ⚠️  Neither screen nor tmux found")
            print(f"   Install with: apt-get install screen")
    
    return True

def check_nvidia_smi():
    """Check nvidia-smi."""
    print("\n7️⃣ Checking nvidia-smi...")
    
    try:
        result = subprocess.run(
            ['nvidia-smi'],
            capture_output=True,
            text=True,
            timeout=5
        )
        
        if result.returncode == 0:
            print("   ✅ nvidia-smi working")
            
            # Parse GPU info
            lines = result.stdout.split('\n')
            for line in lines:
                if 'NVIDIA' in line or 'RTX' in line or 'Tesla' in line or 'A100' in line or 'A10' in line:
                    print(f"   ℹ️  {line.strip()}")
            
            return True
        else:
            print("   ⚠️  nvidia-smi failed")
            return False
            
    except Exception as e:
        print(f"   ⚠️  nvidia-smi not available: {e}")
        return False

def run_quick_test():
    """Run quick functionality test."""
    print("\n8️⃣ Running quick functionality test...")
    
    try:
        print("   Testing PyTorch + GPU...")
        import torch
        
        # Create tensors on GPU
        x = torch.randn(1000, 1000, device='cuda')
        y = torch.randn(1000, 1000, device='cuda')
        
        # Matrix multiplication
        z = torch.mm(x, y)
        
        # Check result
        assert z.shape == (1000, 1000)
        assert z.device.type == 'cuda'
        
        print("   ✅ GPU computation works!")
        
        # Test neural network
        print("   Testing neural network...")
        model = torch.nn.Sequential(
            torch.nn.Linear(100, 50),
            torch.nn.ReLU(),
            torch.nn.Linear(50, 10)
        ).cuda()
        
        x = torch.randn(32, 100, device='cuda')
        y = model(x)
        
        assert y.shape == (32, 10)
        
        print("   ✅ Neural network works!")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Quick test failed: {e}")
        return False

def main():
    """Run all checks."""
    
    checks = [
        ("Python Version", check_python),
        ("GPU Availability", check_gpu),
        ("Required Packages", check_packages),
        ("Disk Space", check_disk_space),
        ("Workspace Structure", check_workspace),
        ("RunPod Features", check_runpod_specific),
        ("NVIDIA Tools", check_nvidia_smi),
        ("Quick Functionality", run_quick_test)
    ]
    
    results = []
    
    for name, check_fn in checks:
        try:
            result = check_fn()
            results.append((name, result))
        except Exception as e:
            print(f"   ❌ Error running check: {e}")
            results.append((name, False))
    
    # Summary
    print("\n" + "="*70)
    print("📊 VERIFICATION SUMMARY")
    print("="*70)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}: {name}")
    
    print("="*70)
    print(f"Score: {passed}/{total} checks passed")
    print("="*70)
    
    if passed == total:
        print("\n🎉 ALL CHECKS PASSED!")
        print("✅ Your RunPod environment is ready for benchmarks!")
        print("\n🚀 Next step:")
        print("   python3 run_benchmarks.py --mode test")
        return 0
    elif passed >= total - 2:
        print("\n⚠️  MOSTLY READY - minor issues detected")
        print("✅ You can proceed, but fix warnings if possible")
        print("\n🚀 Next step:")
        print("   python3 run_benchmarks.py --mode test")
        return 0
    else:
        print("\n❌ SETUP INCOMPLETE")
        print("⚠️  Fix the failed checks before running benchmarks")
        print("\n🔧 Common fixes:")
        print("   pip install -r requirements.txt")
        print("   apt-get install screen")
        return 1

if __name__ == '__main__':
    sys.exit(main())
