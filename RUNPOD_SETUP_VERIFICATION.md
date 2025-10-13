# ✅ RunPod Setup Verification Guide

**Run this checklist BEFORE starting benchmarks on RunPod!**

---

## 🚀 **QUICK VERIFICATION (1 minute)**

```bash
# Run automatic verification script:
python3 verify_runpod_setup.py
```

**Expected output:**

```
🎉 ALL CHECKS PASSED!
✅ Your RunPod environment is ready for benchmarks!
```

If you see this, **skip to step 5** and start benchmarks!

---

## 🔧 **MANUAL VERIFICATION** (if auto-check fails)

### **Step 1: Check Python** ⏱️ 10 seconds

```bash
python3 --version
```

**Expected:** Python 3.8 or higher  
**✅ Good:** `Python 3.10.12` or similar  
**❌ Bad:** `Python 2.7` or `Python 3.6`

**Fix if needed:**

```bash
# RunPod usually has Python 3.10+ pre-installed
# If not, contact RunPod support
```

---

### **Step 2: Check GPU** ⏱️ 10 seconds

```bash
nvidia-smi
```

**Expected output:**

```
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 525.xx       Driver Version: 525.xx       CUDA Version: 12.x   |
|-------------------------------+----------------------+----------------------+
|   0  NVIDIA RTX 4090    On   | 00000000:01:00.0 Off |                  N/A |
```

**✅ Good signs:**

- You see GPU name (RTX 4090, A10, A100, etc.)
- GPU memory shown (24GB, 40GB, etc.)
- CUDA version shown

**❌ Bad signs:**

- "command not found"
- No GPU listed
- Error messages

**Fix if needed:**

```bash
# This should not happen on RunPod
# If it does, your pod didn't start correctly
# Stop and restart the pod
```

---

### **Step 3: Check PyTorch + GPU** ⏱️ 20 seconds

```bash
python3 -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"
```

**Expected output:**

```
PyTorch: 2.1.2
CUDA available: True
GPU: NVIDIA GeForce RTX 4090
```

**✅ Good:** All three lines show correct info  
**❌ Bad:** `CUDA available: False`

**Fix if needed:**

```bash
# If PyTorch not installed:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# If CUDA not available, restart pod
```

---

### **Step 4: Check Dependencies** ⏱️ 30 seconds

```bash
python3 -c "import numpy, pandas, matplotlib, sklearn; print('✅ All core packages installed')"
```

**Expected output:**

```
✅ All core packages installed
```

**❌ If you see import errors:**

```bash
pip install -r requirements.txt
```

---

### **Step 5: Test GPU Computation** ⏱️ 10 seconds

```bash
python3 -c "import torch; x=torch.randn(1000,1000).cuda(); y=torch.randn(1000,1000).cuda(); z=torch.mm(x,y); print('✅ GPU computation works!')"
```

**Expected output:**

```
✅ GPU computation works!
```

**✅ This confirms GPU is working!**

---

### **Step 6: Check Disk Space** ⏱️ 5 seconds

```bash
df -h /workspace
```

**Expected output:**

```
Filesystem      Size  Used Avail Use% Mounted on
...            50G   5.0G   45G  10% /workspace
```

**✅ Good:** At least 10GB free  
**⚠️ Warning:** Less than 10GB free (might run out during benchmarks)

---

### **Step 7: Check Files** ⏱️ 10 seconds

```bash
ls -la | grep -E "(run_benchmarks|test_core|requirements)"
```

**Expected output:**

```
-rw-r--r--  1 root  run_benchmarks.py
-rw-r--r--  1 root  test_core_benchmark.py
-rw-r--r--  1 root  requirements.txt
```

**✅ Good:** All three files present  
**❌ Bad:** Files missing (you're in wrong directory)

**Fix if needed:**

```bash
# Navigate to correct directory
cd /workspace/symbioAI
# or
cd ~/symbioAI
```

---

## 🎯 **FINAL VERIFICATION TEST**

### **Run Core Benchmark Test** ⏱️ 1 minute

```bash
python3 test_core_benchmark.py
```

**Expected output:**

```
✅✅✅ CORE BENCHMARK LOOP WORKS!
✅ PyTorch training/eval works correctly
```

**If you see this = READY TO GO!** ✅

---

## ✅ **VERIFICATION CHECKLIST**

**Before starting full benchmarks, confirm:**

- [x] Python 3.8+ installed
- [x] GPU detected by nvidia-smi
- [x] PyTorch installed and sees GPU
- [x] CUDA available in PyTorch
- [x] Core packages installed
- [x] GPU computation works
- [x] At least 10GB disk space
- [x] Project files present
- [x] test_core_benchmark.py passes

**All checked? You're ready!** 🚀

---

## 🚀 **START BENCHMARKS**

```bash
# Create screen session (so it runs if you disconnect)
screen -S benchmarks

# Start benchmarks
python3 run_benchmarks.py --mode full

# Detach from screen: Ctrl+A, then D
```

**Monitor progress:**

```bash
# Reattach to screen
screen -r benchmarks

# Check GPU usage
nvidia-smi
```

---

## 🆘 **TROUBLESHOOTING**

### **Issue: "No module named 'torch'"**

```bash
pip install torch torchvision
```

### **Issue: "CUDA not available"**

```bash
# Restart your RunPod instance
# Or select PyTorch template when launching
```

### **Issue: "Permission denied"**

```bash
chmod +x run_benchmarks.py
```

### **Issue: "Out of disk space"**

```bash
# Clean up
rm -rf ~/.cache/pip
rm -rf data/*.tar.gz

# Or increase container disk when launching pod
```

### **Issue: "GPU out of memory"**

```bash
# Reduce batch size in run_benchmarks.py
# Or use smaller model
```

---

## 📊 **EXPECTED RESOURCE USAGE**

For full benchmarks (16 hours):

| Resource       | Usage     | Your GPU   |
| -------------- | --------- | ---------- |
| **GPU Memory** | 6-8 GB    | 24 GB ✅   |
| **Disk Space** | 5-10 GB   | 50 GB ✅   |
| **RAM**        | 8-16 GB   | 32 GB ✅   |
| **CPU**        | 4-8 cores | 8 cores ✅ |

**Your RunPod pod is MORE than enough!** 💪

---

## ✅ **QUICK REFERENCE**

```bash
# Verify setup
python3 verify_runpod_setup.py

# Test core functionality
python3 test_core_benchmark.py

# Start benchmarks
screen -S benchmarks
python3 run_benchmarks.py --mode full

# Detach: Ctrl+A, D

# Check status later
screen -r benchmarks
nvidia-smi

# Stop GPU when done
# (In RunPod dashboard: click "Stop")
```

---

**All set? Let's go!** 🚀
