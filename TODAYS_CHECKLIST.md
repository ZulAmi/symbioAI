# 🚀 TODAY'S ACTION CHECKLIST

**Goal:** Get benchmarks running on GPU by end of today  
**Time Required:** 30-60 minutes active work  
**Date:** October 12, 2025

---

## 🎯 **HAVING LAMBDA LABS ISSUES?**

**✅ BETTER OPTIONS AVAILABLE!**

Lambda Labs filesystem confusing? Try these instead:

1. **RunPod** (easiest, cheaper) → See: `RUNPOD_QUICKSTART.md` ⭐
2. **Vast.ai** (cheapest, 50% off) → See: `VAST_AI_QUICKSTART.md` 🏆
3. **Comparison guide** → See: `GPU_PROVIDER_COMPARISON.md`

**Or continue with Lambda Labs below:**

---

## ✅ **PHASE 1: LAMBDA LABS SETUP** (Next 30 minutes)

### **Task 1: Create Account** ⏱️ 5 minutes

- [ ] Go to https://lambdalabs.com
- [ ] Click "Sign Up"
- [ ] Enter email and password
- [ ] Verify email
- [ ] Add payment method
- [ ] Check for free credits (usually $10-20!)

**Status:** ******\_\_\_******  
**Notes:** ******\_\_\_******

---

### **Task 2: Launch GPU Instance** ⏱️ 5 minutes

- [ ] Click "Cloud" → "Instances"
- [ ] Click "Launch Instance"
- [ ] **GPU: Select one of these (in order of preference):**
  - [ ] 1x RTX 4090 ($0.50/hour) - Best if available
  - [ ] **1x A10 24GB ($0.60-0.80/hour)** - ⭐ Great alternative!
  - [ ] 1x A100 40GB ($1.10/hour) - Faster but pricier
  - [ ] 1x RTX A6000 ($0.80/hour) - Also good
- [ ] Region: Pick closest (US-West or US-East)
- [ ] **Filesystem/Base Image: Select one of these:**
  - [ ] **"Lambda Stack"** - ⭐ BEST (PyTorch + everything pre-installed)
  - [ ] **"PyTorch"** - ✅ Also good (PyTorch + CUDA)
  - [ ] ❌ Avoid: GPU Base, Ubuntu bare, TensorFlow
- [ ] SSH Key: Upload or generate new
- [ ] Click "Launch"
- [ ] Wait 1-2 minutes for instance to start
- [ ] **WRITE DOWN YOUR INSTANCE IP:** ******\_\_\_******
- [ ] **WRITE DOWN YOUR GPU TYPE:** ******\_\_\_******

**Status:** ******\_\_\_******  
**Instance IP:** ******\_\_\_******

---

### **Task 3: Test SSH Connection** ⏱️ 2 minutes

```bash
# In your Mac terminal:
ssh ubuntu@<YOUR-INSTANCE-IP>
```

- [ ] SSH connection successful
- [ ] You see Linux terminal prompt

**Status:** ******\_\_\_******

---

## ✅ **PHASE 2: UPLOAD CODE** (Next 15 minutes)

### **Task 4: Package Your Code** ⏱️ 3 minutes

```bash
# In NEW Mac terminal (keep SSH open):
cd /Users/zulhilmirahmat/Development/programming

# Create compressed file
tar -czf symbio_ai.tar.gz "Symbio AI"
```

- [ ] Tar file created successfully
- [ ] File size looks reasonable (< 100MB)

**Status:** ******\_\_\_******

---

### **Task 5: Upload to Lambda** ⏱️ 5 minutes

```bash
# Upload (replace <YOUR-IP> with your instance IP)
scp symbio_ai.tar.gz ubuntu@<YOUR-IP>:~/
```

- [ ] Upload started
- [ ] Upload completed (might take 2-5 minutes)

**Upload time:** ******\_\_\_******  
**Status:** ******\_\_\_******

---

### **Task 6: Extract on Lambda** ⏱️ 1 minute

```bash
# Back in SSH terminal:
tar -xzf symbio_ai.tar.gz
cd "Symbio AI"
ls -la
```

- [ ] Files extracted successfully
- [ ] Can see your code files

**Status:** ******\_\_\_******

---

## ✅ **PHASE 3: INSTALL & TEST** (Next 10 minutes)

### **Task 7: Install Dependencies** ⏱️ 5-10 minutes

```bash
# On Lambda terminal:
pip install -r requirements.txt
```

- [ ] Installation started
- [ ] No errors during installation
- [ ] Installation completed

**Time taken:** ******\_\_\_******  
**Status:** ******\_\_\_******

---

### **Task 8: Run Core Test** ⏱️ 2 minutes

```bash
# Verify everything works:
python3 test_core_benchmark.py
```

**Expected output:**

```
✅✅✅ CORE BENCHMARK LOOP WORKS!
✅ PyTorch training/eval works correctly
```

- [ ] Test completed successfully
- [ ] Saw "CORE BENCHMARK LOOP WORKS"
- [ ] No errors

**Status:** ******\_\_\_******

---

## ✅ **PHASE 4: START BENCHMARKS** (Next 5 minutes)

### **Task 9: Decide Which Mode** ⏱️ 1 minute

Choose one:

- [ ] **TEST MODE** (30 min, $0.25) - Just verify it works
- [ ] **FAST MODE** (4-6 hrs, $2-3) - Quick results
- [ ] **FULL MODE** (12-16 hrs, $6-8) - ⭐ RECOMMENDED for publication
- [ ] **PUBLICATION MODE** (24-48 hrs, $12-24) - Maximum quality

**My choice:** ******\_\_\_******

---

### **Task 10: Start Benchmarks in Tmux** ⏱️ 3 minutes

```bash
# Create tmux session
tmux new -s benchmarks

# For TEST mode (verify it works):
python3 run_benchmarks.py --mode test

# For FULL mode (recommended):
python3 run_benchmarks.py --mode full

# For PUBLICATION mode (best quality):
python3 run_benchmarks.py --mode publication
```

- [ ] Tmux session created
- [ ] Benchmark script started
- [ ] Seeing progress output
- [ ] Detached from tmux (Ctrl+B then D)

**Mode started:** ******\_\_\_******  
**Start time:** ******\_\_\_******  
**Status:** ******\_\_\_******

---

## ✅ **PHASE 5: VERIFY & RELAX** (Next 5 minutes)

### **Task 11: Final Checks** ⏱️ 2 minutes

```bash
# Check it's still running:
tmux attach -t benchmarks

# See activity? Good!
# Detach again: Ctrl+B then D

# Check GPU usage:
nvidia-smi

# Should show GPU in use
```

- [ ] Benchmarks still running
- [ ] GPU being utilized
- [ ] No errors visible

**GPU Utilization:** ******\_\_\_******  
**Status:** ******\_\_\_******

---

### **Task 12: Set Completion Reminder** ⏱️ 1 minute

Based on your mode:

- TEST: Check back in 30-60 minutes
- FAST: Check tomorrow morning
- FULL: Check tomorrow morning
- PUBLICATION: Check in 2 days

**Expected completion:** ******\_\_\_******

- [ ] Set phone/calendar reminder
- [ ] Noted when to check back

---

## 🎉 **YOU'RE DONE FOR TODAY!**

### **What happens now:**

1. ✅ GPU is running your benchmarks
2. ⏰ It will run for hours (automatically)
3. 💾 Results will be saved to `experiments/results/`
4. 💤 You can close your laptop and relax!

### **Tomorrow you'll:**

1. Download results
2. Run analysis
3. Write paper
4. Submit to arXiv!

---

## 📊 **COST TRACKING:**

| Item                | Time       | Cost        |
| ------------------- | ---------- | ----------- |
| Setup/testing       | **\_** hrs | $**\_**     |
| Benchmark mode      | **\_**     | **\_**      |
| **Estimated total** | **\_** hrs | **$**\_**** |

---

## 🆘 **IF SOMETHING GOES WRONG:**

### **Can't SSH:**

```bash
# Try with explicit key:
ssh -i ~/.ssh/<your-key> ubuntu@<YOUR-IP>
```

### **Upload too slow:**

```bash
# Alternative - use git:
# On Lambda:
git clone https://github.com/ZulAmi/symbioAI.git
cd symbioAI
```

### **Benchmarks crash:**

```bash
# Check what happened:
tmux attach -t benchmarks

# Look for error messages
# If it's adapter-related, that's expected (we skip those)
```

### **Need help:**

- Check LAMBDA_LABS_QUICKSTART.md
- Check PREFLIGHT_VERIFICATION_COMPLETE.md
- Ask me for help!

---

## ✅ **FINAL CHECKLIST:**

**Before you close your laptop:**

- [ ] Lambda Labs instance running
- [ ] Code uploaded and extracted
- [ ] Dependencies installed
- [ ] Core test passed
- [ ] Benchmarks started in tmux
- [ ] Confirmed GPU in use
- [ ] Set reminder for tomorrow
- [ ] Noted instance IP for tomorrow

**Instance IP:** ******\_\_\_******  
**Start time:** ******\_\_\_******  
**Expected done:** ******\_\_\_******  
**Estimated cost:** $**\_**

---

## 🎯 **TOMORROW'S PREVIEW:**

When benchmarks complete, you'll:

1. **Download results** (5 min)

   ```bash
   scp -r ubuntu@<YOUR-IP>:~/Symbio\ AI/experiments/results ./
   ```

2. **Run analysis** (30 min)

   ```bash
   python3 experiments/analysis/results_analyzer.py
   ```

3. **Update paper** (3-4 hours)

   - Insert results
   - Add figures
   - Write analysis

4. **Submit to arXiv** (1 hour)
   - Format LaTeX
   - Upload

**Then you're published!** 🎓

---

## 📝 **NOTES:**

Use this space for any notes, issues, or observations:

---

---

---

---

---

**You've got this!** 🚀  
**See you tomorrow with results!** 📊
