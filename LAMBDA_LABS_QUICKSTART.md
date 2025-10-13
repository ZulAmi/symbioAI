# 🚀 Quick Start Guide - Lambda Labs Setup

**Goal:** Get your benchmarks running on GPU in the next 30 minutes!

---

## ⏰ **RIGHT NOW (Next 30 Minutes):**

### **Step 1: Sign Up for Lambda Labs** (5 minutes)

1. Go to: https://lambdalabs.com
2. Click "Sign Up" (top right)
3. Use your email + password
4. Verify email
5. Add payment method (credit card)
6. You usually get $10-20 free credits! 🎁

---

### **Step 2: Launch GPU Instance** (5 minutes)

1. Click "Cloud" → "Instances" in menu
2. Click **"Launch Instance"** button

3. **Choose GPU:**

   - ✅ **RECOMMENDED:** 1x RTX 4090 ($0.50/hour)
   - Alternative: 1x A100 40GB ($1.10/hour) - 2x faster

4. **Choose Region:**

   - Pick closest: US-West (California) or US-East (Virginia)
   - Or whatever has availability

5. **Choose Filesystem:**

   - Select: "PyTorch" (pre-configured!)
   - This has everything: PyTorch, CUDA, Python

6. **SSH Key:**

   - If you have one: Upload it
   - If not: Click "Generate new key pair" and save it

7. Click **"Launch"** 🚀

**Wait 1-2 minutes for instance to start...**

---

### **Step 3: Connect to Your Instance** (5 minutes)

Lambda will show you the SSH command:

```bash
# They give you something like:
ssh ubuntu@<instance-ip>

# On your Mac terminal:
ssh ubuntu@123.45.67.89
```

**First time:** Type "yes" when asked about fingerprint

**You're in!** You should see a Linux terminal. ✅

---

### **Step 4: Upload Your Code** (10 minutes)

**On your Mac (in a new terminal):**

```bash
# Go to your project
cd /Users/zulhilmirahmat/Development/programming

# Create a tar file (compress your code)
tar -czf symbio_ai.tar.gz "Symbio AI"

# Upload to Lambda (replace <instance-ip> with your IP)
scp symbio_ai.tar.gz ubuntu@<instance-ip>:~/

# This might take 2-5 minutes depending on your internet
```

**Back on Lambda terminal:**

```bash
# Extract the code
tar -xzf symbio_ai.tar.gz
cd "Symbio AI"

# Install dependencies
pip install -r requirements.txt

# This takes 5-10 minutes
```

---

### **Step 5: Run Quick Test** (5 minutes)

**Make sure everything works:**

```bash
# Run our verified core test
python3 test_core_benchmark.py

# Should show:
# ✅✅✅ CORE BENCHMARK LOOP WORKS!
```

**If that works, you're ready!** 🎉

---

## 🚀 **START BENCHMARKS NOW!**

### **Option A: Quick Test (30 minutes) - Verify Everything**

```bash
# Run in tmux so it keeps going if you disconnect
tmux new -s benchmarks

# Run quick test
python3 run_benchmarks.py --mode test

# This takes 30 minutes
# You can detach: Press Ctrl+B then D
# Re-attach later: tmux attach -t benchmarks
```

### **Option B: Full Benchmarks (12-16 hours) - Go Straight to Publication**

```bash
# Run in tmux
tmux new -s benchmarks

# Run full benchmarks
python3 run_benchmarks.py --mode full

# Detach and let it run overnight
# Press Ctrl+B then D
```

**Cost:**

- Test mode: ~$0.25 (30 min)
- Full mode: ~$6-8 (12-16 hours)

---

## 📊 **Monitor Progress** (Optional)

**Check on it later:**

```bash
# SSH back in
ssh ubuntu@<instance-ip>

# Re-attach to tmux session
tmux attach -t benchmarks

# See what's happening!
```

**Or just let it run overnight and check tomorrow morning!** ☕

---

## 💾 **After Benchmarks Complete** (Tomorrow)

### **Download Results:**

```bash
# On your Mac terminal
scp -r ubuntu@<instance-ip>:~/Symbio\ AI/experiments/results ./benchmark_results/

# Download to your Mac
```

### **Stop the Instance (IMPORTANT!):**

1. Go back to Lambda Labs website
2. Click "Instances"
3. Click "Terminate" on your instance
4. Confirm

**Don't forget this or you'll keep paying!** 💰

---

## 🎯 **COMPLETE TIMELINE:**

```
TODAY (Saturday):
  Now:        Sign up Lambda Labs (5 min)
  Now + 5:    Launch instance (5 min)
  Now + 10:   Upload code (10 min)
  Now + 20:   Install deps (10 min)
  Now + 30:   Start benchmarks (2 min)
  Tonight:    GPU works while you sleep 😴

TOMORROW (Sunday):
  Morning:    Benchmarks done! ✅
  10am:       Download results (5 min)
  11am:       Run analysis (30 min)
  Afternoon:  Write results (3-4 hours)
  Evening:    Polish paper (2 hours)

MONDAY:
  Morning:    Submit to arXiv! 🎉
```

---

## 🆘 **TROUBLESHOOTING:**

### **Problem: Can't SSH**

```bash
# Make sure you saved the SSH key
# Try with explicit key:
ssh -i ~/.ssh/lambda_key ubuntu@<instance-ip>
```

### **Problem: Upload too slow**

```bash
# Alternative: Use git
# On Lambda:
git clone https://github.com/ZulAmi/symbioAI.git
cd symbioAI
```

### **Problem: Out of memory**

```bash
# Reduce batch size in run_benchmarks.py
# Change: batch_size: 128 → 64
```

### **Problem: Instance won't start**

- Try different region
- Try different GPU type
- Check Lambda status page

---

## ✅ **CHECKLIST:**

Before starting benchmarks:

- [ ] Lambda Labs account created
- [ ] GPU instance launched
- [ ] SSH connection working
- [ ] Code uploaded
- [ ] Dependencies installed (`pip install -r requirements.txt`)
- [ ] Quick test passed (`python3 test_core_benchmark.py`)
- [ ] Tmux session started
- [ ] Benchmarks running

**All checked? You're good to go!** 🚀

---

## 💰 **COST TRACKING:**

| Activity        | Time           | Cost (@$0.50/hr) |
| --------------- | -------------- | ---------------- |
| Quick test      | 30 min         | $0.25            |
| Setup/testing   | 1 hour         | $0.50            |
| Full benchmarks | 12-16 hrs      | $6-8             |
| **Total**       | **~14-18 hrs** | **~$7-9**        |

**Budget:** Plan for $10-15 to be safe

---

## 🎯 **NEXT STEPS AFTER THIS:**

1. ✅ Get Lambda Labs running (NOW - next 30 min)
2. ✅ Start benchmarks (TONIGHT)
3. ✅ Download results (TOMORROW morning)
4. ✅ Run analysis (TOMORROW afternoon)
5. ✅ Submit paper (MONDAY)

**Let's do this!** 🚀

---

**Ready to start? Let me know if you need help with any step!**
