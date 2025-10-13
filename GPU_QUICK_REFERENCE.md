# 📋 GPU PROVIDER QUICK REFERENCE CARD

**Print this or keep it open while you work!**

---

## 🎯 **THE CHOICE:**

```
┌─────────────────────────────────────────┐
│  EASIEST: RunPod ($6.44)               │
│  → https://www.runpod.io/              │
│  → RUNPOD_QUICKSTART.md                │
│  → 10 minutes setup                    │
└─────────────────────────────────────────┘

┌─────────────────────────────────────────┐
│  CHEAPEST: Vast.ai ($4.13)             │
│  → https://vast.ai/                    │
│  → VAST_AI_QUICKSTART.md               │
│  → 15 minutes setup                    │
└─────────────────────────────────────────┘

┌─────────────────────────────────────────┐
│  LAMBDA: Lambda Labs ($11.45)          │
│  → https://lambdalabs.com/             │
│  → LAMBDA_LABS_QUICKSTART.md           │
│  → 20-30 minutes (filesystem issues)   │
└─────────────────────────────────────────┘
```

---

## ⚡ **RUNPOD SPEEDRUN** (10 minutes)

```bash
# 1. Sign up (2 min)
Go to: https://www.runpod.io/
Click: "Console" → Sign up

# 2. Launch (3 min)
Click: "Deploy"
Select: RTX 4090 or A10
Template: "RunPod PyTorch"
Click: "Deploy On-Demand"

# 3. Connect (1 min)
Click: "Connect" → "Jupyter Lab"
Opens in browser!

# 4. Setup (4 min)
# In Jupyter terminal:
git clone https://github.com/ZulAmi/symbioAI.git
cd symbioAI
pip install -r requirements.txt
python3 test_core_benchmark.py

# 5. Run (2 min)
screen -S benchmarks
python3 run_benchmarks.py --mode full
# Ctrl+A, D to detach
```

**Done! Come back in 16 hours!**

---

## 💰 **VAST.AI SPEEDRUN** (15 minutes)

```bash
# 1. Sign up (3 min)
Go to: https://vast.ai/
Sign up → Add $10

# 2. Find GPU (5 min)
Click: "Search"
Filter: RTX 4090, >95% reliability
Sort: Price (low to high)
Click: "Rent" on good option

# 3. Configure (2 min)
Template: "pytorch/pytorch:latest"
Click: "Create Instance"

# 4. Connect (2 min)
Copy SSH command
Paste in terminal

# 5. Setup (3 min)
git clone https://github.com/ZulAmi/symbioAI.git
cd symbioAI
pip install -r requirements.txt
python3 test_core_benchmark.py
screen -S benchmarks
python3 run_benchmarks.py --mode full
# Ctrl+A, D
```

**Done! Come back in 16 hours!**

---

## 🔧 **COMMON COMMANDS**

### **Check benchmark progress:**

```bash
# RunPod/Vast.ai:
screen -r benchmarks     # Reattach
# Ctrl+A, D to detach

# Or check GPU:
nvidia-smi
```

### **Download results (tomorrow):**

```bash
# From your Mac:
scp -r <user>@<ip>:~/symbioAI/experiments/results ./
```

### **Stop GPU (when done):**

```bash
# RunPod: Click "Stop" in web UI
# Vast.ai: Click "Destroy" in web UI
```

---

## 📊 **BENCHMARK MODES**

```
TEST:        30 min   $0.25      (verify works)
FAST:        4-6 hrs  $2-3       (quick results)
FULL:        12-16 hrs $6-8      (recommended!)
PUBLICATION: 24-48 hrs $12-24    (maximum quality)
```

---

## 🆘 **EMERGENCY CONTACTS**

**RunPod Discord:** discord.gg/runpod  
**Vast.ai Discord:** discord.gg/vastai  
**Email me:** (ask for help anytime!)

---

## ✅ **SUCCESS CHECKLIST**

**Setup Phase:**

- [ ] Account created
- [ ] GPU launched
- [ ] Connected (SSH or Jupyter)
- [ ] Code uploaded
- [ ] Dependencies installed
- [ ] Test passed ✅

**Running Phase:**

- [ ] Benchmarks started in screen/tmux
- [ ] GPU showing activity (nvidia-smi)
- [ ] Detached successfully
- [ ] Noted when to check back
- [ ] Set reminder on phone

**Tomorrow Phase:**

- [ ] Benchmarks completed
- [ ] Downloaded results
- [ ] Stopped/destroyed GPU
- [ ] Ran analysis
- [ ] Celebrated! 🎉

---

## 💡 **PRO TIPS**

1. **Save frequently:**

   ```bash
   git add .
   git commit -m "checkpoint"
   git push
   ```

2. **Monitor costs:**

   - RunPod: Dashboard shows running cost
   - Vast.ai: Shows cost in "My Instances"

3. **Don't forget to stop GPU!**

   - Set phone alarm for tomorrow
   - Check email reminders

4. **If disconnected:**
   - Results are still saved
   - Just reconnect and check screen/tmux

---

## 🎯 **FINAL REMINDER**

**All three providers give IDENTICAL results!**

Only difference:

- 💰 How much you pay
- ⏱️ How long setup takes
- 😊 How easy it is

**Pick one and GO!** Don't overthink! 🚀

---

**Good luck! See you in 16 hours with results!** 🎓
