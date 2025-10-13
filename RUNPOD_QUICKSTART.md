# 🚀 RunPod Quick Start Guide

**Recommended:** RunPod is easier and cheaper than Lambda Labs!

---

## ✅ **SETUP (15 MINUTES)**

### **Step 1: Sign Up** (3 minutes)

1. Go to: https://www.runpod.io/
2. Click "Console" (top right)
3. Sign up with Google/GitHub
4. Add $10 credit (they often give free credits!)

---

### **Step 2: Launch GPU Pod** (5 minutes)

1. Click "➕ Deploy" (or "Pods" → "GPU Pods")

2. **Choose GPU:**

   - ✅ RTX 4090 ($0.39/hr) - Best choice!
   - ✅ RTX A4000 ($0.29/hr) - Budget option
   - ✅ A10 ($0.34/hr) - Also great

3. **Choose Template:**

   - Select: **"RunPod PyTorch"** ⭐ (Has everything!)
   - Or: "RunPod Fast Stable Diffusion" (Also has PyTorch)

4. **Settings:**

   - Container Disk: 50 GB (default is fine)
   - Volume Disk: Skip (not needed)
   - Expose HTTP Ports: Leave default
   - Expose TCP Ports: Leave default

5. Click **"Deploy On-Demand"**

6. Wait 30 seconds - instance will start!

---

### **Step 3: Connect** (2 minutes)

**Option A: Web Terminal (MOST RELIABLE)**

1. Click "Connect" → "Enable Web Terminal"
2. Opens terminal directly in browser
3. If it stops/disconnects, just refresh the page
4. More stable than Jupyter for long sessions

**Option B: Jupyter Notebook (ALSO GOOD)**

1. Click "Connect" → "Jupyter Lab"
2. Opens in browser - no SSH needed!
3. Click Terminal icon to get command line
4. Good for mixed code/terminal work

**Option C: SSH (Advanced)**

1. Click "Connect" → "SSH"
2. Copy the SSH command shown
3. Paste in your Mac terminal
4. Most stable but requires key setup

---

### **Step 4: Upload Code** (5 minutes)

**Option A: Using Git Clone (RECOMMENDED - Fastest!)** ⭐

```bash
# In Jupyter terminal or SSH:
cd /workspace
git clone https://github.com/ZulAmi/symbioAI.git
cd symbioAI

# Check files are there
ls -la
```

**Option B: Using SCP (if git doesn't work)**

```bash
# On your Mac - Package code first:
cd /Users/zulhilmirahmat/Development/programming
tar -czf symbio_ai.tar.gz "Symbio AI" --exclude='.git' --exclude='__pycache__' --exclude='data' --exclude='experiments/results'

# Upload (get SSH command from RunPod "Connect" button)
scp -P <PORT> symbio_ai.tar.gz root@<IP>:/workspace/

# Extract on RunPod:
ssh -p <PORT> root@<IP>
cd /workspace
tar -xzf symbio_ai.tar.gz
cd "Symbio AI"
```

**💡 Pro tip:** Git clone is faster and easier! Just make sure your code is pushed to GitHub first.

---

### **Step 5: Install & Run** (5 minutes)

```bash
# Install dependencies
pip install -r requirements.txt

# ✅ VERIFY SETUP (recommended!)
python3 verify_runpod_setup.py

# Should see: "🎉 ALL CHECKS PASSED!"
# If yes, continue. If no, fix issues shown.

# Test core functionality
python3 test_core_benchmark.py

# Should see: "✅✅✅ CORE BENCHMARK LOOP WORKS!"

# Start benchmarks in screen (like tmux)
screen -S benchmarks
python3 run_benchmarks.py --mode full

# Detach: Ctrl+A then D
```

---

## 💰 **COST COMPARISON**

| GPU       | RunPod       | Lambda Labs | Savings          |
| --------- | ------------ | ----------- | ---------------- |
| RTX 4090  | **$0.39/hr** | $0.50/hr    | **22% cheaper!** |
| A10 24GB  | **$0.34/hr** | $0.70/hr    | **51% cheaper!** |
| A100 40GB | **$1.14/hr** | $1.10/hr    | Similar          |

**For 16-hour benchmark run:**

- RunPod RTX 4090: **$6.24** ✅
- Lambda RTX 4090: $8.00
- **You save: $1.76** (or more if using A10)

---

## ⭐ **WHY RUNPOD IS BETTER**

| Feature               | RunPod           | Lambda Labs       |
| --------------------- | ---------------- | ----------------- |
| **Ease of use**       | ⭐⭐⭐⭐⭐       | ⭐⭐⭐            |
| **Price**             | **Cheaper**      | More expensive    |
| **Availability**      | ✅ Almost always | ❌ Often sold out |
| **Filesystem issues** | ✅ None          | ⚠️ Confusing      |
| **Jupyter built-in**  | ✅ Yes           | ❌ No             |
| **Setup time**        | **10 min**       | 20-30 min         |
| **Support**           | Great            | Good              |

---

## 🎯 **QUICK START CHECKLIST**

- [ ] Go to runpod.io
- [ ] Sign up (Google/GitHub)
- [ ] Add $10 credit
- [ ] Click "Deploy"
- [ ] Select RTX 4090 or A10
- [ ] Choose "RunPod PyTorch" template
- [ ] Deploy On-Demand
- [ ] Connect via Jupyter or SSH
- [ ] Upload code (git clone or scp)
- [ ] pip install -r requirements.txt
- [ ] python3 test_core_benchmark.py
- [ ] screen -S benchmarks
- [ ] python3 run_benchmarks.py --mode full
- [ ] Ctrl+A, D (detach)

---

## 🆘 **TROUBLESHOOTING**

### **SSH Connection Issues:**

**"container is not running" error:**

- Pod has stopped - go to RunPod console
- Check pod status (should be "Running")
- Click "Start" to restart pod
- Wait 30 seconds, try SSH again

**"Your SSH client doesn't support PTY" error:**

- This happens when running SSH with a remote command and no TTY. Fix by forcing a TTY and using bash -lc:

```bash
ssh -tt <POD_ID>@ssh.runpod.io -i ~/.ssh/id_ed25519 -- 'bash -lc "cd /workspace && git clone https://github.com/ZulAmi/symbioAI.git && cd symbioAI && ls -la"'
```

- If you still get prompted for passphrase often, add your key to the agent on macOS:

```bash
eval "$(ssh-agent -s)"
ssh-add --apple-use-keychain ~/.ssh/id_ed25519   # use -K on older macOS
```

- Alternative: connect interactively first, then run commands:

```bash
ssh <POD_ID>@ssh.runpod.io -i ~/.ssh/id_ed25519
# inside the pod
cd /workspace && git clone https://github.com/ZulAmi/symbioAI.git && cd symbioAI && ls -la
```

**Web Terminal keeps stopping:**

- Use SSH instead (more stable)
- Generate SSH key: `ssh-keygen -t ed25519 -C "your@email.com"`
- Copy public key: `cat ~/.ssh/id_ed25519.pub`
- Paste in RunPod SSH key field

### **Can't connect:**

- Check pod status is "Running"
- Click "Stop" then "Start" to restart
- If still broken, deploy new pod

### **Out of credits:**

- Click "Billing" → "Add Credits"

### **Upload too slow:**

- Use git clone instead of scp
- RunPod has fast internet!

### **Pod keeps stopping:**

- Pods auto-stop when idle (normal)
- Set up screen/tmux for long jobs:
  ```bash
  screen -S benchmarks
  python3 run_benchmarks.py
  # Ctrl+A, D to detach
  ```

---

## 🎉 **YOU'RE DONE!**

RunPod is running your benchmarks. Check back in 12-16 hours!

**Cost:** ~$6-7 total (cheaper than Lambda!)  
**Ease:** 5/5 stars (easier than Lambda!)  
**Result:** Same great benchmarks! 🚀
