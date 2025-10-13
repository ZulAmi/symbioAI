# 🚀 Vast.ai Quick Start Guide

**For the CHEAPEST GPU rental** (50-70% cheaper than Lambda/RunPod)

---

## ⚡ **WHY VAST.AI?**

- ✅ **CHEAPEST:** RTX 4090 from $0.20-0.30/hour (vs $0.50)
- ✅ **Marketplace model:** Rent from individuals
- ✅ **More control:** Pick exact specs you want
- ⚠️ **Trade-off:** Slightly less reliable than big providers

**Cost for benchmarks:**

- RTX 4090: $0.25/hr × 16 hrs = **$4.00** 🎯
- A10: $0.15/hr × 18 hrs = **$2.70** 🎯

---

## ✅ **SETUP (20 MINUTES)**

### **Step 1: Sign Up** (5 minutes)

1. Go to: https://vast.ai/
2. Click "Sign In" → "Sign Up"
3. Add $10 credit
4. Upload SSH key or generate new one

---

### **Step 2: Find GPU** (5 minutes)

1. Click "Search" tab

2. **Set Filters:**

   - GPU Model: RTX 4090 (or A10, RTX 3090)
   - Min VRAM: 20 GB
   - Min Upload: 100 Mbps
   - Location: North America
   - Template: PyTorch

3. **Sort by:** $/hr (cheapest first)

4. Look for:

   - ✅ High reliability score (>95%)
   - ✅ Good connection speed
   - ✅ Reasonable price ($0.20-0.40/hr)

5. Click "Rent" on a good option

---

### **Step 3: Configure Instance** (3 minutes)

1. **Image/Template:** Select "pytorch/pytorch:latest"

2. **Disk Space:** 30 GB (default)

3. **On-demand:** Yes (can stop anytime)

4. Click "Create Instance"

5. Wait 1-2 minutes for it to start

---

### **Step 4: Connect** (2 minutes)

1. Click "Connect" button
2. Copy SSH command
3. Paste in Mac terminal

```bash
# Will look like:
ssh -p 12345 root@123.456.789.0 -L 8080:localhost:8080
```

---

### **Step 5: Setup & Run** (5 minutes)

```bash
# On Vast.ai instance:

# Clone your code
git clone https://github.com/ZulAmi/symbioAI.git
cd symbioAI

# Install dependencies
pip install -r requirements.txt

# Test
python3 test_core_benchmark.py

# Run in screen
screen -S benchmarks
python3 run_benchmarks.py --mode full

# Detach: Ctrl+A, D
```

---

## 💰 **COST COMPARISON (16-hour benchmark)**

| Provider     | RTX 4090     | Total Cost   |
| ------------ | ------------ | ------------ |
| **Vast.ai**  | **$0.25/hr** | **$4.00** 🏆 |
| RunPod       | $0.39/hr     | $6.24        |
| Lambda Labs  | $0.50/hr     | $8.00        |
| **SAVINGS:** | -            | **Save $4!** |

---

## ⭐ **PROS & CONS**

### **✅ Pros:**

- 50-70% cheaper!
- More GPU availability
- More variety of GPUs
- Can find great deals

### **⚠️ Cons:**

- Slightly more complex setup
- Individual hosts (not data centers)
- Connection can be slower
- Rare chance of host canceling

---

## 🎯 **BEST PRACTICES**

1. **Check reliability score:** Pick hosts with >95%
2. **Check connection speed:** Min 100 Mbps upload
3. **Start with small test:** Run test mode first
4. **Save your work:** Git commit frequently
5. **Use screen/tmux:** In case connection drops

---

## 🆘 **TROUBLESHOOTING**

### **Instance stopped unexpectedly:**

- Host may have terminated
- Your work is saved if you committed to git
- Rent another instance and continue

### **Slow upload:**

- Use git clone instead of scp
- Choose host with faster connection

### **Can't connect:**

- Check your SSH key is correct
- Try the "Open SSH" button in Vast.ai console

---

## 📊 **RECOMMENDATION**

**Use Vast.ai if:**

- ✅ Budget is tight ($4 vs $8)
- ✅ You're comfortable with SSH
- ✅ You want cheapest option

**Use RunPod if:**

- ✅ You want easiest setup
- ✅ You want Jupyter notebook
- ✅ You prefer reliability over price

**Use Lambda if:**

- ✅ You already started there
- ✅ You don't mind paying more

---

## ✅ **QUICK CHECKLIST**

- [ ] Sign up at vast.ai
- [ ] Add $10 credits
- [ ] Upload SSH key
- [ ] Search for RTX 4090
- [ ] Filter: >95% reliability, >100 Mbps
- [ ] Rent cheapest good option
- [ ] Select PyTorch template
- [ ] Create instance
- [ ] SSH connect
- [ ] git clone symbioAI
- [ ] pip install -r requirements.txt
- [ ] python3 test_core_benchmark.py
- [ ] screen -S benchmarks
- [ ] python3 run_benchmarks.py --mode full
- [ ] Ctrl+A, D

---

**Total cost: ~$4 (HALF the price of Lambda!)** 🎉
