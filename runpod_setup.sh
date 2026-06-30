#!/bin/bash
# ============================================================
# RUNPOD SETUP SCRIPT — SymbioAI Causal Continual Learning
# Run once on a fresh RunPod instance to set up everything.
# ============================================================

set -e  # Exit on error

echo "============================================================"
echo "SYMBIO AI — RUNPOD SETUP"
echo "============================================================"
echo ""

# 1. Clone SymbioAI (with Mammoth as submodule)
echo "[1/4] Cloning SymbioAI with Mammoth submodule..."
cd /workspace
if [ ! -d "symbioAI" ]; then
    git clone --recurse-submodules https://github.com/ZulAmi/symbioAI.git
    echo "✅ SymbioAI cloned (includes mammoth/ submodule)"
else
    cd symbioAI
    git pull origin main
    git submodule update --init --recursive
    echo "✅ SymbioAI updated"
    cd ..
fi

# 2. Enter the package root
cd /workspace/symbioAI/symbioAI

# 3. Install dependencies
echo ""
echo "[2/4] Installing dependencies..."
pip install -q -e ".[tracking,viz]"
echo "✅ Dependencies installed"

# 4. Verify setup
echo ""
echo "[3/4] Verifying installation..."
python - <<'EOF'
import torch
cuda = torch.cuda.is_available()
device = torch.cuda.get_device_name(0) if cuda else "CPU"
print(f"✅ PyTorch {torch.__version__} | CUDA: {cuda} | Device: {device}")
from pathlib import Path
mammoth_main = Path("mammoth/utils/main.py")
if mammoth_main.exists():
    print(f"✅ Mammoth submodule found at {mammoth_main}")
else:
    print("❌ ERROR: mammoth submodule not found — run: git submodule update --init --recursive")
    exit(1)
EOF

# 5. Smoke test (1 epoch, vanilla mode — fast, no GPU needed to verify wiring)
echo ""
echo "[4/4] Running 1-epoch smoke test (vanilla DER++)..."
python run_optimized_true_causality.py --use_causal_sampling 0 --n_epochs 1 --seed 1
echo "✅ Smoke test passed"

echo ""
echo "============================================================"
echo "READY TO RUN EXPERIMENTS"
echo "============================================================"
echo ""
echo "All commands should be run from:  /workspace/symbioAI/symbioAI/"
echo ""
echo "Quick reference:"
echo ""
echo "  # Vanilla DER++ baseline (~43 min / seed)"
echo "  python run_optimized_true_causality.py --use_causal_sampling 0 --seed 1"
echo ""
echo "  # TRUE interventional causality (~3-5 h / seed)"
echo "  python run_optimized_true_causality.py --use_causal_sampling 3 --seed 1"
echo ""
echo "  # All 5 seeds (vanilla + TRUE) — for paper results"
echo "  for seed in 1 2 3 4 5; do"
echo "    python run_optimized_true_causality.py --use_causal_sampling 0 --seed \$seed"
echo "    python run_optimized_true_causality.py --use_causal_sampling 3 --seed \$seed"
echo "  done"
echo ""
echo "  # W&B logging"
echo "  WANDB_API_KEY=<your_key> python run_optimized_true_causality.py --use_causal_sampling 3 --seed 1 --wandb"
echo ""
echo "Modes: 0=vanilla  1=heuristic  2=hybrid  3=TRUE-causality  4=influence-fn"
echo ""
