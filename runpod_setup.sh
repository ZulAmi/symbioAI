#!/bin/bash
# ============================================================
# RUNPOD COMPLETE SETUP SCRIPT
# Run this on a fresh RunPod instance to set up everything
# ============================================================

set -e  # Exit on error

echo "============================================================"
echo "SYMBIO AI - RUNPOD SETUP"
echo "============================================================"
echo ""

# 1. Clone the official Mammoth framework
echo "[1/6] Cloning Mammoth framework..."
cd /workspace
if [ ! -d "mammoth" ]; then
    git clone https://github.com/aimagelab/mammoth.git
    echo "✅ Mammoth cloned"
else
    echo "⚠️  Mammoth already exists, skipping"
fi

# 2. Clone YOUR repository
echo ""
echo "[2/6] Cloning SymbioAI repository..."
if [ ! -d "symbioAI" ]; then
    git clone https://github.com/ZulAmi/symbioAI.git
    echo "✅ SymbioAI cloned"
else
    cd symbioAI
    git pull origin main
    echo "✅ SymbioAI updated"
    cd ..
fi

# 3. Copy custom files INTO Mammoth
echo ""
echo "[3/6] Copying custom files to Mammoth..."
cp /workspace/symbioAI/training/derpp_causal.py /workspace/mammoth/models/
cp /workspace/symbioAI/training/causal_inference.py /workspace/mammoth/models/
echo "✅ Custom files copied"

# 4. Install Mammoth dependencies
echo ""
echo "[4/6] Installing Mammoth dependencies..."
cd /workspace/mammoth
pip install -q -r requirements.txt
echo "✅ Dependencies installed"

# 5. Verify setup
echo ""
echo "[5/6] Verifying installation..."
if [ -f "/workspace/mammoth/models/derpp_causal.py" ]; then
    echo "✅ derpp_causal.py found"
else
    echo "❌ ERROR: derpp_causal.py not found!"
    exit 1
fi

if [ -f "/workspace/mammoth/models/causal_inference.py" ]; then
    echo "✅ causal_inference.py found"
else
    echo "❌ ERROR: causal_inference.py not found!"
    exit 1
fi

# 6. Ready to run
echo ""
echo "[6/6] Setup complete!"
echo ""
echo "============================================================"
echo "READY TO RUN EXPERIMENTS"
echo "============================================================"
echo ""
echo "Run the following to start 3 parallel experiments:"
echo ""
echo "  bash /workspace/symbioAI/runpod_run_experiments.sh"
echo ""
echo "Or manually start each in tmux:"
echo ""
echo "# Vanilla DER++"
echo "tmux new -s vanilla -d"
echo "tmux send-keys -t vanilla 'cd /workspace/mammoth && python utils/main.py --model derpp --dataset seq-cifar100 --buffer_size 500 --alpha 0.1 --beta 0.5 --n_epochs 5 --lr 0.03 --lr_milestones 3 4 --seed 1 2>&1 | grep -v NNPACK | tee /workspace/vanilla_5ep.log' C-m"
echo ""
echo "# Graph Heuristic"
echo "tmux new -s graph -d"
echo "tmux send-keys -t graph 'cd /workspace/mammoth && python utils/main.py --model derpp-causal --dataset seq-cifar100 --buffer_size 500 --alpha 0.1 --beta 0.5 --n_epochs 5 --lr 0.03 --lr_milestones 3 4 --enable_causal_graph_learning 1 --use_causal_sampling 1 --seed 1 2>&1 | grep -v NNPACK | tee /workspace/graph_5ep.log' C-m"
echo ""
echo "# TRUE Interventional"
echo "tmux new -s true -d"
echo "tmux send-keys -t true 'cd /workspace/mammoth && python utils/main.py --model derpp-causal --dataset seq-cifar100 --buffer_size 500 --alpha 0.1 --beta 0.5 --n_epochs 5 --lr 0.03 --lr_milestones 3 4 --use_causal_sampling 3 --debug 1 --temperature 2.0 --seed 1 2>&1 | grep -v NNPACK | tee /workspace/true_5ep.log' C-m"
echo ""
echo "Monitor with: tmux attach -t vanilla (or graph, true)"
echo "Detach with: Ctrl+B then D"
echo ""
