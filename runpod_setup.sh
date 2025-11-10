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
# Copy model files
cp /workspace/symbioAI/training/derpp_causal.py /workspace/mammoth/models/
cp /workspace/symbioAI/training/causal_inference.py /workspace/mammoth/models/

# Copy Python runner script
cp /workspace/symbioAI/run_optimized_true_causality.py /workspace/mammoth/

# Copy validation directory (for results)
cp -r /workspace/symbioAI/validation /workspace/mammoth/

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
echo "Quick Start - Run optimized TRUE causality (3x faster):"
echo ""
echo "  cd /workspace/mammoth"
echo "  python3 run_optimized_true_causality.py"
echo ""
echo "This will run 5-epoch optimized TRUE causality (~15 min vs 43 min baseline)"
echo ""
echo "Options:"
echo "  # 1-epoch quick test"
echo "  python3 run_optimized_true_causality.py --n_epochs 1"
echo ""
echo "  # Different seed"
echo "  python3 run_optimized_true_causality.py --seed 42"
echo ""
echo "  # See all options"
echo "  python3 run_optimized_true_causality.py --help"
echo ""
