#!/bin/bash
#
# Causal-DER 5-Epoch Stable Experiment
# =====================================
# 
# This script runs a STABLE 5-epoch Causal-DER experiment with corrected hyperparameters
# to prevent NaN explosions caused by high learning rates and permissive gradient clipping.
#
# Date: October 19, 2025
# Hardware: Apple Silicon (MPS)
# Framework: Mammoth v2024
#

set -e  # Exit on error

echo "=============================================="
echo "Causal-DER 5-Epoch Stable Experiment"
echo "=============================================="
echo ""
echo "Configuration:"
echo "  LR: 0.005 (6x lower than DER++)"
echo "  Gradient Clip: 1.0 (10x stricter)"
echo "  Buffer: 500"
echo "  Alpha: 0.3, Beta: 0.5"
echo "  Epochs: 5 per task"
echo ""
echo "Expected runtime: ~10-15 minutes"
echo "=============================================="
echo ""

cd /Users/zulhilmirahmat/Development/programming/Symbio\ AI/mammoth

python3 utils/main.py \
  --model causal-der \
  --dataset seq-cifar100 \
  --buffer_size 500 \
  --alpha 0.3 \
  --beta 0.5 \
  --n_epochs 5 \
  --batch_size 32 \
  --lr 0.005 \
  --optim_wd 0.0 \
  --optim_mom 0.0 \
  --clip_grad 1.0 \
  --mixed_precision 0 \
  --seed 42 \
  --dataset_config xder \
  2>&1 | tee ~/causal_der_5epoch_stable_$(date +%Y%m%d_%H%M%S).log

echo ""
echo "=============================================="
echo "Experiment completed!"
echo "=============================================="
echo ""
echo "Check results with:"
echo "  tail -100 ~/causal_der_5epoch_stable_*.log | grep -E 'Class-IL|Task-IL|NaN'"
echo ""
