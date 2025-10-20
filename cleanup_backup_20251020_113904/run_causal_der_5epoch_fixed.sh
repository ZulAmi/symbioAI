#!/bin/bash
#
# Causal-DER 5-Epoch FIXED Experiment
# ====================================
# Apply Priority 1 fixes to recover performance towards DER++ baseline.
#
# Changes vs stable:
# - clip_grad: 1.0 -> 10.0
# - use_causal_sampling: 1 -> 0
# - use_mir_sampling: 1 -> 0
# - temperature: 2.0 -> 1.0
# - mixed_precision: 0 (keep disabled for stability)
# - feature_kd_weight: 0.0 (ensure no feature KD)
#
# Date: October 19, 2025
# Hardware: Apple Silicon (MPS)
# Framework: Mammoth v2024
#
set -e

echo "=============================================="
echo "Causal-DER 5-Epoch FIXED Experiment"
echo "=============================================="
echo ""
echo "Configuration (Priority 1 fixes):"
echo "  LR: 0.005"
echo "  Gradient Clip: 10.0"
echo "  Buffer: 500"
echo "  Alpha: 0.3, Beta: 0.5"
echo "  Temperature: 1.0"
echo "  Causal Sampling: OFF"
echo "  MIR Sampling: OFF"
echo "  Mixed Precision: OFF"
echo ""
echo "Expected runtime: ~60-80 minutes"
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
  --clip_grad 10.0 \
  --temperature 1.0 \
  --use_causal_sampling 0 \
  --use_mir_sampling 0 \
  --mixed_precision 0 \
  --feature_kd_weight 0.0 \
  --store_features 0 \
  --seed 42 \
  --dataset_config xder \
  2>&1 | tee ~/causal_der_5epoch_fixed_$(date +%Y%m%d_%H%M%S).log

echo ""
echo "=============================================="
echo "Experiment completed!"
echo "=============================================="
echo ""
echo "Check results with:"
echo "  tail -100 ~/causal_der_5epoch_fixed_*.log | grep -E 'Class-IL|Task-IL'"
echo ""