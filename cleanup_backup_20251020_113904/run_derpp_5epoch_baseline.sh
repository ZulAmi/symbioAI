#!/bin/bash
#
# DER++ Baseline 5-Epoch Experiment
# ==================================
# 
# Run vanilla DER++ with the SAME configuration as Causal-DER
# for fair comparison.
#
# Date: October 19, 2025
# Hardware: Apple Silicon (MPS)
# Framework: Mammoth v2024
#

set -e  # Exit on error

echo "=============================================="
echo "DER++ Baseline 5-Epoch Experiment"
echo "=============================================="
echo ""
echo "Configuration (matching Causal-DER where possible):"
echo "  LR: 0.005"
echo "  Gradient Clip: N/A (standard DER++ has no gradient clipping)"
echo "  Buffer: 500"
echo "  Alpha: 0.3, Beta: 0.5"
echo "  Epochs: 5 per task"
echo ""
echo "NOTE: Gradient clipping unavailable in standard DER++,"
echo "      only available in Causal-DER. This may affect"
echo "      comparison if gradients explode in DER++."
echo ""
echo "Expected runtime: ~60-80 minutes"
echo "=============================================="
echo ""

cd /Users/zulhilmirahmat/Development/programming/Symbio\ AI/mammoth

python3 utils/main.py \
  --model derpp \
  --dataset seq-cifar100 \
  --buffer_size 500 \
  --alpha 0.3 \
  --beta 0.5 \
  --n_epochs 5 \
  --batch_size 32 \
  --lr 0.005 \
  --optim_wd 0.0 \
  --optim_mom 0.0 \
  --seed 42 \
  --dataset_config xder \
  2>&1 | tee ~/derpp_5epoch_baseline_$(date +%Y%m%d_%H%M%S).log

echo ""
echo "=============================================="
echo "DER++ Baseline completed!"
echo "=============================================="
echo ""
echo "Compare results:"
echo "  Causal-DER: 0.9% Class-IL, 11.25% Task-IL"
echo "  DER++ baseline: Check log for results"
echo ""
