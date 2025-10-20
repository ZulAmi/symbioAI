#!/bin/bash
#
# Causal-DER 50-Epoch FIXED Experiment (Prepared)
# ===============================================
# Same Priority 1 fixes, scaled to 50 epochs for full benchmark runs.
# NOTE: Prepared only. Not executed automatically.
#
# Settings:
# - clip_grad: 10.0
# - use_causal_sampling: 0
# - use_mir_sampling: 0
# - temperature: 1.0
# - mixed_precision: 0
# - feature_kd_weight: 0.0, store_features: 0
# - n_epochs: 50
#
# Date: October 19, 2025
#
set -e

cd /Users/zulhilmirahmat/Development/programming/Symbio\ AI/mammoth

echo "=============================================="
echo "Causal-DER 50-Epoch FIXED (Prepared)"
echo "=============================================="
echo "lr=0.005, clip_grad=10.0, T=1.0, causal_sampling=0, MIR=0, epochs=50"
echo "=============================================="

python3 utils/main.py \
  --model causal-der \
  --dataset seq-cifar100 \
  --buffer_size 500 \
  --alpha 0.3 \
  --beta 0.5 \
  --n_epochs 50 \
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
  2>&1 | tee ~/causal_der_50epoch_fixed_$(date +%Y%m%d_%H%M%S).log