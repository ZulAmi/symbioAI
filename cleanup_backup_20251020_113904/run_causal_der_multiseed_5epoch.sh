#!/bin/bash
#
# Causal-DER 5-Epoch Multi-Seed (Prepared)
# ========================================
# Runs Causal-DER with Priority 1 fixes across multiple seeds.
# NOTE: This script is prepared but NOT executed automatically.
#
# Fixes applied:
# - clip_grad: 10.0 (was 1.0)
# - use_causal_sampling: 0 (disable)
# - use_mir_sampling: 0 (disable)
# - temperature: 1.0 (was 2.0)
# - mixed_precision: 0 (off)
# - feature_kd_weight: 0.0, store_features: 0
#
# Date: October 19, 2025
# Hardware: Apple Silicon (MPS)
# Framework: Mammoth v2024
#
set -e

SEEDS=(42 43 44)
PROJECT_ROOT="/Users/zulhilmirahmat/Development/programming/Symbio AI"
cd "$PROJECT_ROOT/mammoth"

echo "=============================================="
echo "Causal-DER 5-Epoch Multi-Seed (Prepared)"
echo "=============================================="
echo "Seeds: ${SEEDS[*]}"
echo "Config: lr=0.005, clip_grad=10.0, T=1.0, causal_sampling=0, MIR=0"
echo "=============================================="

for SEED in "${SEEDS[@]}"; do
  echo "\n--- Prepared command for seed=${SEED} ---\n"
  echo "python3 utils/main.py \\
  --model causal-der \\
  --dataset seq-cifar100 \\
  --buffer_size 500 \\
  --alpha 0.3 \\
  --beta 0.5 \\
  --n_epochs 5 \\
  --batch_size 32 \\
  --lr 0.005 \\
  --optim_wd 0.0 \\
  --optim_mom 0.0 \\
  --clip_grad 10.0 \\
  --temperature 1.0 \\
  --use_causal_sampling 0 \\
  --use_mir_sampling 0 \\
  --mixed_precision 0 \\
  --feature_kd_weight 0.0 \\
  --store_features 0 \\
  --seed ${SEED} \\
  --dataset_config xder \\
  2>&1 | tee ~/causal_der_5epoch_multiseed_${SEED}_$(date +%Y%m%d_%H%M%S).log"

done

echo "\nNOTE: To run, remove the echo and execute the commands above or uncomment the block below."
: <<'RUN_IT'
for SEED in "${SEEDS[@]}"; do
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
    --seed ${SEED} \
    --dataset_config xder \
    2>&1 | tee ~/causal_der_5epoch_multiseed_${SEED}_$(date +%Y%m%d_%H%M%S).log
done
RUN_IT
