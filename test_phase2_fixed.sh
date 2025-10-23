#!/bin/bash

################################################################
# Phase 2 FIXED: Improved Importance-Weighted Sampling
################################################################
# 
# Previous (Phase 2v1): 69.32% Task-IL (WORSE than baseline ❌)
# Phase 1 Baseline:     70.19% Task-IL 
# Target:               71-72% Task-IL (+1-2% improvement)
#
# FIXES:
# 1. Better formula: loss * uncertainty^2 (multiplicative)
# 2. More noise: 0.1 instead of 0.01 (10x increase)
# 3. Balanced sampling: 50/50 instead of 70/30
# 4. Better normalization: min-max scaling for loss
#
################################################################

echo "================================================================"
echo "PHASE 2 FIXED: IMPROVED IMPORTANCE SAMPLING"
echo "================================================================"
echo ""
echo "Previous Results:"
echo "  Phase 1 Baseline:  70.19% Task-IL ✅"
echo "  Phase 2 (v1):      69.32% Task-IL ❌ (-0.87%)"
echo ""
echo "Target: 71-72% Task-IL (+1-2% over baseline)"
echo ""
echo "Improvements:"
echo "  1. Formula: loss * uncertainty^2 (emphasize both factors)"
echo "  2. Noise: 0.1 (10x larger for diversity)"
echo "  3. Balance: 50% importance, 50% random (less bias)"
echo "  4. Better loss normalization (min-max scaling)"
echo ""
echo "Configuration:"
echo "  alpha=0.3, beta=0.5, lr=0.03, momentum=0"
echo "  buffer_size=500, n_epochs=5"
echo "  batch_size=32, minibatch_size=32"
echo "  use_importance_sampling=1, importance_weight=0.5"
echo ""
echo "================================================================"
echo ""

cd mammoth

python3 utils/main.py \
  --model causal-der \
  --dataset seq-cifar100 \
  --buffer_size 500 \
  --alpha 0.3 \
  --beta 0.5 \
  --n_epochs 5 \
  --batch_size 32 \
  --minibatch_size 32 \
  --lr 0.03 \
  --optim_mom 0.0 \
  --optim_wd 0.0 \
  --lr_scheduler multisteplr \
  --lr_milestones 3 4 \
  --sched_multistep_lr_gamma 0.2 \
  --use_importance_sampling 1 \
  --importance_weight 0.5 \
  --seed 1 \
  2>&1 | tee ../validation/results/phase2_fixed_seed1.log

cd ..

echo ""
echo "================================================================"
echo "PHASE 2 FIXED TEST COMPLETE!"
echo "================================================================"
echo ""
echo "Baseline (Phase 1):  70.19% Task-IL"
echo "Previous (Phase 2):  69.32% Task-IL ❌"
echo "Fixed (Phase 2):     [Check log above]"
echo ""
echo "Results saved to: validation/results/phase2_fixed_seed1.log"
echo ""
echo "Decision Tree:"
echo "  ✅ If 71-72%: FIXED! Importance sampling works with better formula"
echo "  ⚠️  If 70-71%: Small gain, acceptable. Try more tweaks or move to Phase 3"
echo "  ❌ If <70%:  Still broken. Consider alternative approaches"
echo ""
