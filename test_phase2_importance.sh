#!/bin/bash

################################################################
# Phase 2: Importance-Weighted Sampling Test
################################################################
# 
# Phase 1 Baseline: 70.19% Task-IL (VALIDATED ✅)
# Phase 2 Target:   71-72% Task-IL (+1-2% improvement)
#
# New Features:
# - Importance scoring based on loss + (1 - confidence)
# - 70% sampling by importance, 30% random
# - Tracks importance statistics per task
#
################################################################

echo "================================================================"
echo "PHASE 2: IMPORTANCE-WEIGHTED SAMPLING TEST"
echo "================================================================"
echo ""
echo "Baseline (Phase 1):  70.19% Task-IL"
echo "Target (Phase 2):    71-72% Task-IL (+1-2% improvement)"
echo ""
echo "New Features:"
echo "  - Importance scoring: loss + (1 - confidence)"
echo "  - Sampling: 70% by importance, 30% random"
echo "  - Importance stats tracking per task"
echo ""
echo "Configuration:"
echo "  alpha=0.3, beta=0.5, lr=0.03, momentum=0"
echo "  buffer_size=500, n_epochs=5"
echo "  batch_size=32, minibatch_size=32"
echo "  use_importance_sampling=1, importance_weight=0.7"
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
  --importance_weight 0.7 \
  --nowand 1 \
  --seed 1 \
  2>&1 | tee ../validation/results/phase2_importance_seed1.log

cd ..

echo ""
echo "================================================================"
echo "PHASE 2 TEST COMPLETE!"
echo "================================================================"
echo ""
echo "Baseline (Phase 1):  70.19% Task-IL"
echo "Phase 2 Result:      [Check log above]"
echo ""
echo "Results saved to: validation/results/phase2_importance_seed1.log"
echo ""
echo "Next steps:"
echo "  ✅ If improved (71-72%): Phase 2 SUCCESS! Proceed to Phase 3"
echo "  ⚠️  If same (70%): Importance sampling neutral, try Phase 3 anyway"
echo "  ❌ If worse (<70%): Debug importance scoring formula"
echo ""
