#!/bin/bash

# Test Quick Wins Implementation - Phase 2 Causal with Optimizations
# Expected improvement: 70.32% → 72-73% Task-IL
#
# QUICK WIN OPTIMIZATIONS:
# 1. Warm start blending: Gradual uniform→causal transition (Tasks 0-1 uniform, 2-3 blended, 4+ full causal)
# 2. Smoother importance: 0.7*causal + 0.3*recency (vs binary weighting)
# 3. Adaptive sparsification: 0.9→0.7 quantile over tasks (keep more edges early)

echo "================================================"
echo "Quick Wins Test - Phase 2 Causal + Optimizations"
echo "================================================"
echo ""
echo "Expected performance: 72-73% Task-IL (vs 70.32% baseline)"
echo "Improvements:"
echo "  - Warm start blending (reduce cold start penalty)"
echo "  - Smoother importance scores (better diversity)"
echo "  - Adaptive sparsification (more edges early)"
echo ""
echo "Running with seed=1 for direct comparison..."
echo ""

python3 mammoth/utils/main.py \
    --model derpp-causal \
    --dataset seq-cifar100 \
    --buffer_size 500 \
    --alpha 0.3 \
    --beta 0.5 \
    --n_epochs 5 \
    --batch_size 32 \
    --lr 0.03 \
    --lr_scheduler multisteplr \
    --lr_milestones 3 4 \
    --sched_multistep_lr_gamma 0.2 \
    --minibatch_size 32 \
    --enable_causal_graph_learning 1 \
    --use_causal_sampling 1 \
    --num_tasks 10 \
    --feature_dim 512 \
    --causal_cache_size 200 \
    --seed 1 \
    2>&1 | tee validation/results/quickwins_phase2_seed1.log

echo ""
echo "================================================"
echo "Quick Wins Test Complete!"
echo "================================================"
echo ""
echo "Results saved to: validation/results/quickwins_phase2_seed1.log"
echo ""
echo "Compare with baselines:"
echo "  - Official DER++: 73.81% (no causal)"
echo "  - Phase 2 baseline: 70.32% (causal without optimizations)"
echo "  - Quick wins target: 72-73% (causal with optimizations)"
echo ""
