#!/bin/bash
#
# Test Official DER++ Causal Extension - Phase 2 (Causal ON)
# ===========================================================
#
# This tests causal graph learning ON TOP of official DER++.
#
# Expected Performance:
#   - Baseline (causal OFF): 73.81% Task-IL
#   - Causal (graph ON): ≥70% Task-IL (target)
#
# File: training/derpp_causal.py (extends official derpp.py)
#

echo "========================================================================"
echo "🔬 Testing Causal Graph Learning ON (Phase 2)"
echo "========================================================================"
echo ""
echo "Architecture:"
echo "  Base:       mammoth/models/derpp.py (official 73.81%)"
echo "  Extension:  training/derpp_causal.py (causal layer)"
echo ""
echo "Configuration:"
echo "  --enable_causal_graph_learning 1  ✅ CAUSAL ON"
echo "  --use_causal_sampling 1           ✅ USE GRAPH FOR REPLAY"
echo ""
echo "Baseline: 73.81% Task-IL (official DER++)"
echo "Target:   ≥70% Task-IL (acceptable drop ≤3.81%)"
echo ""
echo "========================================================================"
echo ""

# Create results directory
mkdir -p validation/results/official_derpp_causal

echo "Running full 5-epoch test with causal graph learning..."
echo "ETA: ~52 minutes"
echo ""

# Run with causal graph learning enabled
python3 mammoth/utils/main.py \
    --model derpp-causal \
    --dataset seq-cifar100 \
    --buffer_size 500 \
    --alpha 0.3 \
    --beta 0.5 \
    --n_epochs 5 \
    --batch_size 32 \
    --minibatch_size 32 \
    --lr 0.03 \
    --lr_scheduler multisteplr \
    --lr_milestones 3 4 \
    --sched_multistep_lr_gamma 0.2 \
    --enable_causal_graph_learning 1 \
    --use_causal_sampling 1 \
    --temperature 2.0 \
    --num_tasks 10 \
    --feature_dim 512 \
    --causal_cache_size 200 \
    --seed 1 \
    2>&1 | tee validation/results/official_derpp_causal/causal_on_5epoch_seed1.log

# Extract result
echo ""
echo "========================================================================"
echo "Causal Graph Learning Results (5 epochs):"
echo "========================================================================"
grep "Accuracy for 10 task" validation/results/official_derpp_causal/causal_on_5epoch_seed1.log | tail -1

# Compare with baseline
echo ""
echo "Comparison:"
echo "  Official DER++ baseline:  73.81%"
echo "  This test (causal ON):    [see above]"
echo ""
echo "Log: validation/results/official_derpp_causal/causal_on_5epoch_seed1.log"
echo ""
echo "Decision criteria:"
echo "  ✅ GO to Option A if:     ≥70% Task-IL"
echo "  🟡 MAYBE if:              65-70% Task-IL (interesting problem)"
echo "  ❌ NO-GO if:              <65% Task-IL (too broken)"
echo ""
