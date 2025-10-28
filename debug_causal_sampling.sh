#!/bin/bash
################################################################################
# Quick Debug: Causal Sampling Diagnostics
# 
# Runs 1 EPOCH only to quickly see importance score distributions
# Expected runtime: ~10-15 minutes
################################################################################

RESULTS_DIR="validation/results/debug_causal_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RESULTS_DIR"

# Base configuration (1 epoch for speed)
BASE_PARAMS="--dataset seq-cifar100 \
--buffer_size 500 \
--n_epochs 1 \
--batch_size 32 \
--minibatch_size 32 \
--lr 0.03 \
--lr_scheduler multisteplr \
--lr_milestones 3 4 \
--sched_multistep_lr_gamma 0.2 \
--seed 1"

echo "╔════════════════════════════════════════════════════════════════════╗"
echo "║              DEBUG: CAUSAL SAMPLING DIAGNOSTICS                    ║"
echo "╚════════════════════════════════════════════════════════════════════╝"
echo ""
echo "PURPOSE: Quickly capture importance score distributions"
echo "RUNTIME: ~10-15 minutes (1 epoch only)"
echo ""
echo "Will print per-task diagnostics showing:"
echo "  • Importance score statistics (mean/std/min/max)"
echo "  • Top-10 samples by probability (task ID, importance, prob)"
echo "  • Warmup blend factor"
echo "  • Causal vs final sampling probability distributions"
echo ""
echo "Results will be saved to: $RESULTS_DIR"
echo ""

# Run with default fixed parameters (blend=0.3, warmup=5)
echo "════════════════════════════════════════════════════════════════════"
echo "  Running: Fixed CausalDER (blend=0.3, warmup=5) - 1 EPOCH"
echo "════════════════════════════════════════════════════════════════════"
python3 mammoth/utils/main.py --model derpp-causal $BASE_PARAMS \
  --alpha 0.3 \
  --beta 0.5 \
  --enable_causal_graph_learning 1 \
  --use_causal_sampling 1 \
  2>&1 | tee "$RESULTS_DIR/debug_causal.log"

echo ""
echo "════════════════════════════════════════════════════════════════════"
echo "  Debug run complete!"
echo "════════════════════════════════════════════════════════════════════"
echo ""
echo "Check the log for [DEBUG] sections showing importance distributions"
echo ""
echo "Key things to look for:"
echo "  1. Are importance scores concentrated (high std, big gap min→max)?"
echo "     → If yes: Need diversity (temperature scaling or top-k)"
echo ""
echo "  2. Are top-10 samples all from same 1-2 tasks?"
echo "     → If yes: Recent bias problem, need better blending"
echo ""
echo "  3. Is mean importance score very low (< 0.2)?"
echo "     → If yes: Causal graph not providing signal, check indexing"
echo ""
echo "  4. Does causal_blend stay at 0 for many tasks?"
echo "     → If yes: Warmup too long, reduce causal_warmup_tasks"
echo ""
echo "Results saved to: $RESULTS_DIR/debug_causal.log"
echo ""
echo "To view DEBUG sections only:"
echo "  grep DEBUG $RESULTS_DIR/debug_causal.log"
echo ""
