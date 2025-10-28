#!/bin/bash
################################################################################
# CausalDER - ABLATION STUDIES
# 
# Purpose: Prove each component contributes to performance
# Critical for publication - shows your method is scientifically sound
################################################################################

RESULTS_DIR="validation/results/ablations_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RESULTS_DIR"

# Base configuration (matches your validated experiments)
BASE_PARAMS="--dataset seq-cifar100 \
--buffer_size 500 \
--n_epochs 5 \
--batch_size 32 \
--minibatch_size 32 \
--lr 0.03 \
--lr_scheduler multisteplr \
--lr_milestones 3 4 \
--sched_multistep_lr_gamma 0.2 \
--seed 1"

echo "╔════════════════════════════════════════════════════════════════════╗"
echo "║                    ABLATION STUDIES - CausalDER                    ║"
echo "╚════════════════════════════════════════════════════════════════════╝"
echo ""
echo "Purpose: Validate each component's contribution"
echo "Expected time: ~5 hours (6 experiments × 50 mins)"
echo "Results will be saved to: $RESULTS_DIR"
echo ""

# Experiment 1: Full CausalDER (reference)
echo "════════════════════════════════════════════════════════════════════"
echo "  Experiment 1/6: Full CausalDER"
echo "  Expected: ~72.01% (your validated result)"
echo "════════════════════════════════════════════════════════════════════"
python3 mammoth/utils/main.py --model derpp-causal $BASE_PARAMS \
  --alpha 0.3 \
  --beta 0.5 \
  --enable_causal_graph_learning 1 \
  --use_causal_sampling 1 \
  --causal_blend_ratio 0.7 \
  2>&1 | tee "$RESULTS_DIR/1_full_causal.log"

acc=$(grep "Accuracy for 10 task" "$RESULTS_DIR/1_full_causal.log" | grep "Task-IL" | tail -1 | grep -oE '[0-9]+\.[0-9]+' | head -1)
echo "  ✅ Full CausalDER: $acc%"
echo ""

# Experiment 2: Uniform sampling (no causal - equivalent to DER++)
echo "════════════════════════════════════════════════════════════════════"
echo "  Experiment 2/6: Uniform Sampling (No Causal)"
echo "  Expected: ~72.99% (should match baseline)"
echo "  Purpose: Shows performance without causal components"
echo "════════════════════════════════════════════════════════════════════"
python3 mammoth/utils/main.py --model derpp $BASE_PARAMS \
  --alpha 0.3 \
  --beta 0.5 \
  2>&1 | tee "$RESULTS_DIR/2_uniform_sampling.log"

acc=$(grep "Accuracy for 10 task" "$RESULTS_DIR/2_uniform_sampling.log" | grep "Task-IL" | tail -1 | grep -oE '[0-9]+\.[0-9]+' | head -1)
echo "  ✅ Uniform Sampling: $acc%"
echo ""

# Experiment 3: Graph learning only (no causal sampling)
echo "════════════════════════════════════════════════════════════════════"
echo "  Experiment 3/6: Graph Learning Only (No Causal Sampling)"
echo "  Expected: ~70-71%"
echo "  Purpose: Shows causal sampling is necessary"
echo "════════════════════════════════════════════════════════════════════"
python3 mammoth/utils/main.py --model derpp-causal $BASE_PARAMS \
  --alpha 0.3 \
  --beta 0.5 \
  --enable_causal_graph_learning 1 \
  --use_causal_sampling 0 \
  2>&1 | tee "$RESULTS_DIR/3_graph_only.log"

acc=$(grep "Accuracy for 10 task" "$RESULTS_DIR/3_graph_only.log" | grep "Task-IL" | tail -1 | grep -oE '[0-9]+\.[0-9]+' | head -1)
echo "  ✅ Graph Only: $acc%"
echo ""

# Experiment 4: Random graph (proves learned > random)
echo "════════════════════════════════════════════════════════════════════"
echo "  Experiment 4/6: Random Graph Baseline"
echo "  Expected: ~70-71%"
echo "  Purpose: Proves learned causal structure > random structure"
echo "════════════════════════════════════════════════════════════════════"
# Note: This requires adding --use_random_graph flag to your code
# For now, we'll skip this if the flag doesn't exist
echo "  ⚠️  Skipping random graph (requires code modification)"
echo "  To implement: Add random graph generation in causal_inference.py"
echo ""

# Experiment 5: No warm start blending
echo "════════════════════════════════════════════════════════════════════"
echo "  Experiment 5/6: No Warm Start Blending"
echo "  Expected: ~71-72%"
echo "  Purpose: Shows warm start helps early tasks"
echo "════════════════════════════════════════════════════════════════════"
# Note: This requires modifying causal blend to be 1.0 from start
echo "  ⚠️  Skipping (requires modifying warm start logic)"
echo "  To implement: Set initial causal_blend=1.0 instead of gradual ramp"
echo ""

# Experiment 6: Simpler importance weighting
echo "════════════════════════════════════════════════════════════════════"
echo "  Experiment 6/6: Simple Binary Weighting (No Hybrid)"
echo "  Expected: ~71-72%"
echo "  Purpose: Shows hybrid weighting (0.7 causal + 0.3 recency) helps"
echo "════════════════════════════════════════════════════════════════════"
echo "  ⚠️  Skipping (requires modifying importance weighting)"
echo "  To implement: Use binary weighting instead of 0.7/0.3 blend"
echo ""

# Summary
echo "╔════════════════════════════════════════════════════════════════════╗"
echo "║                        ABLATION SUMMARY                            ║"
echo "╚════════════════════════════════════════════════════════════════════╝"
echo ""
echo "Completed Experiments:"
echo "───────────────────────────────────────────────────────────────────"

exp1=$(grep "Accuracy for 10 task" "$RESULTS_DIR/1_full_causal.log" 2>/dev/null | grep "Task-IL" | tail -1 | grep -oE '[0-9]+\.[0-9]+' | head -1)
exp2=$(grep "Accuracy for 10 task" "$RESULTS_DIR/2_uniform_sampling.log" 2>/dev/null | grep "Task-IL" | tail -1 | grep -oE '[0-9]+\.[0-9]+' | head -1)
exp3=$(grep "Accuracy for 10 task" "$RESULTS_DIR/3_graph_only.log" 2>/dev/null | grep "Task-IL" | tail -1 | grep -oE '[0-9]+\.[0-9]+' | head -1)

if [ -n "$exp1" ]; then
    echo "  1. Full CausalDER:         $exp1%"
fi
if [ -n "$exp2" ]; then
    echo "  2. Uniform Sampling:       $exp2%"
    if [ -n "$exp1" ]; then
        delta=$(python3 -c "print(f'{float($exp2) - float($exp1):.2f}')" 2>/dev/null || echo "N/A")
        echo "     → Cost of causal:     $delta%"
    fi
fi
if [ -n "$exp3" ]; then
    echo "  3. Graph Only:             $exp3%"
    if [ -n "$exp1" ]; then
        delta=$(python3 -c "print(f'{float($exp1) - float($exp3):.2f}')" 2>/dev/null || echo "N/A")
        echo "     → Causal sampling:    +$delta%"
    fi
fi

echo ""
echo "Results saved to: $RESULTS_DIR"
echo ""
echo "For your paper:"
echo "  • Table 2: Copy these results to your ablation table"
echo "  • Figure: Create bar chart comparing all configurations"
echo "  • Text: Explain why each component matters"
echo ""
echo "Next steps:"
echo "  1. Analyze the ablation results"
echo "  2. Create ablation table for paper"
echo "  3. Write ablation section explaining each component's role"
echo ""
