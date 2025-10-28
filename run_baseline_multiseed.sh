#!/bin/bash
################################################################################
# CRITICAL: DER++ Baseline Multi-Seed (CIFAR-100)
# Need this for statistical comparison with CausalDER
################################################################################

RESULTS_DIR="validation/results/baseline_multiseed_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RESULTS_DIR"

echo "╔════════════════════════════════════════════════════════════════════╗"
echo "║      DER++ BASELINE MULTI-SEED - CIFAR-100 (Statistical Test)     ║"
echo "╚════════════════════════════════════════════════════════════════════╝"
echo ""
echo "Running 5 seeds for statistical comparison with CausalDER"
echo "CausalDER result: 72.01 ± 0.56%"
echo ""
echo "Results will be saved to: $RESULTS_DIR"
echo ""

# Common parameters (match CausalDER experiments exactly)
COMMON_PARAMS="--model derpp \
--dataset seq-cifar100 \
--buffer_size 500 \
--alpha 0.3 \
--beta 0.5 \
--n_epochs 5 \
--batch_size 32 \
--lr 0.03 \
--minibatch_size 32 \
--lr_scheduler multisteplr \
--lr_milestones 3 4 \
--sched_multistep_lr_gamma 0.2"

# Run 5 seeds
for seed in 1 2 3 4 5; do
    echo "════════════════════════════════════════════════════════════════════"
    echo "  Running DER++ Baseline - Seed $seed/5"
    echo "  Expected: ~73-74% based on previous runs"
    echo "════════════════════════════════════════════════════════════════════"
    
    python3 mammoth/utils/main.py $COMMON_PARAMS \
        --seed $seed \
        2>&1 | tee "$RESULTS_DIR/baseline_seed${seed}.log"
    
    # Extract accuracy immediately
    acc=$(grep "Accuracy for 10 task" "$RESULTS_DIR/baseline_seed${seed}.log" | grep "Task-IL" | tail -1 | grep -oE '[0-9]+\.[0-9]+' | head -1)
    
    if [ -n "$acc" ]; then
        echo "  ✅ Seed $seed completed: Task-IL = $acc%"
    else
        echo "  ❌ Seed $seed FAILED"
    fi
    
    echo ""
done

# Summarize results
echo "╔════════════════════════════════════════════════════════════════════╗"
echo "║                    BASELINE RESULTS SUMMARY                        ║"
echo "╚════════════════════════════════════════════════════════════════════╝"
echo ""

echo "DER++ Baseline (5 seeds):"
echo "───────────────────────────────────────────────────────────────────"
for seed in 1 2 3 4 5; do
    if [ -f "$RESULTS_DIR/baseline_seed${seed}.log" ]; then
        acc=$(grep "Accuracy for 10 task" "$RESULTS_DIR/baseline_seed${seed}.log" | grep "Task-IL" | tail -1 | grep -oE '[0-9]+\.[0-9]+' | head -1)
        if [ -n "$acc" ]; then
            echo "  Seed $seed: $acc%"
        fi
    fi
done
echo ""

echo "CausalDER (already completed):"
echo "───────────────────────────────────────────────────────────────────"
echo "  Mean: 72.01 ± 0.56%"
echo "  Seeds: [72.11, 71.66, 72.21, 71.31, 72.77]"
echo ""

echo "Next step: Run statistical t-test to compare baseline vs causal"
echo "Results saved to: $RESULTS_DIR"
echo ""
