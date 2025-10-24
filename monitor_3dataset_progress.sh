#!/bin/bash

################################################################################
# Monitor 3-Dataset Validation Progress
################################################################################

LOGDIR="validation/results/quick_validation_3datasets"

echo "═══════════════════════════════════════════════════════════════════"
echo "📊 3-Dataset Validation Progress Monitor"
echo "═══════════════════════════════════════════════════════════════════"
echo ""

# Function to get latest accuracy from log
get_latest_accuracy() {
    local logfile=$1
    if [ -f "$logfile" ]; then
        # Get last line with "Accuracy for"
        local acc=$(grep "Accuracy for" "$logfile" | tail -1 | grep -oE '[0-9]+\.[0-9]+' | tail -1 || echo "")
        if [ -n "$acc" ]; then
            echo "${acc}%"
        else
            # Count completed epochs/tasks
            local completed=$(grep -c "Task.*completed" "$logfile" 2>/dev/null || echo "0")
            echo "Running (${completed} tasks done)"
        fi
    else
        echo "Not started"
    fi
}

# CIFAR-10 Status
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🔍 CIFAR-10 (5 tasks, 5 epochs each)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "  DER++ Baseline:  $(get_latest_accuracy ${LOGDIR}/cifar10/baseline_seed1.log)"
echo "  CausalDER:       $(get_latest_accuracy ${LOGDIR}/cifar10/causal_seed1.log)"
echo ""

# MNIST Status
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🔍 MNIST (5 tasks, 5 epochs each) - Multi-Seed"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
for seed in 1 2 3 4 5; do
    echo "  Seed ${seed}:  $(get_latest_accuracy ${LOGDIR}/mnist/seed_${seed}.log)"
done
echo ""

# Overall Progress
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📈 Overall Progress"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Count completed runs
COMPLETED=0
[ -f "${LOGDIR}/cifar10/baseline_seed1.log" ] && grep -q "Accuracy for 5 task" "${LOGDIR}/cifar10/baseline_seed1.log" && ((COMPLETED++))
[ -f "${LOGDIR}/cifar10/causal_seed1.log" ] && grep -q "Accuracy for 5 task" "${LOGDIR}/cifar10/causal_seed1.log" && ((COMPLETED++))
for seed in 1 2 3 4 5; do
    [ -f "${LOGDIR}/mnist/seed_${seed}.log" ] && grep -q "Accuracy for 5 task" "${LOGDIR}/mnist/seed_${seed}.log" && ((COMPLETED++))
done

echo "  Completed: ${COMPLETED}/7 experiments"
echo ""

if [ $COMPLETED -eq 7 ]; then
    echo "✅ ALL EXPERIMENTS COMPLETE!"
    echo ""
    echo "Run this to see final results:"
    echo "  ./show_3dataset_results.sh"
else
    echo "⏳ Still running... Check again in a few minutes."
    echo ""
    echo "To monitor in real-time:"
    echo "  watch -n 30 ./monitor_3dataset_progress.sh"
fi

echo ""
echo "═══════════════════════════════════════════════════════════════════"
