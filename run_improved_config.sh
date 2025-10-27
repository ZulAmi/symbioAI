#!/bin/bash
################################################################################
# CausalDER - IMPROVED Configuration Test
# 
# Testing all recommended improvements at once:
# ✅ Longer training: 5 → 10 epochs
# ✅ Larger buffer: 500 → 1000 samples
# ✅ Larger causal cache: 200 → 500 samples
# ✅ Weight decay: 0.0 → 0.0001 (regularization)
# ✅ Momentum: 0.0 → 0.9 (smoother optimization)
# ✅ Higher alpha: 0.3 → 0.5 (stronger distillation)
# ✅ Higher causal blend: 0.6 → 0.7 (more causal sampling)
# ✅ Larger batch: 32 → 64 (more stable gradients)
#
# Expected: 72.01% → 75-78% Task-IL accuracy
################################################################################

RESULTS_DIR="validation/results/improved_config_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RESULTS_DIR"

echo "╔════════════════════════════════════════════════════════════════════╗"
echo "║         CausalDER - IMPROVED CONFIGURATION TEST                    ║"
echo "╚════════════════════════════════════════════════════════════════════╝"
echo ""
echo "Baseline (old config): 72.01 ± 0.56% Task-IL"
echo "Expected (new config): 75-78% Task-IL (+3-6% improvement)"
echo ""
echo "Improvements applied:"
echo "  • Epochs: 5 → 10"
echo "  • Buffer: 500 → 1000"
echo "  • Causal cache: 200 → 500"
echo "  • Weight decay: 0.0 → 0.0001"
echo "  • Momentum: 0.0 → 0.9"
echo "  • Alpha: 0.3 → 0.5"
echo "  • Causal blend: 0.6 → 0.7"
echo "  • Batch size: 32 → 64"
echo ""
echo "Results will be saved to: $RESULTS_DIR"
echo ""

# IMPROVED configuration
IMPROVED_PARAMS="--model derpp-causal \
--dataset seq-cifar100 \
--enable_causal_graph_learning 1 \
--use_causal_sampling 1 \
--buffer_size 1000 \
--causal_cache_size 500 \
--n_epochs 10 \
--batch_size 64 \
--minibatch_size 64 \
--lr 0.03 \
--lr_scheduler multisteplr \
--lr_milestones 5 8 \
--sched_multistep_lr_gamma 0.2 \
--alpha 0.5 \
--beta 0.5 \
--optim_wd 0.0001 \
--optim_mom 0.9 \
--causal_blend_ratio 0.7 \
--temperature 2.0 \
--feature_dim 512"

# Run 3 seeds for validation
for seed in 1 2 3; do
    echo "════════════════════════════════════════════════════════════════════"
    echo "  Running IMPROVED CausalDER - Seed $seed/3"
    echo "  Expected: ~75-78% Task-IL"
    echo "  Estimated time: ~1.5 hours per seed"
    echo "════════════════════════════════════════════════════════════════════"
    
    python3 mammoth/utils/main.py $IMPROVED_PARAMS \
        --seed $seed \
        2>&1 | tee "$RESULTS_DIR/improved_seed${seed}.log"
    
    # Extract accuracy immediately
    task_il=$(grep "Accuracy for 10 task" "$RESULTS_DIR/improved_seed${seed}.log" | grep "Task-IL" | tail -1 | grep -oE '[0-9]+\.[0-9]+' | head -1)
    class_il=$(grep "Accuracy for 10 task" "$RESULTS_DIR/improved_seed${seed}.log" | grep "Class-IL" | tail -1 | grep -oE '[0-9]+\.[0-9]+' | head -1)
    
    if [ -n "$task_il" ]; then
        echo ""
        echo "  ✅ Seed $seed completed!"
        echo "     Task-IL:  $task_il%"
        echo "     Class-IL: $class_il%"
        
        # Calculate improvement
        improvement=$(python3 -c "print(f'{float($task_il) - 72.01:.2f}')" 2>/dev/null || echo "N/A")
        echo "     Improvement vs baseline: +$improvement%"
    else
        echo "  ❌ Seed $seed FAILED"
    fi
    
    echo ""
done

# Summarize results
echo "╔════════════════════════════════════════════════════════════════════╗"
echo "║                    IMPROVEMENT SUMMARY                             ║"
echo "╚════════════════════════════════════════════════════════════════════╝"
echo ""

echo "OLD Configuration (72.01% baseline):"
echo "───────────────────────────────────────────────────────────────────"
echo "  Seeds: [72.11, 71.66, 72.21, 71.31, 72.77]"
echo "  Mean:  72.01 ± 0.56%"
echo ""

echo "NEW Improved Configuration:"
echo "───────────────────────────────────────────────────────────────────"
for seed in 1 2 3; do
    if [ -f "$RESULTS_DIR/improved_seed${seed}.log" ]; then
        acc=$(grep "Accuracy for 10 task" "$RESULTS_DIR/improved_seed${seed}.log" | grep "Task-IL" | tail -1 | grep -oE '[0-9]+\.[0-9]+' | head -1)
        if [ -n "$acc" ]; then
            echo "  Seed $seed: $acc%"
        fi
    fi
done

# Calculate mean if all completed
echo ""
echo "Calculating statistics..."
python3 - <<EOF
import re
import numpy as np
from pathlib import Path

results_dir = Path("$RESULTS_DIR")
accuracies = []

for seed in [1, 2, 3]:
    log_file = results_dir / f"improved_seed{seed}.log"
    if log_file.exists():
        content = log_file.read_text()
        matches = re.findall(r'Accuracy for 10 task.*?\[Task-IL\]:\s*([\d.]+)', content)
        if matches:
            accuracies.append(float(matches[-1]))

if accuracies:
    mean = np.mean(accuracies)
    std = np.std(accuracies, ddof=1) if len(accuracies) > 1 else 0
    
    print(f"  Mean: {mean:.2f} ± {std:.2f}%")
    print(f"  Range: [{min(accuracies):.2f}, {max(accuracies):.2f}]%")
    print()
    
    improvement = mean - 72.01
    print(f"  🎯 IMPROVEMENT: +{improvement:.2f}%")
    print()
    
    if improvement >= 3.0:
        print("  ✅ SUCCESS! Significant improvement achieved!")
        print("     This is publication-worthy progress.")
    elif improvement >= 1.0:
        print("  ✅ GOOD! Moderate improvement achieved.")
    elif improvement >= 0:
        print("  ⚠️  MARGINAL: Small improvement. May need more tuning.")
    else:
        print("  ❌ REGRESSION: Performance decreased. Check logs.")
else:
    print("  No results available yet.")
EOF

echo ""
echo "Results saved to: $RESULTS_DIR"
echo ""
echo "Next steps:"
echo "  1. If improvement ≥3%, run 2 more seeds for 5-seed validation"
echo "  2. Ablation study: Test each improvement individually"
echo "  3. Update paper with improved results"
echo "  4. Run baseline DER++ with same improved hyperparameters"
echo ""
