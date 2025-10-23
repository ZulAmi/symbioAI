#!/bin/bash

################################################################################
# Multi-Seed Experiment Runner
################################################################################
# 
# Runs experiments with multiple random seeds to compute mean ± std.
# Critical for statistical significance in publication.
#
# Usage:
#   ./run_multiseed.sh <config_name> [num_seeds]
#
# Example:
#   ./run_multiseed.sh baseline 5
#   ./run_multiseed.sh graph_learning 3
#
# Author: Symbio AI
# Date: October 22, 2025
################################################################################

set -e  # Exit on error

# Configuration
CONFIG_NAME="${1:-baseline}"
NUM_SEEDS="${2:-5}"
OUTPUT_DIR="validation/results/multiseed"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
EXPERIMENT_DIR="${OUTPUT_DIR}/${CONFIG_NAME}_${TIMESTAMP}"

# Create output directory
mkdir -p "${EXPERIMENT_DIR}"

echo "================================"
echo "🔬 Multi-Seed Experiment Runner"
echo "================================"
echo "Configuration: ${CONFIG_NAME}"
echo "Number of seeds: ${NUM_SEEDS}"
echo "Output directory: ${EXPERIMENT_DIR}"
echo ""

# Common parameters for all experiments (NEW: derpp-causal with proper LR schedule)
COMMON_ARGS="--model derpp-causal --dataset seq-cifar100 \
--buffer_size 500 --alpha 0.3 --beta 0.5 --n_epochs 5 \
--batch_size 32 --minibatch_size 32 --lr 0.03 \
--lr_scheduler multisteplr --lr_milestones 3 4 --sched_multistep_lr_gamma 0.2"

# Configuration-specific parameters
case "${CONFIG_NAME}" in
  baseline)
    echo "📋 Running BASELINE configuration (Official DER++, causal OFF)"
    SPECIFIC_ARGS="--enable_causal_graph_learning 0 --use_causal_sampling 0"
    ;;
  
  causal)
    echo "📋 Running CAUSAL configuration (Phase 2: causal graph learning ON)"
    SPECIFIC_ARGS="--enable_causal_graph_learning 1 --use_causal_sampling 1 \
--num_tasks 10 --feature_dim 512 --causal_cache_size 200"
    ;;
  
  quickwins)
    echo "📋 Running QUICK WINS configuration (Phase 2 with optimizations)"
    SPECIFIC_ARGS="--enable_causal_graph_learning 1 --use_causal_sampling 1 \
--num_tasks 10 --feature_dim 512 --causal_cache_size 200"
    echo "   (Warm start + smoother importance + adaptive sparsification)"
    ;;
  
  *)
    echo "❌ Unknown configuration: ${CONFIG_NAME}"
    echo "Available: baseline, causal, quickwins"
    exit 1
    ;;
esac

# Run experiments for each seed
echo ""
echo "Starting experiments..."
echo "------------------------"

SEED_RESULTS=()
SEED_LOGS=()

for seed in $(seq 1 ${NUM_SEEDS}); do
  echo ""
  echo "🌱 Running seed ${seed}/${NUM_SEEDS}..."
  
  LOG_FILE="${EXPERIMENT_DIR}/seed_${seed}.log"
  SEED_LOGS+=("${LOG_FILE}")
  
  # Run experiment
  python3 mammoth/utils/main.py ${COMMON_ARGS} ${SPECIFIC_ARGS} --seed ${seed} \
    > "${LOG_FILE}" 2>&1
  
  # Extract final Task-IL accuracy
  FINAL_ACC=$(grep -oP "Task 9: Task-IL Acc: \K[0-9.]+" "${LOG_FILE}" | tail -1 || echo "0.0")
  SEED_RESULTS+=("${FINAL_ACC}")
  
  echo "   ✅ Seed ${seed} complete: ${FINAL_ACC}% Task-IL"
done

echo ""
echo "========================"
echo "📊 Results Summary"
echo "========================"

# Compute statistics using Python
RESULTS_FILE="${EXPERIMENT_DIR}/summary.txt"
RESULTS_JSON="${EXPERIMENT_DIR}/summary.json"

cat > /tmp/compute_stats.py << 'EOF'
import sys
import json
import numpy as np

# Read results from command line args
results = [float(x) for x in sys.argv[1:]]

mean = np.mean(results)
std = np.std(results)
median = np.median(results)
min_val = np.min(results)
max_val = np.max(results)

# Print summary
print(f"Configuration: {sys.argv[0]}")
print(f"Number of seeds: {len(results)}")
print(f"")
print(f"Final Task-IL Accuracy:")
print(f"  Mean:   {mean:.2f}%")
print(f"  Std:    {std:.2f}%")
print(f"  Median: {median:.2f}%")
print(f"  Range:  [{min_val:.2f}%, {max_val:.2f}%]")
print(f"")
print(f"Reporting Format: {mean:.2f} ± {std:.2f}%")
print(f"")
print(f"Individual Seeds:")
for i, acc in enumerate(results, 1):
    print(f"  Seed {i}: {acc:.2f}%")

# Save JSON
summary = {
    'configuration': sys.argv[0],
    'num_seeds': len(results),
    'mean': float(mean),
    'std': float(std),
    'median': float(median),
    'min': float(min_val),
    'max': float(max_val),
    'individual_results': results
}

with open(sys.argv[-1], 'w') as f:
    json.dump(summary, f, indent=2)
EOF

python3 /tmp/compute_stats.py "${CONFIG_NAME}" ${SEED_RESULTS[@]} "${RESULTS_JSON}" | tee "${RESULTS_FILE}"

echo ""
echo "========================"
echo "✅ Experiment Complete!"
echo "========================"
echo "Results saved to: ${EXPERIMENT_DIR}"
echo "Summary: ${RESULTS_FILE}"
echo "JSON: ${RESULTS_JSON}"
echo "Logs: ${EXPERIMENT_DIR}/seed_*.log"
echo ""

# Create plots if visualization module exists
if [ -f "visualization/publication_figures.py" ]; then
  echo "📊 Generating visualizations..."
  
  python3 << EOF
import sys
sys.path.append('.')
from visualization.publication_figures import load_metrics_from_log, plot_accuracy_vs_task
import json
import numpy as np

# Load results from all seeds
all_task_il = []
all_class_il = []

for log_file in ${SEED_LOGS[@]}:
    try:
        metrics = load_metrics_from_log(log_file)
        all_task_il.append(metrics['task_il'])
        all_class_il.append(metrics['class_il'])
    except Exception as e:
        print(f"Warning: Could not load {log_file}: {e}")

if all_task_il:
    # Pad to same length
    max_len = max(len(x) for x in all_task_il)
    task_il_padded = [x + [np.nan]*(max_len-len(x)) for x in all_task_il]
    class_il_padded = [x + [np.nan]*(max_len-len(x)) for x in all_class_il]
    
    # Compute mean and std
    task_il_mean = np.nanmean(task_il_padded, axis=0)
    class_il_mean = np.nanmean(class_il_padded, axis=0)
    
    # Plot
    results_dict = {
        'task_il': task_il_mean.tolist(),
        'class_il': class_il_mean.tolist()
    }
    
    plot_accuracy_vs_task(
        results_dict,
        output_path='${EXPERIMENT_DIR}/accuracy_plot.pdf',
        title='Accuracy: ${CONFIG_NAME} (${NUM_SEEDS} seeds)'
    )
    
    print(f"✅ Plot saved to ${EXPERIMENT_DIR}/accuracy_plot.pdf")
else:
    print("⚠️  No results to plot")
EOF

fi

echo ""
echo "🎉 All done! Use results for publication."
