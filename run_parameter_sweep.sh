#!/bin/bash

################################################################################
# Systematic Parameter Sweep: Phase 3 Causal Graph Learning
################################################################################
#
# Goal: Find optimal parameters to improve Phase 3 performance
# 
# Scientific Controls:
#   - Same test parameters as baseline (buffer=500, epochs=5, lr=0.03, etc.)
#   - Same seed (seed=1) for reproducibility
#   - Only change Phase 3-specific parameters
#
# Current Results:
#   - Baseline (no graph):  70.19% Task-IL
#   - Phase 3 (default):    62.30% Task-IL (-7.89%)
#
# Parameters to Test:
#   1. Cache size: [0, 50, 100, 200]
#   2. Sparsification: [0.3, 0.5, 0.7, 0.9, None]
#
# Author: Symbio AI
# Date: October 23, 2025
################################################################################

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo "========================================================================"
echo "🔬 SYSTEMATIC PARAMETER SWEEP - PHASE 3 CAUSAL GRAPH LEARNING"
echo "========================================================================"
echo ""
echo "Scientific Method: Controlled Experiments"
echo ""
echo "FIXED Parameters (same as baseline):"
echo "  --buffer_size 500"
echo "  --alpha 0.3"
echo "  --beta 0.5"
echo "  --n_epochs 5"
echo "  --batch_size 32"
echo "  --minibatch_size 32"
echo "  --lr 0.03"
echo "  --optim_mom 0.0"
echo "  --optim_wd 0.0"
echo "  --seed 1"
echo ""
echo "VARIABLE Parameters (Phase 3 only):"
echo "  1. Feature cache size: [0, 50, 100, 200]"
echo "  2. Graph sparsification: [0.3, 0.5, 0.7, 0.9, None]"
echo ""
echo "Current Results:"
echo "  Baseline (no graph):  70.19% Task-IL"
echo "  Phase 3 (default):    62.30% Task-IL"
echo "  Gap to close:         7.89%"
echo ""
echo "========================================================================"
echo ""

# Configuration
RESULTS_DIR="validation/results/parameter_sweep"
mkdir -p "${RESULTS_DIR}"

# Backup original files
echo "Creating backups..."
cp training/causal_der_v2.py training/causal_der_v2.py.backup_sweep
cp training/causal_inference.py training/causal_inference.py.backup_sweep
echo "✅ Backups created"
echo ""

# Results tracking
RESULTS_FILE="${RESULTS_DIR}/sweep_results.csv"
echo "experiment,cache_size,sparsification,task_il,class_il,improvement_vs_phase3,gap_vs_baseline" > "${RESULTS_FILE}"

# Baseline references
BASELINE_TASK_IL=70.19
PHASE3_DEFAULT_TASK_IL=62.30

# Fixed command line arguments (same as baseline)
FIXED_ARGS="--model causal-der --dataset seq-cifar100 \
--buffer_size 500 --alpha 0.3 --beta 0.5 --n_epochs 5 \
--batch_size 32 --minibatch_size 32 \
--lr 0.03 --optim_mom 0.0 --optim_wd 0.0 \
--enable_causal_graph_learning 1 \
--seed 1"

################################################################################
# Experiment 1: Cache Size Sweep (with default sparsification=0.7)
################################################################################

echo ""
echo "========================================================================"
echo "📊 EXPERIMENT SET 1: CACHE SIZE SWEEP"
echo "========================================================================"
echo ""
echo "Hypothesis: Large cache (200) interferes with buffer"
echo "Evidence:   Early tasks show -10% to -14% drop"
echo ""
echo "Testing cache sizes: 0, 50, 100, 200"
echo "Sparsification: 0.7 (default)"
echo ""
echo "========================================================================"
echo ""

CACHE_SIZES=(0 50 100 200)

for CACHE_SIZE in "${CACHE_SIZES[@]}"; do
    echo ""
    echo "--------------------------------------------------------------------"
    echo -e "${BLUE}Testing Cache Size: ${CACHE_SIZE}${NC}"
    echo "--------------------------------------------------------------------"
    
    # Restore original files
    cp training/causal_der_v2.py.backup_sweep training/causal_der_v2.py
    cp training/causal_inference.py.backup_sweep training/causal_inference.py
    
    # Modify cache size
    if [ "$CACHE_SIZE" -eq 0 ]; then
        echo "  Disabling feature cache completely..."
        # Comment out cache storage
        sed -i '' 's/if len(cache\['"'"'features'"'"'\]) < 200:/if False:  # DISABLED/' training/causal_der_v2.py
    else
        echo "  Setting cache size to ${CACHE_SIZE}..."
        sed -i '' "s/if len(cache\['"'"'features'"'"'\]) < 200:/if len(cache['features']) < ${CACHE_SIZE}:/" training/causal_der_v2.py
        sed -i '' "s/Store up to 200 samples per task/Store up to ${CACHE_SIZE} samples per task/" training/causal_der_v2.py
    fi
    
    # Keep sparsification at default (0.7)
    echo "  Sparsification: 0.7 (default)"
    
    # Run experiment
    LOG_FILE="${RESULTS_DIR}/cache${CACHE_SIZE}_spars0.7_seed1.log"
    
    echo ""
    echo "  Running experiment (ETA: 52 minutes)..."
    echo "  Log: ${LOG_FILE}"
    echo ""
    
    python3 mammoth/utils/main.py ${FIXED_ARGS} 2>&1 | tee "${LOG_FILE}"
    
    # Extract results
    TASK_IL=$(grep "Task-IL" "${LOG_FILE}" | tail -1 | grep -oE "[0-9]+\.[0-9]+" | head -1)
    CLASS_IL=$(grep "Class-IL" "${LOG_FILE}" | tail -1 | grep -oE "[0-9]+\.[0-9]+" | tail -1)
    
    IMPROVEMENT=$(python3 -c "print(f'{${TASK_IL} - ${PHASE3_DEFAULT_TASK_IL}:.2f}')")
    GAP=$(python3 -c "print(f'{${BASELINE_TASK_IL} - ${TASK_IL}:.2f}')")
    
    # Save results
    echo "cache${CACHE_SIZE}_spars0.7,${CACHE_SIZE},0.7,${TASK_IL},${CLASS_IL},${IMPROVEMENT},${GAP}" >> "${RESULTS_FILE}"
    
    echo ""
    echo "  Results:"
    echo "    Task-IL:              ${TASK_IL}%"
    echo "    Class-IL:             ${CLASS_IL}%"
    echo "    vs Phase 3 default:   ${IMPROVEMENT:+}${IMPROVEMENT}%"
    echo "    Gap to baseline:      ${GAP}%"
    echo ""
    
    # Quick analysis
    if (( $(echo "${TASK_IL} >= ${BASELINE_TASK_IL}" | bc -l) )); then
        echo -e "  ${GREEN}✅ SUCCESS: Beats baseline!${NC}"
    elif (( $(echo "${TASK_IL} > ${PHASE3_DEFAULT_TASK_IL} + 2" | bc -l) )); then
        echo -e "  ${YELLOW}⚠️  GOOD: Significant improvement${NC}"
    elif (( $(echo "${TASK_IL} > ${PHASE3_DEFAULT_TASK_IL}" | bc -l) )); then
        echo -e "  ${YELLOW}🤔 MARGINAL: Small improvement${NC}"
    else
        echo -e "  ${RED}❌ NO IMPROVEMENT${NC}"
    fi
    
    echo ""
    sleep 2
done

################################################################################
# Experiment 2: Sparsification Sweep (with best cache size from Exp 1)
################################################################################

echo ""
echo "========================================================================"
echo "📊 EXPERIMENT SET 2: SPARSIFICATION SWEEP"
echo "========================================================================"
echo ""
echo "Hypothesis: Too much sparsification (0.7) discards useful signals"
echo "Evidence:   Max edge strength only 0.475 (all weak)"
echo ""

# Determine best cache size from Experiment 1
echo "Analyzing Experiment 1 results to find best cache size..."
BEST_CACHE_SIZE=$(tail -n +2 "${RESULTS_FILE}" | grep "spars0.7" | sort -t',' -k4 -nr | head -1 | cut -d',' -f2)
BEST_CACHE_TASK_IL=$(tail -n +2 "${RESULTS_FILE}" | grep "spars0.7" | sort -t',' -k4 -nr | head -1 | cut -d',' -f4)

echo ""
echo "Best cache size from Experiment 1: ${BEST_CACHE_SIZE}"
echo "  Task-IL: ${BEST_CACHE_TASK_IL}%"
echo ""
echo "Testing sparsification quantiles: 0.3, 0.5, 0.7, 0.9, None"
echo "Cache size: ${BEST_CACHE_SIZE} (best from Exp 1)"
echo ""
echo "========================================================================"
echo ""

SPARSIFICATIONS=("0.3" "0.5" "0.7" "0.9" "none")

for SPARS in "${SPARSIFICATIONS[@]}"; do
    echo ""
    echo "--------------------------------------------------------------------"
    echo -e "${BLUE}Testing Sparsification: ${SPARS}${NC}"
    echo "--------------------------------------------------------------------"
    
    # Restore original files
    cp training/causal_der_v2.py.backup_sweep training/causal_der_v2.py
    cp training/causal_inference.py.backup_sweep training/causal_inference.py
    
    # Set cache size to best from Experiment 1
    if [ "$BEST_CACHE_SIZE" -eq 0 ]; then
        sed -i '' 's/if len(cache\['"'"'features'"'"'\]) < 200:/if False:  # DISABLED/' training/causal_der_v2.py
    else
        sed -i '' "s/if len(cache\['"'"'features'"'"'\]) < 200:/if len(cache['features']) < ${BEST_CACHE_SIZE}:/" training/causal_der_v2.py
    fi
    
    # Modify sparsification
    if [ "$SPARS" = "none" ]; then
        echo "  Disabling sparsification (keep all edges)..."
        sed -i '' 's/threshold = self.task_graph.abs().quantile(0.7)/# threshold = self.task_graph.abs().quantile(0.7)/' training/causal_inference.py
        sed -i '' 's/self.task_graph\[self.task_graph.abs() < threshold\] = 0/# self.task_graph[self.task_graph.abs() < threshold] = 0/' training/causal_inference.py
    else
        echo "  Setting sparsification quantile to ${SPARS}..."
        sed -i '' "s/threshold = self.task_graph.abs().quantile(0.7)/threshold = self.task_graph.abs().quantile(${SPARS})/" training/causal_inference.py
    fi
    
    echo "  Cache size: ${BEST_CACHE_SIZE}"
    echo "  Sparsification: ${SPARS}"
    
    # Run experiment
    LOG_FILE="${RESULTS_DIR}/cache${BEST_CACHE_SIZE}_spars${SPARS}_seed1.log"
    
    echo ""
    echo "  Running experiment (ETA: 52 minutes)..."
    echo "  Log: ${LOG_FILE}"
    echo ""
    
    python3 mammoth/utils/main.py ${FIXED_ARGS} 2>&1 | tee "${LOG_FILE}"
    
    # Extract results
    TASK_IL=$(grep "Task-IL" "${LOG_FILE}" | tail -1 | grep -oE "[0-9]+\.[0-9]+" | head -1)
    CLASS_IL=$(grep "Class-IL" "${LOG_FILE}" | tail -1 | grep -oE "[0-9]+\.[0-9]+" | tail -1)
    
    IMPROVEMENT=$(python3 -c "print(f'{${TASK_IL} - ${PHASE3_DEFAULT_TASK_IL}:.2f}')")
    GAP=$(python3 -c "print(f'{${BASELINE_TASK_IL} - ${TASK_IL}:.2f}')")
    
    # Save results
    echo "cache${BEST_CACHE_SIZE}_spars${SPARS},${BEST_CACHE_SIZE},${SPARS},${TASK_IL},${CLASS_IL},${IMPROVEMENT},${GAP}" >> "${RESULTS_FILE}"
    
    echo ""
    echo "  Results:"
    echo "    Task-IL:              ${TASK_IL}%"
    echo "    Class-IL:             ${CLASS_IL}%"
    echo "    vs Phase 3 default:   ${IMPROVEMENT:+}${IMPROVEMENT}%"
    echo "    Gap to baseline:      ${GAP}%"
    echo ""
    
    # Quick analysis
    if (( $(echo "${TASK_IL} >= ${BASELINE_TASK_IL}" | bc -l) )); then
        echo -e "  ${GREEN}✅ SUCCESS: Beats baseline!${NC}"
    elif (( $(echo "${TASK_IL} > ${PHASE3_DEFAULT_TASK_IL} + 2" | bc -l) )); then
        echo -e "  ${YELLOW}⚠️  GOOD: Significant improvement${NC}"
    elif (( $(echo "${TASK_IL} > ${PHASE3_DEFAULT_TASK_IL}" | bc -l) )); then
        echo -e "  ${YELLOW}🤔 MARGINAL: Small improvement${NC}"
    else
        echo -e "  ${RED}❌ NO IMPROVEMENT${NC}"
    fi
    
    echo ""
    sleep 2
done

################################################################################
# Final Analysis
################################################################################

echo ""
echo "========================================================================"
echo "📊 FINAL ANALYSIS - ALL EXPERIMENTS"
echo "========================================================================"
echo ""

# Restore original files
cp training/causal_der_v2.py.backup_sweep training/causal_der_v2.py
cp training/causal_inference.py.backup_sweep training/causal_inference.py
echo "✅ Original files restored"
echo ""

# Display all results sorted by Task-IL
echo "All Results (sorted by Task-IL):"
echo "----------------------------------------------------------------"
printf "%-25s %-12s %-15s %-10s %-10s %-15s\n" "Experiment" "Cache" "Sparsify" "Task-IL" "Class-IL" "vs Default"
echo "----------------------------------------------------------------"

tail -n +2 "${RESULTS_FILE}" | sort -t',' -k4 -nr | while IFS=',' read -r exp cache spars task_il class_il improve gap; do
    printf "%-25s %-12s %-15s %-10s %-10s %+14s%%\n" "$exp" "$cache" "$spars" "${task_il}%" "${class_il}%" "$improve"
done

echo "----------------------------------------------------------------"
echo ""

# Find best configuration
BEST_CONFIG=$(tail -n +2 "${RESULTS_FILE}" | sort -t',' -k4 -nr | head -1)
BEST_EXP=$(echo "$BEST_CONFIG" | cut -d',' -f1)
BEST_CACHE=$(echo "$BEST_CONFIG" | cut -d',' -f2)
BEST_SPARS=$(echo "$BEST_CONFIG" | cut -d',' -f3)
BEST_TASK_IL=$(echo "$BEST_CONFIG" | cut -d',' -f4)
BEST_IMPROVE=$(echo "$BEST_CONFIG" | cut -d',' -f6)
BEST_GAP=$(echo "$BEST_CONFIG" | cut -d',' -f7)

echo ""
echo "🏆 BEST CONFIGURATION:"
echo "  Experiment:           ${BEST_EXP}"
echo "  Cache size:           ${BEST_CACHE}"
echo "  Sparsification:       ${BEST_SPARS}"
echo "  Task-IL:              ${BEST_TASK_IL}%"
echo "  Improvement:          ${BEST_IMPROVE:+}${BEST_IMPROVE}%"
echo "  Gap to baseline:      ${BEST_GAP}%"
echo ""

# Decision tree
python3 << EOF
baseline = ${BASELINE_TASK_IL}
phase3_default = ${PHASE3_DEFAULT_TASK_IL}
best = float("${BEST_TASK_IL}")

print("="*70)
print("📋 DECISION & RECOMMENDATIONS")
print("="*70)
print()

if best >= baseline + 1.0:
    print("✅ EXCELLENT SUCCESS: Parameter tuning WORKS!")
    print()
    print(f"   Best result: {best:.2f}% Task-IL")
    print(f"   Beats baseline by: +{best - baseline:.2f}%")
    print(f"   Recovery from default: +{best - phase3_default:.2f}%")
    print()
    print("📌 NEXT STEPS:")
    print(f"   1. Update default: cache={BEST_CACHE}, sparsification={BEST_SPARS}")
    print("   2. Run multi-seed validation: ./run_multiseed.sh graph_learning 5")
    print("   3. Pursue Option A (causal paper) - we have positive results!")
    print("   4. Test gradient-based causality for further gains")
    
elif best >= baseline:
    print("✅ SUCCESS: Matched/exceeded baseline!")
    print()
    print(f"   Best result: {best:.2f}% Task-IL")
    print(f"   vs Baseline: {best - baseline:+.2f}%")
    print(f"   Recovery: +{best - phase3_default:.2f}%")
    print()
    print("📌 NEXT STEPS:")
    print(f"   1. Update default: cache={BEST_CACHE}, sparsification={BEST_SPARS}")
    print("   2. Graph learning doesn't hurt (neutral)")
    print("   3. Option A possible if multi-seed confirms")
    print("   4. Consider Option C with balanced view")
    
elif best > phase3_default + 3.0:
    print("⚠️  GOOD PROGRESS: Significant improvement, but still below baseline")
    print()
    print(f"   Best result: {best:.2f}% Task-IL")
    print(f"   Improvement: +{best - phase3_default:.2f}% from default")
    print(f"   Gap to baseline: -{baseline - best:.2f}%")
    print()
    print("📌 NEXT STEPS:")
    print("   1. Try gradient-based causal estimation")
    print("   2. Test on different datasets (Split-MNIST)")
    print("   3. Consider hybrid: use only when helpful")
    print("   4. Option C likely (honest analysis)")
    
else:
    print("❌ NO SUFFICIENT IMPROVEMENT: Parameter tuning didn't solve the problem")
    print()
    print(f"   Best result: {best:.2f}% Task-IL")
    print(f"   Improvement: +{best - phase3_default:.2f}% from default")
    print(f"   Still {baseline - best:.2f}% below baseline")
    print()
    print("📌 NEXT STEPS:")
    print("   1. Problem is algorithmic, not just parameters")
    print("   2. Pursue Option C (analysis paper)")
    print("   3. Focus: 'When Does Causal Inference Help CL?'")
    print("   4. Honest assessment more valuable than negative results")

print()
print("="*70)
EOF

echo ""
echo "Results saved to: ${RESULTS_FILE}"
echo ""
echo "Backups available:"
echo "  training/causal_der_v2.py.backup_sweep"
echo "  training/causal_inference.py.backup_sweep"
echo ""
echo "Parameter sweep complete! 🎉"
echo ""

