#!/bin/bash

################################################################################
# Quick Single Parameter Test
################################################################################
#
# Fast validation of single parameter change before full sweep
# Uses SAME test parameters as baseline for valid comparison
#
# Usage:
#   ./quick_param_test.sh cache <size>       # Test cache size
#   ./quick_param_test.sh sparsify <value>   # Test sparsification
#
# Examples:
#   ./quick_param_test.sh cache 50           # Test cache=50
#   ./quick_param_test.sh sparsify 0.5       # Test quantile=0.5
#   ./quick_param_test.sh sparsify none      # Test no sparsification
#
# Author: Symbio AI
# Date: October 23, 2025
################################################################################

set -e

if [ "$#" -lt 2 ]; then
    echo "Usage:"
    echo "  $0 cache <size>         # e.g., $0 cache 50"
    echo "  $0 sparsify <value>     # e.g., $0 sparsify 0.5 or $0 sparsify none"
    exit 1
fi

PARAM_TYPE=$1
PARAM_VALUE=$2

# Results
RESULTS_DIR="validation/results/quick_tests"
mkdir -p "${RESULTS_DIR}"

# Baselines
BASELINE_TASK_IL=70.19
PHASE3_DEFAULT=62.30

# Fixed arguments (SAME AS BASELINE)
FIXED_ARGS="--model causal-der --dataset seq-cifar100 \
--buffer_size 500 --alpha 0.3 --beta 0.5 --n_epochs 5 \
--batch_size 32 --minibatch_size 32 \
--lr 0.03 --optim_mom 0.0 --optim_wd 0.0 \
--enable_causal_graph_learning 1 \
--seed 1"

echo "========================================================================"
echo "🧪 QUICK PARAMETER TEST"
echo "========================================================================"
echo ""
echo "Scientific Control: Same test parameters as baseline"
echo ""
echo "FIXED (same as baseline):"
echo "  buffer=500, epochs=5, lr=0.03, batch=32, seed=1"
echo ""
echo "TESTING:"

# Backup
echo ""
echo "Creating backups..."
cp training/causal_der_v2.py training/causal_der_v2.py.backup_quick
cp training/causal_inference.py training/causal_inference.py.backup_quick
echo "✅ Backups created"
echo ""

if [ "$PARAM_TYPE" = "cache" ]; then
    CACHE_SIZE=$PARAM_VALUE
    echo "  Parameter: Cache Size = ${CACHE_SIZE}"
    echo "  Sparsification: 0.7 (default)"
    
    LOG_FILE="${RESULTS_DIR}/cache${CACHE_SIZE}_seed1.log"
    
    # Modify cache
    if [ "$CACHE_SIZE" -eq 0 ]; then
        echo ""
        echo "Disabling cache completely..."
        sed -i '' 's/if len(cache\['"'"'features'"'"'\]) < 200:/if False:  # DISABLED/' training/causal_der_v2.py
    else
        echo ""
        echo "Setting cache size to ${CACHE_SIZE}..."
        sed -i '' "s/if len(cache\['"'"'features'"'"'\]) < 200:/if len(cache['features']) < ${CACHE_SIZE}:/" training/causal_der_v2.py
    fi
    
elif [ "$PARAM_TYPE" = "sparsify" ]; then
    SPARS_VALUE=$PARAM_VALUE
    echo "  Parameter: Sparsification = ${SPARS_VALUE}"
    echo "  Cache size: 200 (default)"
    
    LOG_FILE="${RESULTS_DIR}/sparsify${SPARS_VALUE}_seed1.log"
    
    # Modify sparsification
    if [ "$SPARS_VALUE" = "none" ]; then
        echo ""
        echo "Disabling sparsification (keep all edges)..."
        sed -i '' 's/threshold = self.task_graph.abs().quantile(0.7)/# threshold = self.task_graph.abs().quantile(0.7)/' training/causal_inference.py
        sed -i '' 's/self.task_graph\[self.task_graph.abs() < threshold\] = 0/# self.task_graph[self.task_graph.abs() < threshold] = 0/' training/causal_inference.py
    else
        echo ""
        echo "Setting sparsification quantile to ${SPARS_VALUE}..."
        sed -i '' "s/threshold = self.task_graph.abs().quantile(0.7)/threshold = self.task_graph.abs().quantile(${SPARS_VALUE})/" training/causal_inference.py
    fi
else
    echo "Error: Unknown parameter type '${PARAM_TYPE}'"
    echo "Must be 'cache' or 'sparsify'"
    exit 1
fi

echo ""
echo "========================================================================"
echo ""
echo "Running experiment (ETA: ~52 minutes)..."
echo "Log: ${LOG_FILE}"
echo ""

read -p "Press Enter to start..."

python3 mammoth/utils/main.py ${FIXED_ARGS} 2>&1 | tee "${LOG_FILE}"

echo ""
echo "========================================================================"
echo "📊 RESULTS"
echo "========================================================================"
echo ""

# Extract results
TASK_IL=$(grep "Task-IL" "${LOG_FILE}" | tail -1 | grep -oE "[0-9]+\.[0-9]+" | head -1)
CLASS_IL=$(grep "Class-IL" "${LOG_FILE}" | tail -1 | grep -oE "[0-9]+\.[0-9]+" | tail -1)

echo "Configuration:"
if [ "$PARAM_TYPE" = "cache" ]; then
    echo "  Cache size:       ${CACHE_SIZE}"
    echo "  Sparsification:   0.7 (default)"
else
    echo "  Cache size:       200 (default)"
    echo "  Sparsification:   ${SPARS_VALUE}"
fi
echo ""

echo "Results:"
echo "  Task-IL:          ${TASK_IL}%"
echo "  Class-IL:         ${CLASS_IL}%"
echo ""

echo "Comparison:"
echo "  Baseline:         ${BASELINE_TASK_IL}%"
echo "  Phase 3 default:  ${PHASE3_DEFAULT}%"
echo "  This test:        ${TASK_IL}%"
echo ""

IMPROVE_VS_DEFAULT=$(python3 -c "print(f'{${TASK_IL} - ${PHASE3_DEFAULT}:+.2f}')")
GAP_VS_BASELINE=$(python3 -c "print(f'{${BASELINE_TASK_IL} - ${TASK_IL}:.2f}')")

echo "  vs Default:       ${IMPROVE_VS_DEFAULT}%"
echo "  Gap to baseline:  ${GAP_VS_BASELINE}%"
echo ""

# Analysis
python3 << EOF
baseline = ${BASELINE_TASK_IL}
default = ${PHASE3_DEFAULT}
result = float("${TASK_IL}")

print("="*70)
print("📋 ANALYSIS")
print("="*70)
print()

if result >= baseline:
    print("✅ SUCCESS: BEATS BASELINE!")
    print()
    print(f"   This parameter change works!")
    print(f"   Improvement: +{result - baseline:.2f}% vs baseline")
    print()
    print("📌 NEXT STEPS:")
    print("   1. Keep this parameter value")
    print("   2. Try optimizing the other parameter")
    print("   3. Run full parameter sweep to find optimal combination")
    
elif result > default + 2.0:
    print("⚠️  GOOD: Significant improvement")
    print()
    print(f"   Improvement: +{result - default:.2f}% vs default")
    print(f"   Still {baseline - result:.2f}% below baseline")
    print()
    print("📌 NEXT STEPS:")
    print("   1. This helps, but not enough alone")
    print("   2. Try combining with other parameter changes")
    print("   3. Run full parameter sweep")
    
elif result > default:
    print("🤔 MARGINAL: Small improvement")
    print()
    print(f"   Improvement: +{result - default:.2f}% vs default")
    print(f"   Still {baseline - result:.2f}% below baseline")
    print()
    print("📌 NEXT STEPS:")
    print("   1. Minor help, try other parameters")
    print("   2. Consider combination of changes")
    
else:
    print("❌ NO IMPROVEMENT")
    print()
    print(f"   Change: {result - default:+.2f}%")
    print("   This parameter is not the problem")
    print()
    print("📌 NEXT STEPS:")
    print("   1. Try different parameter")
    print("   2. Consider algorithmic changes")

print()
print("="*70)
EOF

echo ""
echo "Log saved: ${LOG_FILE}"
echo ""
echo "To restore original files:"
echo "  cp training/causal_der_v2.py.backup_quick training/causal_der_v2.py"
echo "  cp training/causal_inference.py.backup_quick training/causal_inference.py"
echo ""

# Restore
echo "Restoring original files..."
cp training/causal_der_v2.py.backup_quick training/causal_der_v2.py
cp training/causal_inference.py.backup_quick training/causal_inference.py
echo "✅ Original files restored"
echo ""

