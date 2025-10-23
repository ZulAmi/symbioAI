#!/bin/bash

################################################################################
# Test Causal Graph Learning (Phase 3)
################################################################################
#
# CRITICAL TEST: Determine if causal graph learning provides benefit.
# 
# Decision Point:
#   - If result > 71% Task-IL → Proceed with 6-week publication plan
#   - If result < 71% Task-IL → Pivot to analysis paper
#
# Author: Symbio AI
# Date: October 22, 2025
################################################################################

set -e

echo "========================================"
echo "🔬 Phase 3: Causal Graph Learning Test"
echo "========================================"
echo ""
echo "⚠️  CRITICAL DECISION POINT"
echo ""
echo "This test determines paper direction:"
echo "  > 71% Task-IL → Full causal paper"
echo "  < 71% Task-IL → Analysis/pivot"
echo ""
echo "Baseline: 70.19% Task-IL"
echo "Target:   71%+ Task-IL (improvement needed)"
echo ""
echo "========================================"
echo ""

# Configuration
OUTPUT_DIR="validation/results"
LOG_FILE="${OUTPUT_DIR}/phase3_graph_learning_seed1.log"

mkdir -p "${OUTPUT_DIR}"

read -p "Press Enter to start test (will take ~52 minutes)..."

echo ""
echo "Starting Phase 3 test..."
echo ""

# NOTE: --enable_causal_graph_learning must be added to mammoth/models/causal_der.py first!
# For now, this will fail gracefully if not implemented

python3 mammoth/utils/main.py --model causal-der --dataset seq-cifar100 \
  --buffer_size 500 --alpha 0.3 --beta 0.5 --n_epochs 5 \
  --batch_size 32 --minibatch_size 32 \
  --lr 0.03 --optim_mom 0.0 --optim_wd 0.0 \
  --enable_causal_graph_learning 1 \
  --seed 1 \
  2>&1 | tee "${LOG_FILE}"

echo ""
echo "========================================"
echo "📊 Phase 3 Results"
echo "========================================"
echo ""

# Extract results
BASELINE_ACC=70.19
PHASE3_ACC=$(grep "Task-IL" "${LOG_FILE}" | tail -1 | grep -oE "[0-9]+\.[0-9]+" | head -1)

echo "Baseline:  ${BASELINE_ACC}%"
echo "Phase 3:   ${PHASE3_ACC}%"
echo "Gain:      $(python3 -c "print(f'{${PHASE3_ACC} - ${BASELINE_ACC}:+.2f}%')")"
echo ""

# Decision analysis
python3 << EOF
baseline = ${BASELINE_ACC}
phase3 = float("${PHASE3_ACC}")
improvement = phase3 - baseline

print("="*60)
print("📋 DECISION ANALYSIS")
print("="*60)

if phase3 > 71.0:
    print("")
    print("✅ SUCCESS: Graph learning provides benefit!")
    print("")
    print(f"   Improvement: {improvement:+.2f}%")
    print(f"   New performance: {phase3:.2f}% Task-IL")
    print("")
    print("📌 RECOMMENDATION:")
    print("   → Proceed with Option A (6-week publication plan)")
    print("   → Focus: Causal graph learning for continual learning")
    print("   → Run multi-seed validation: ./run_multiseed.sh graph_learning 5")
    print("")
elif phase3 > baseline + 0.5:
    print("")
    print("⚠️  MARGINAL: Small improvement detected")
    print("")
    print(f"   Improvement: {improvement:+.2f}%")
    print(f"   New performance: {phase3:.2f}% Task-IL")
    print("")
    print("📌 RECOMMENDATION:")
    print("   → Test with multiple seeds to confirm significance")
    print("   → Run: ./run_multiseed.sh graph_learning 5")
    print("   → If mean improvement > 1%, proceed with paper")
    print("   → Otherwise, consider Option C (pivot)")
    print("")
else:
    print("")
    print("❌ NO BENEFIT: Graph learning doesn't help")
    print("")
    print(f"   Change: {improvement:+.2f}% (within noise)")
    print(f"   Performance: {phase3:.2f}% Task-IL")
    print("")
    print("📌 RECOMMENDATION:")
    print("   → Pursue Option C (4-week pivot)")
    print("   → Focus: Analysis paper on limitations")
    print("   → Title: 'When Does Causal Inference Help Continual Learning?'")
    print("   → Honest assessment more valuable than negative results")
    print("")

print("="*60)
EOF

echo ""
echo "Log saved to: ${LOG_FILE}"
echo ""
echo "Next steps based on results above ^^^"
echo ""
