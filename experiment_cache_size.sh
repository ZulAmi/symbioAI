#!/bin/bash

################################################################################
# Quick Experiment: Test Reduced Cache Size
################################################################################
#
# Hypothesis: Feature cache (200 samples/task) interferes with main buffer
# Evidence: Early tasks show -10% to -14% accuracy drop
# 
# Test: Reduce cache from 200 → 50 samples per task
# Expected: +4-6% Task-IL improvement
#
# Author: Symbio AI
# Date: October 23, 2025
################################################################################

set -e

echo "========================================"
echo "🧪 Quick Experiment: Reduced Cache Size"
echo "========================================"
echo ""
echo "Hypothesis: Cache interference causes performance drop"
echo "Evidence:   Early tasks: -10% to -14% accuracy"
echo "Change:     200 samples → 50 samples per task"
echo "Expected:   62.3% → 66-68% Task-IL"
echo ""
echo "========================================"
echo ""

# Backup original file
echo "Creating backup..."
cp training/causal_der_v2.py training/causal_der_v2.py.backup_cache200
echo "✅ Backup saved: training/causal_der_v2.py.backup_cache200"
echo ""

# Modify cache size
echo "Modifying cache size (200 → 50)..."
sed -i '' 's/if len(cache\['"'"'features'"'"'\]) < 200:/if len(cache["features"]) < 50:/' training/causal_der_v2.py

# Also update comment
sed -i '' 's/Store up to 200 samples per task/Store up to 50 samples per task/' training/causal_der_v2.py

echo "✅ Modified training/causal_der_v2.py"
echo ""

# Verify change
echo "Verifying change..."
grep -n "< 50" training/causal_der_v2.py | head -1
echo ""

read -p "Press Enter to run experiment (~52 minutes)..."
echo ""

# Run experiment
OUTPUT_DIR="validation/results"
LOG_FILE="${OUTPUT_DIR}/phase3_cache50_seed1.log"

mkdir -p "${OUTPUT_DIR}"

python3 mammoth/utils/main.py --model causal-der --dataset seq-cifar100 \
  --buffer_size 500 --alpha 0.3 --beta 0.5 --n_epochs 5 \
  --batch_size 32 --minibatch_size 32 \
  --lr 0.03 --optim_mom 0.0 --optim_wd 0.0 \
  --enable_causal_graph_learning 1 \
  --seed 1 \
  2>&1 | tee "${LOG_FILE}"

echo ""
echo "========================================"
echo "📊 Results Comparison"
echo "========================================"
echo ""

# Extract results
BASELINE_ACC=70.19
CACHE200_ACC=62.30
CACHE50_ACC=$(grep "Task-IL" "${LOG_FILE}" | tail -1 | grep -oE "[0-9]+\.[0-9]+" | head -1)

echo "Baseline (no graph):        ${BASELINE_ACC}%"
echo "Cache 200 (current):        ${CACHE200_ACC}%"
echo "Cache 50 (experiment):      ${CACHE50_ACC}%"
echo ""
echo "Change from Cache 200:      $(python3 -c "print(f'{${CACHE50_ACC} - ${CACHE200_ACC}:+.2f}%')")"
echo "Gap to baseline:            $(python3 -c "print(f'{${BASELINE_ACC} - ${CACHE50_ACC}:.2f}%')")"
echo ""

# Analysis
python3 << EOF
baseline = ${BASELINE_ACC}
cache200 = ${CACHE200_ACC}
cache50 = float("${CACHE50_ACC}")

improvement = cache50 - cache200
gap = baseline - cache50

print("="*60)
print("📋 ANALYSIS")
print("="*60)
print()

if cache50 >= baseline:
    print("✅ SUCCESS: Cache reduction WORKS!")
    print()
    print(f"   Improvement: +{improvement:.2f}% from Cache 200")
    print(f"   Surpassed baseline by: +{cache50 - baseline:.2f}%")
    print()
    print("📌 NEXT STEPS:")
    print("   1. Keep cache_size=50 as new default")
    print("   2. Test cache_size=0 (no cache at all)")
    print("   3. Test cache_size=25 (even smaller)")
    print("   4. Run multi-seed validation")
    
elif cache50 > cache200 + 2.0:
    print("⚠️  GOOD PROGRESS: Significant improvement")
    print()
    print(f"   Improvement: +{improvement:.2f}% from Cache 200")
    print(f"   Still below baseline by: -{gap:.2f}%")
    print()
    print("📌 NEXT STEPS:")
    print("   1. Try cache_size=25 or 0 (smaller/no cache)")
    print("   2. Combine with reduced sparsification")
    print("   3. Test alternative causal estimation methods")
    
elif cache50 > cache200:
    print("🤔 MARGINAL: Small improvement")
    print()
    print(f"   Improvement: +{improvement:.2f}% from Cache 200")
    print(f"   Still below baseline by: -{gap:.2f}%")
    print()
    print("📌 NEXT STEPS:")
    print("   1. Cache is part of the problem, but not all")
    print("   2. Try sparsification threshold next")
    print("   3. Consider gradient-based causal estimation")
    
else:
    print("❌ NO IMPROVEMENT: Cache size not the issue")
    print()
    print(f"   Change: {improvement:+.2f}% (worse or no change)")
    print()
    print("📌 NEXT STEPS:")
    print("   1. Restore cache_size=200")
    print("   2. Focus on sparsification threshold")
    print("   3. Try gradient-based causal estimation")
    print("   4. Consider fundamental algorithm issues")

print()
print("="*60)
EOF

echo ""
echo "Log saved to: ${LOG_FILE}"
echo ""
echo "To restore original:"
echo "  cp training/causal_der_v2.py.backup_cache200 training/causal_der_v2.py"
echo ""

