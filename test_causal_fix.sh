#!/bin/bash
################################################################################
# Quick Test: Verify Causal Sampling Fix
# 
# Tests the FIXED causal sampling with corrected graph indexing
# Expected: Should match or BEAT baseline (73.81%)
################################################################################

RESULTS_DIR="validation/results/causal_fix_test_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RESULTS_DIR"

# Base configuration
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
echo "║              TESTING CAUSAL SAMPLING FIX                           ║"
echo "╚════════════════════════════════════════════════════════════════════╝"
echo ""
echo "FIXES APPLIED:"
echo "  1. ✅ Flipped graph indexing: graph[current, old] instead of [old, current]"
echo "  2. ✅ Reduced blend ratio: 0.3 (was 0.7) - less aggressive"
echo "  3. ✅ Extended warmup: 5 tasks (was 3) - more stable"
echo "  4. ✅ Simplified importance: direct causal strength (no double-blend)"
echo ""
echo "COMPARISON:"
echo "  Previous Full CausalDER (buggy): 71.75%"
echo "  Baseline (uniform DER++):        73.81%"
echo "  Target (fixed CausalDER):        >73.81% (beat baseline!)"
echo ""
echo "Results will be saved to: $RESULTS_DIR"
echo ""

# Test 1: Fixed CausalDER with NEW defaults (blend=0.3, warmup=5)
echo "════════════════════════════════════════════════════════════════════"
echo "  Test 1: Fixed CausalDER (NEW defaults: blend=0.3, warmup=5)"
echo "  Expected: Should match or beat baseline (>73.8%)"
echo "════════════════════════════════════════════════════════════════════"
python3 mammoth/utils/main.py --model derpp-causal $BASE_PARAMS \
  --alpha 0.3 \
  --beta 0.5 \
  --enable_causal_graph_learning 1 \
  --use_causal_sampling 1 \
  2>&1 | tee "$RESULTS_DIR/1_fixed_causal_default.log"

acc=$(grep "Accuracy for 10 task" "$RESULTS_DIR/1_fixed_causal_default.log" | tail -1 | awk -F'Task-IL\]:' '{print $2}' | awk '{print $1}')
echo "  ✅ Fixed CausalDER (defaults): $acc%"
echo ""

# Test 2: Try even MORE conservative (blend=0.1, warmup=7)
echo "════════════════════════════════════════════════════════════════════"
echo "  Test 2: Ultra-conservative (blend=0.1, warmup=7)"
echo "  Expected: Very close to baseline"
echo "════════════════════════════════════════════════════════════════════"
python3 mammoth/utils/main.py --model derpp-causal $BASE_PARAMS \
  --alpha 0.3 \
  --beta 0.5 \
  --enable_causal_graph_learning 1 \
  --use_causal_sampling 1 \
  --causal_blend_ratio 0.1 \
  --causal_warmup_tasks 7 \
  2>&1 | tee "$RESULTS_DIR/2_ultra_conservative.log"

acc=$(grep "Accuracy for 10 task" "$RESULTS_DIR/2_ultra_conservative.log" | tail -1 | awk -F'Task-IL\]:' '{print $2}' | awk '{print $1}')
echo "  ✅ Ultra-conservative: $acc%"
echo ""

# Test 3: Try MORE aggressive (blend=0.5, warmup=5)
echo "════════════════════════════════════════════════════════════════════"
echo "  Test 3: More aggressive (blend=0.5, warmup=5)"
echo "  Expected: Higher than defaults if fix is correct"
echo "════════════════════════════════════════════════════════════════════"
python3 mammoth/utils/main.py --model derpp-causal $BASE_PARAMS \
  --alpha 0.3 \
  --beta 0.5 \
  --enable_causal_graph_learning 1 \
  --use_causal_sampling 1 \
  --causal_blend_ratio 0.5 \
  --causal_warmup_tasks 5 \
  2>&1 | tee "$RESULTS_DIR/3_more_aggressive.log"

acc=$(grep "Accuracy for 10 task" "$RESULTS_DIR/3_more_aggressive.log" | tail -1 | awk -F'Task-IL\]:' '{print $2}' | awk '{print $1}')
echo "  ✅ More aggressive: $acc%"
echo ""

# Summary
echo "╔════════════════════════════════════════════════════════════════════╗"
echo "║                        FIX VALIDATION SUMMARY                      ║"
echo "╚════════════════════════════════════════════════════════════════════╝"
echo ""
echo "Results:"
echo "───────────────────────────────────────────────────────────────────"

test1=$(grep "Accuracy for 10 task" "$RESULTS_DIR/1_fixed_causal_default.log" 2>/dev/null | tail -1 | awk -F'Task-IL\]:' '{print $2}' | awk '{print $1}')
test2=$(grep "Accuracy for 10 task" "$RESULTS_DIR/2_ultra_conservative.log" 2>/dev/null | tail -1 | awk -F'Task-IL\]:' '{print $2}' | awk '{print $1}')
test3=$(grep "Accuracy for 10 task" "$RESULTS_DIR/3_more_aggressive.log" 2>/dev/null | tail -1 | awk -F'Task-IL\]:' '{print $2}' | awk '{print $1}')

baseline=73.81
previous_buggy=71.75

if [ -n "$test1" ]; then
    echo "  1. Fixed (blend=0.3, warmup=5):     $test1%"
    delta=$(python3 -c "print(f'{float($test1) - $baseline:.2f}')" 2>/dev/null || echo "N/A")
    echo "     vs Baseline (73.81%):           $delta%"
fi

if [ -n "$test2" ]; then
    echo "  2. Ultra-conservative (0.1, 7):    $test2%"
    delta=$(python3 -c "print(f'{float($test2) - $baseline:.2f}')" 2>/dev/null || echo "N/A")
    echo "     vs Baseline (73.81%):           $delta%"
fi

if [ -n "$test3" ]; then
    echo "  3. Aggressive (blend=0.5, warmup=5): $test3%"
    delta=$(python3 -c "print(f'{float($test3) - $baseline:.2f}')" 2>/dev/null || echo "N/A")
    echo "     vs Baseline (73.81%):           $delta%"
fi

echo ""
echo "Reference:"
echo "  Previous buggy CausalDER:  71.75%"
echo "  Baseline (DER++):          73.81%"
echo ""
echo "ANALYSIS:"
if [ -n "$test1" ]; then
    result=$(python3 -c "
if $test1 > $baseline:
    print('✅ SUCCESS! Fixed CausalDER BEATS baseline!')
    print('   The bug fix worked - causal sampling now helps!')
elif $test1 > $previous_buggy + 1.0:
    print('⚠️  PARTIAL SUCCESS: Better than buggy version, but not baseline yet')
    print('   May need further tuning of blend ratio / warmup')
else:
    print('❌ STILL BROKEN: Fix did not help')
    print('   Need to investigate further')
" 2>/dev/null)
    echo "$result"
fi

echo ""
echo "Results saved to: $RESULTS_DIR"
echo ""
