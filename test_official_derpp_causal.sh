#!/bin/bash
#
# Test Official DER++ Causal Extension
# =====================================
#
# This tests the NEW clean implementation that properly extends
# official Mammoth DER++ instead of reimplementing from scratch.
#
# Expected Performance:
#   - Phase 1 (baseline, causal OFF): 73.81% Task-IL (official DER++)
#   - Phase 2 (causal ON): TBD - needs validation
#
# File: training/derpp_causal.py (NEW - extends official derpp.py)
# NOT:  training/causal_der_v2.py (OLD - custom implementation)
#

echo "========================================================================"
echo "🧪 Testing Official DER++ Causal Extension"
echo "========================================================================"
echo ""
echo "NEW Architecture:"
echo "  Base:       mammoth/models/derpp.py (official 73.81%)"
echo "  Extension:  training/derpp_causal.py (causal layer)"
echo "  Adapter:    mammoth/models/derpp_causal.py"
echo ""
echo "OLD Architecture (DEPRECATED):"
echo "  Custom:     training/causal_der_v2.py (70.19% → 63.74%)"
echo ""
echo "========================================================================"
echo ""

# Test 1: Official DER++ baseline (Phase 1 - causal OFF)
echo "Test 1: Phase 1 Baseline (Causal OFF)"
echo "--------------------------------------"
echo "Expected: 73.81% Task-IL (match official DER++)"
echo "Model: derpp-causal with --enable_causal_graph_learning 0"
echo ""
echo "Parameters (matching official DER++):"
echo "  --buffer_size 500"
echo "  --alpha 0.3"
echo "  --beta 0.5"
echo "  --n_epochs 5"
echo "  --batch_size 32"
echo "  --minibatch_size 32"
echo "  --lr 0.03"
echo "  --lr_scheduler multisteplr"
echo "  --lr_milestones 3 4"
echo "  --sched_multistep_lr_gamma 0.2"
echo "  --enable_causal_graph_learning 0"
echo "  --use_causal_sampling 0"
echo "  --temperature 2.0"
echo ""
echo "Running full 5-epoch test (same as official baseline)..."
echo "ETA: ~52 minutes"
echo ""

# Create results directory
mkdir -p validation/results/official_derpp_causal

# Full 5-epoch test (scientifically accurate)
python3 mammoth/utils/main.py \
    --model derpp-causal \
    --dataset seq-cifar100 \
    --buffer_size 500 \
    --alpha 0.3 \
    --beta 0.5 \
    --n_epochs 5 \
    --batch_size 32 \
    --minibatch_size 32 \
    --lr 0.03 \
    --lr_scheduler multisteplr \
    --lr_milestones 3 4 \
    --sched_multistep_lr_gamma 0.2 \
    --enable_causal_graph_learning 0 \
    --use_causal_sampling 0 \
    --temperature 2.0 \
    --seed 1 \
    2>&1 | tee validation/results/official_derpp_causal/baseline_5epoch_seed1.log

# Extract result
RESULT=$(grep "Accuracy for 10 task" validation/results/official_derpp_causal/baseline_5epoch_seed1.log | tail -1)

echo ""
echo "========================================================================"
echo "Phase 1 Baseline Result (5 epochs):"
echo "========================================================================"
echo "$RESULT"
echo ""
echo "Comparison:"
echo "  Official DER++:           73.81% Task-IL"
echo "  This test (causal OFF):   [see above]"
echo ""
echo "Log: validation/results/official_derpp_causal/baseline_5epoch_seed1.log"
echo ""
echo "Next steps:"
echo "  ✅ If matched (~73.81%): Architecture validated! Run Phase 2"
echo "  ❌ If not matched: Debug derpp_causal.py implementation"
echo ""
echo "To run Phase 2 (causal ON):"
echo "  ./test_official_derpp_causal_phase2.sh"
echo ""
