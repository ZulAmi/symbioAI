#!/bin/bash
# Compare all three causality modes
# Mode 0: Vanilla DER++
# Mode 1: Heuristic-only
# Mode 2: Hybrid (Heuristic → TRUE Causal)

echo "=============================================="
echo "CAUSALITY MODES COMPARISON"
echo "=============================================="
echo ""
echo "Testing three configurations:"
echo "  1. Mode 0: Vanilla DER++ (baseline)"
echo "  2. Mode 1: Heuristic-only (fast feature filtering)"
echo "  3. Mode 2: Hybrid (heuristic → TRUE causal)"
echo ""
echo "Quick validation: 3 tasks, 1 epoch each"
echo ""

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BASE_DIR="validation/results/mode_comparison_${TIMESTAMP}"
mkdir -p "$BASE_DIR"

# Common arguments
COMMON_ARGS="--model derpp-causal \
    --dataset seq-cifar100 \
    --buffer_size 500 \
    --alpha 0.5 \
    --beta 0.5 \
    --lr 0.03 \
    --optim_mom 0.9 \
    --optim_wd 0.0001 \
    --n_epochs 1 \
    --batch_size 128 \
    --minibatch_size 128 \
    --enable_causal_graph_learning 0 \
    --use_causal_sampling 1 \
    --seed 1"

echo "=============================================="
echo "MODE 0: VANILLA DER++ (Baseline)"
echo "=============================================="
echo ""

python3 mammoth/utils/main.py \
    $COMMON_ARGS \
    --use_true_causality 0 \
    2>&1 | tee "$BASE_DIR/mode0_vanilla.log"

echo ""
echo "Mode 0 complete. Pausing 5 seconds..."
sleep 5

echo ""
echo "=============================================="
echo "MODE 1: HEURISTIC-ONLY"
echo "=============================================="
echo ""

python3 mammoth/utils/main.py \
    $COMMON_ARGS \
    --use_true_causality 1 \
    --causal_effect_threshold 0.05 \
    2>&1 | tee "$BASE_DIR/mode1_heuristic.log"

echo ""
echo "Mode 1 complete. Pausing 5 seconds..."
sleep 5

echo ""
echo "=============================================="
echo "MODE 2: HYBRID (Heuristic → TRUE Causal)"
echo "=============================================="
echo ""

python3 mammoth/utils/main.py \
    $COMMON_ARGS \
    --use_true_causality 2 \
    --causal_hybrid_candidates 200 \
    --causal_num_interventions 50 \
    --causal_effect_threshold 0.05 \
    2>&1 | tee "$BASE_DIR/mode2_hybrid.log"

echo ""
echo "=============================================="
echo "ALL MODES COMPLETE!"
echo "=============================================="
echo ""
echo "Results saved to: $BASE_DIR/"
echo ""
echo "Compare final accuracy across modes:"
grep "Average accuracy:" "$BASE_DIR"/mode*.log
echo ""
echo "Check Stage 2 TRUE causal effects in mode 2:"
grep -A 3 "STAGE 2" "$BASE_DIR/mode2_hybrid.log"
echo ""
echo "Expected differences:"
echo "  Mode 0 (vanilla): No causal selection"
echo "  Mode 1 (heuristic): Fast, correlation-based"
echo "  Mode 2 (hybrid): TRUE causal intervention (should show effect variation)"
echo ""
