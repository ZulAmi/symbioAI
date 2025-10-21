#!/bin/bash
# Test Causal-DER v2 - Clean DER++ Baseline
# Should match Mammoth DER++ performance: ~56% Task-IL

echo "================================================================"
echo "CAUSAL-DER V2 - PHASE 1: CLEAN DER++ BASELINE"
echo "================================================================"
echo ""
echo "Target: Match Mammoth DER++ performance (56% Task-IL)"
echo "Using OFFICIAL DER++ hyperparameters:"
echo "  alpha=0.1, beta=0.5, lr=0.03, momentum=0"
echo "  buffer_size=500, n_epochs=50"
echo ""
echo "================================================================"
echo ""

python3 mammoth/utils/main.py --model causal-der \
  --dataset seq-cifar100 \
  --buffer_size 500 \
  --alpha 0.1 --beta 0.5 \
  --n_epochs 50 \
  --batch_size 10 \
  --minibatch_size 10 \
  --lr 0.03 \
  --optim_mom 0 \
  --optim_wd 0.0 \
  --seed 1 \
  | tee baseline_test_seed1.log

echo ""
echo "================================================================"
echo "BASELINE TEST COMPLETE!"
echo "================================================================"
echo ""
echo "Expected: Task-IL ≈ 56% (±2%)"
echo "Log saved to: baseline_test_seed1.log"
echo ""
echo "If matched:"
echo "  ✅ Phase 1 complete - baseline verified!"
echo "  → Proceed to Phase 2 (causal importance scoring)"
echo ""
echo "If not matched:"
echo "  ❌ Debug until baseline matches"
echo "  → Do NOT proceed until we have working baseline"
echo ""
