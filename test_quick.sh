#!/bin/bash
# Quick test - 1 epoch to verify no crashes
# For full baseline validation, use test_baseline.sh (50 epochs)

echo "================================================================"
echo "QUICK TEST - Causal-DER v2 Baseline (1 epoch)"
echo "================================================================"
echo ""
echo "This is a quick sanity check to verify:"
echo "  ✅ No import errors"
echo "  ✅ No crashes during training"
echo "  ✅ Buffer fills correctly (500/500)"
echo "  ✅ Loss decreases"
echo ""
echo "For FULL baseline validation (50 epochs), run:"
echo "  ./test_baseline.sh"
echo ""
echo "================================================================"
echo ""

# Create results directory if it doesn't exist
mkdir -p validation/results

python3 mammoth/utils/main.py --model causal-der \
  --dataset seq-cifar100 \
  --buffer_size 500 \
  --alpha 0.1 --beta 0.5 \
  --n_epochs 1 \
  --batch_size 128 \
  --minibatch_size 32 \
  --lr 0.03 \
  --optim_mom 0 \
  --optim_wd 0.0 \
  --seed 1 \
  | tee validation/results/quick_test_1epoch.log

echo ""
echo "================================================================"
echo "QUICK TEST COMPLETE!"
echo "================================================================"
echo ""
echo "Results saved to: validation/results/quick_test_1epoch.log"
echo ""
echo "If successful:"
echo "  ✅ No errors - proceed to full baseline test"
echo "  → Run: ./test_baseline_corrected.sh"
echo ""
echo "If errors:"
echo "  ❌ Fix issues before running full test"
echo ""
