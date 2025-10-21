#!/bin/bash

echo "================================================================"
echo "CAUSAL-DER V2 - PHASE 1: CORRECTED BASELINE TEST"
echo "================================================================"
echo ""
echo "FIXING PREVIOUS ERROR:"
echo "  ❌ Used: alpha=0.1, batch_size=10, lr_gamma=0.1"
echo "  ✅ Using: alpha=0.3, batch_size=32, lr_gamma=0.2 (OFFICIAL CONFIG)"
echo ""
echo "Target: Match Mammoth DER++ performance (~56-70% Task-IL)"
echo "Using OFFICIAL Mammoth derpp.yaml + xder.yaml config:"
echo "  alpha=0.3, beta=0.5, lr=0.03, momentum=0"
echo "  buffer_size=500, n_epochs=50"
echo "  batch_size=32 (NOT 10!)"
echo "  lr_scheduler=multisteplr, lr_gamma=0.2 (NOT 0.1!)"
echo ""
echo "================================================================"
echo ""

cd /Users/zulhilmirahmat/Development/programming/Symbio\ AI/mammoth

python3 utils/main.py \
  --model causal-der \
  --dataset seq-cifar100 \
  --buffer_size 500 \
  --alpha 0.3 \
  --beta 0.5 \
  --lr 0.03 \
  --optim_mom 0.0 \
  --n_epochs 50 \
  --batch_size 32 \
  --lr_scheduler multisteplr \
  --lr_milestones 35 45 \
  --sched_multistep_lr_gamma 0.2 \
  --seed 1 \
  --nowand 1

echo ""
echo "================================================================"
echo "BASELINE TEST COMPLETE!"
echo "================================================================"
echo ""
echo "Expected: Task-IL ~56-70% (matching official DER++)"
echo "Log saved to: baseline_test_corrected_seed1.log"
echo ""
echo "If matched:"
echo "  ✅ Phase 1 complete - baseline verified!"
echo "  → Proceed to Phase 2 (causal importance scoring)"
echo ""
echo "If not matched:"
echo "  ❌ Further debugging needed"
echo "  → Check implementation details"
