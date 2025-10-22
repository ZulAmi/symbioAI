#!/bin/bash

echo "================================================================"
echo "CAUSAL-DER V2 - PHASE 1: TEST CLEAN DER++ IMPLEMENTATION"
echo "================================================================"
echo ""
echo "Testing causal_der_v2.py implementation"
echo "Should match official DER++ baseline: 73.81% Task-IL"
echo ""
echo "Using SAME configuration as official baseline:"
echo "  alpha=0.3, beta=0.5, lr=0.03, momentum=0"
echo "  buffer_size=500, n_epochs=5"
echo "  batch_size=32"
echo "  lr_scheduler=multisteplr, milestones=[3,4], gamma=0.2"
echo ""
echo "Official DER++ Baseline: 73.81% Task-IL (5 epochs)"
echo "Target: Match within 1-2% (71-75% Task-IL acceptable)"
echo ""
echo "================================================================"
echo ""

cd /Users/zulhilmirahmat/Development/programming/Symbio\ AI/mammoth

# Create results directory if it doesn't exist
mkdir -p ../validation/results

# Run test and save output
python3 utils/main.py \
  --model causal-der \
  --dataset seq-cifar100 \
  --buffer_size 500 \
  --alpha 0.3 \
  --beta 0.5 \
  --lr 0.03 \
  --optim_mom 0.0 \
  --n_epochs 5 \
  --batch_size 32 \
  --minibatch_size 32 \
  --lr_scheduler multisteplr \
  --lr_milestones 3 4 \
  --sched_multistep_lr_gamma 0.2 \
  --seed 1 \
  | tee ../validation/results/causal_der_v2_baseline_5epoch_seed1.log

echo ""
echo "================================================================"
echo "CAUSAL-DER V2 BASELINE TEST COMPLETE!"
echo "================================================================"
echo ""
echo "Official DER++ Baseline:  73.81% Task-IL"
echo "Causal-DER V2 Result:     [Check log above]"
echo ""
echo "Results saved to: validation/results/causal_der_v2_baseline_5epoch_seed1.log"
echo ""
echo "Next steps:"
echo "  ✅ If matched (71-75%): Phase 1 complete! Proceed to Phase 2"
echo "  ❌ If not matched: Debug causal_der_v2 implementation"
echo ""
