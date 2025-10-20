#!/bin/bash
# Test Causal-DER in PURE DER++ mode (all causal features disabled)
# This should match DER++ baseline performance

echo "================================================================"
echo "Testing Causal-DER in PURE DER++ MODE"
echo "All causal features disabled - should match DER++ baseline"
echo "================================================================"

python3 mammoth/utils/main.py --model causal-der \
  --dataset seq-cifar100 \
  --buffer_size 500 \
  --alpha 0.3 --beta 0.5 \
  --n_epochs 1 \
  --batch_size 128 \
  --minibatch_size 128 \
  --lr 0.03 \
  --optim_mom 0.9 \
  --optim_wd 0.0 \
  --use_causal_sampling 0 \
  --temperature 2.0 \
  --importance_weight_replay 0 \
  --mixed_precision 0 \
  --use_mir_sampling 0 \
  --mir_candidate_factor 3 \
  --store_features 0 \
  --feature_kd_weight 0.0 \
  --task_bias_strength 1.0 \
  --per_task_cap 50 \
  --buffer_dtype float32 \
  --store_logits_as logits32 \
  --use_neural_causal_discovery 0 \
  --use_counterfactual_replay 0 \
  --use_enhanced_irm 0 \
  --use_ate_pruning 0 \
  --use_task_free_streaming 0 \
  --use_adaptive_controller 0 \
  --enable_causal_graph_learning 0 \
  --seed 1

echo ""
echo "================================================================"
echo "Test completed!"
echo "Expected: Task-IL ≈ 56% (matching DER++ baseline)"
echo "If not matched, there are still bugs to fix"
echo "================================================================"
