#!/bin/bash
# Causal-DER 5-Epoch Stable Run
# Designed for reliable results and paper submission

echo "====================================================="
echo "Causal-DER 5-Epoch Experiment (Stable Configuration)"
echo "====================================================="
echo ""
echo "Settings optimized for:"
echo "  - No NaN/Inf issues"
echo "  - Reliable convergence"
echo "  - Comparable to DER++ baseline"
echo ""
echo "Expected runtime: ~20-25 minutes"
echo "====================================================="

cd "$(dirname "$0")/mammoth"

# Configuration
MODEL="causal-der"
DATASET="seq-cifar100"
BUFFER_SIZE=500
ALPHA=0.3
BETA=0.5
N_EPOCHS=5
BATCH_SIZE=32
LR=0.03  # DER++ exact setting
OPTIM_WD=0.0  # DER++ uses 0
CLIP_GRAD=10.0  # DER++ standard (not 1.0!)
SEED=42

# Run Causal-DER
echo "[1/2] Running Causal-DER (5 epochs)..."
python3 utils/main.py \
  --model $MODEL \
  --dataset $DATASET \
  --buffer_size $BUFFER_SIZE \
  --alpha $ALPHA \
  --beta $BETA \
  --n_epochs $N_EPOCHS \
  --batch_size $BATCH_SIZE \
  --lr $LR \
  --optim_wd $OPTIM_WD \
  --clip_grad $CLIP_GRAD \
  --use_lr_decay 0 \
  --adaptive_replay_weights 0 \
  --mixed_precision 0 \
  --seed $SEED \
  --dataset_config xder \
  2>&1 | tee ~/causal_der_5epoch_stable.log

echo ""
echo "[2/2] Running DER++ Baseline (5 epochs)..."
python3 utils/main.py \
  --model derpp \
  --dataset $DATASET \
  --buffer_size $BUFFER_SIZE \
  --alpha $ALPHA \
  --beta $BETA \
  --n_epochs $N_EPOCHS \
  --batch_size $BATCH_SIZE \
  --lr $LR \
  --optim_wd $OPTIM_WD \
  --seed $SEED \
  --dataset_config xder \
  2>&1 | tee ~/derpp_5epoch_baseline.log

echo ""
echo "====================================================="
echo "Experiments Complete!"
echo "====================================================="
echo "Results saved to:"
echo "  - ~/causal_der_5epoch_stable.log"
echo "  - ~/derpp_5epoch_baseline.log"
echo ""
echo "To view results:"
echo "  tail -50 ~/causal_der_5epoch_stable.log"
echo "  tail -50 ~/derpp_5epoch_baseline.log"
echo "====================================================="
