#!/bin/bash
# NUCLEAR OPTION: Run with ZERO causal features (just test if DER++ works)

echo "========================================================"
echo "DIAGNOSTIC: Testing if plain DER++ works on your system"
echo "========================================================"

cd "$(dirname "$0")/mammoth"

python3 utils/main.py \
  --model derpp \
  --dataset seq-cifar10 \
  --buffer_size 200 \
  --alpha 0.1 \
  --beta 0.5 \
  --n_epochs 1 \
  --batch_size 32 \
  --lr 0.03 \
  --seed 42 \
  2>&1 | tee ~/derpp_diagnostic.log

echo ""
echo "If this crashed with NaN → Mammoth/system issue (not your fault)"
echo "If this worked → Causal features are causing NaN"
