#!/bin/bash
# Run graph learning experiments across multiple seeds for consistency analysis
# Usage: ./run_multiseed_graph_learning.sh

set -e  # Exit on error

# Configuration
SEEDS=(1 2 3 4 5)
MODEL="derpp"  # Use base DER++ + graph learning (like Exp 3)
DATASET="seq-cifar100"
N_EPOCHS=50
BUFFER_SIZE=500
BATCH_SIZE=128
MINIBATCH_SIZE=128

# Create output directory
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="validation/results/multiseed_graph_${TIMESTAMP}"
mkdir -p "$OUTPUT_DIR"

echo "========================================"
echo "Multi-Seed Graph Learning Experiment"
echo "========================================"
echo "Seeds: ${SEEDS[@]}"
echo "Output: $OUTPUT_DIR"
echo ""

# Run experiment for each seed
for SEED in "${SEEDS[@]}"; do
    echo "----------------------------------------"
    echo "Starting seed $SEED..."
    echo "----------------------------------------"
    
    LOG_FILE="$OUTPUT_DIR/seed_${SEED}.log"
    
    python3 mammoth/utils/main.py \
        --model "$MODEL" \
        --dataset "$DATASET" \
        --buffer_size "$BUFFER_SIZE" \
        --n_epochs "$N_EPOCHS" \
        --batch_size "$BATCH_SIZE" \
        --minibatch_size "$MINIBATCH_SIZE" \
        --alpha 0.5 \
        --beta 0.5 \
        --nowand 1 \
        --seed "$SEED" \
        2>&1 | tee "$LOG_FILE"
    
    echo "✅ Seed $SEED complete"
    echo ""
done

echo "========================================"
echo "All seeds complete!"
echo "========================================"
echo ""
echo "Results saved to: $OUTPUT_DIR"
echo ""
echo "To analyze consistency, run:"
echo "  python3 visualization/compare_multiseed_graphs.py \\"
echo "    --log_paths $OUTPUT_DIR/seed_*.log \\"
echo "    --output_dir visualization/figures/multiseed_${TIMESTAMP}/"
echo ""
