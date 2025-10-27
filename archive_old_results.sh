#!/bin/bash
################################################################################
# Archive Obsolete Results (Old Hyperparameters)
# 
# These results used suboptimal hyperparameters:
# - n_epochs=5 (now using 10-20)
# - buffer_size=500 (now using 1000)
# - optim_wd=0.0 (now using 0.0001)
# - optim_mom=0.0 (now using 0.9)
# - batch_size=32 (now using 64)
#
# Archiving instead of deleting for reference.
################################################################################

cd "/Users/zulhilmirahmat/Development/programming/Symbio AI/validation/results"

# Create archive directory
mkdir -p _archived_old_hyperparameters_20251027

echo "Archiving obsolete results with old hyperparameters..."

# Move obsolete results
mv alpha_sweep_20251026_065607/ _archived_old_hyperparameters_20251027/
mv alpha_sweep_20251027_092640/ _archived_old_hyperparameters_20251027/
mv alpha_sweep_20251027_093359/ _archived_old_hyperparameters_20251027/
mv causal_tuning_sweep_20251025_114638/ _archived_old_hyperparameters_20251027/
mv fixed_baseline/ _archived_old_hyperparameters_20251027/
mv official_derpp_causal/ _archived_old_hyperparameters_20251027/
mv quick_tests/ _archived_old_hyperparameters_20251027/

# Remove orphaned files
rm -f causal_der.py  # Duplicate file

echo ""
echo "✅ Archived 7 folders with old hyperparameters"
echo ""
echo "Keeping (still valid):"
echo "  ✓ multiseed/quickwins_20251024_101754/ (baseline reference)"
echo "  ✓ quick_validation_3datasets/ (cross-dataset validation)"
echo ""
echo "Next steps:"
echo "  1. Run improved config: buffer=1000, epochs=10, weight_decay=0.0001"
echo "  2. Compare new results vs old baseline (72.01%)"
echo "  3. Document improvement in paper"
echo ""
