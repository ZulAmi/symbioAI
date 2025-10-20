#!/bin/bash
# Quick Start Script for Causal-DER Experiments

echo "========================================"
echo "Causal-DER Experimental Pipeline"
echo "========================================"
echo ""

# Check Python version
python3 --version

# Navigate to directory
cd "$(dirname "$0")"

echo ""
echo "Choose an option:"
echo "  1) Quick test (1 seed, ~30 minutes)"
echo "  2) Full validation (5 seeds, ~3-4 hours)"
echo "  3) Baseline only (DER++)"
echo "  4) Causal-DER only"
echo "  5) Ablations only"
echo "  6) Statistical analysis only"
echo ""

read -p "Enter choice [1-6]: " choice

case $choice in
    1)
        echo "Running quick test..."
        python3 run_full_pipeline.py --quick
        ;;
    2)
        echo "Running full validation..."
        python3 run_full_pipeline.py
        ;;
    3)
        echo "Running DER++ baseline..."
        python3 run_baseline_validation.py
        ;;
    4)
        echo "Running Causal-DER experiments..."
        python3 run_causal_der_benchmark.py --num_runs 5
        ;;
    5)
        echo "Running ablation studies..."
        python3 run_ablations.py
        ;;
    6)
        echo "Running statistical analysis..."
        python3 statistical_analysis.py
        ;;
    *)
        echo "Invalid choice. Exiting."
        exit 1
        ;;
esac

echo ""
echo "========================================"
echo "Done! Check validation/results/"
echo "========================================"
