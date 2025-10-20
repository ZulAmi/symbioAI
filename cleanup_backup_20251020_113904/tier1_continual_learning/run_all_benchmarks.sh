#!/bin/bash
#
# Comprehensive Continual Learning Benchmarks
# Runs all datasets × all strategies for university collaboration research
#
# Total experiments: 7 datasets × 5 strategies = 35 experiments
# Estimated time: 
#   - Fast datasets (MNIST, Fashion-MNIST): ~5 min each
#   - Medium datasets (CIFAR-10, SVHN): ~15 min each  
#   - Large datasets (CIFAR-100, Omniglot): ~30 min each
#   - TinyImageNet: ~2-4 hours
# Total: ~8-12 hours for all experiments

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Configuration
DATASETS=("mnist" "fashion_mnist" "cifar10" "cifar100" "svhn" "omniglot" "tiny_imagenet")
STRATEGIES=("naive" "ewc" "replay" "multihead" "optimized")
SEED=42

# Results directory
RESULTS_DIR="./results/comprehensive_benchmarks_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RESULTS_DIR"

echo -e "${BLUE}╔════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║   Comprehensive Continual Learning Benchmark Suite        ║${NC}"
echo -e "${BLUE}║   For University Collaboration Research                   ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "${GREEN}Configuration:${NC}"
echo "  Datasets: ${#DATASETS[@]} (${DATASETS[*]})"
echo "  Strategies: ${#STRATEGIES[@]} (${STRATEGIES[*]})"
echo "  Total experiments: $((${#DATASETS[@]} * ${#STRATEGIES[@]}))"
echo "  Seed: $SEED"
echo "  Results directory: $RESULTS_DIR"
echo ""

# Create summary file
SUMMARY_FILE="$RESULTS_DIR/benchmark_summary.txt"
echo "Comprehensive Continual Learning Benchmarks" > "$SUMMARY_FILE"
echo "Started: $(date)" >> "$SUMMARY_FILE"
echo "========================================" >> "$SUMMARY_FILE"
echo "" >> "$SUMMARY_FILE"

# Track progress
TOTAL_EXPERIMENTS=$((${#DATASETS[@]} * ${#STRATEGIES[@]}))
CURRENT_EXPERIMENT=0
SUCCESSFUL=0
FAILED=0

START_TIME=$(date +%s)

# Function to run single experiment
run_experiment() {
    local dataset=$1
    local strategy=$2
    local exp_num=$3
    local total=$4
    
    echo ""
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${GREEN}Experiment $exp_num/$total${NC}"
    echo -e "${YELLOW}Dataset:${NC} $dataset"
    echo -e "${YELLOW}Strategy:${NC} $strategy"
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    
    # Log to summary
    echo "[$exp_num/$total] Dataset: $dataset, Strategy: $strategy" >> "$SUMMARY_FILE"
    
    # Run benchmark
    EXP_START=$(date +%s)
    
    if python3 industry_standard_benchmarks.py \
        --dataset "$dataset" \
        --strategy "$strategy" \
        2>&1 | tee "$RESULTS_DIR/${dataset}_${strategy}.log"; then
        
        EXP_END=$(date +%s)
        EXP_DURATION=$((EXP_END - EXP_START))
        
        echo -e "${GREEN}✅ Success${NC} (${EXP_DURATION}s)"
        echo "  Status: SUCCESS (${EXP_DURATION}s)" >> "$SUMMARY_FILE"
        ((SUCCESSFUL++))
    else
        EXP_END=$(date +%s)
        EXP_DURATION=$((EXP_END - EXP_START))
        
        echo -e "${RED}❌ Failed${NC} (${EXP_DURATION}s)"
        echo "  Status: FAILED (${EXP_DURATION}s)" >> "$SUMMARY_FILE"
        ((FAILED++))
    fi
    
    echo "" >> "$SUMMARY_FILE"
}

# Run all experiments
echo -e "${YELLOW}Starting benchmark suite...${NC}"
echo ""

for dataset in "${DATASETS[@]}"; do
    for strategy in "${STRATEGIES[@]}"; do
        ((CURRENT_EXPERIMENT++))
        run_experiment "$dataset" "$strategy" "$CURRENT_EXPERIMENT" "$TOTAL_EXPERIMENTS"
        
        # Progress update
        PROGRESS=$((CURRENT_EXPERIMENT * 100 / TOTAL_EXPERIMENTS))
        echo -e "${BLUE}Progress: $CURRENT_EXPERIMENT/$TOTAL_EXPERIMENTS ($PROGRESS%)${NC}"
        echo -e "${GREEN}Successful: $SUCCESSFUL${NC} | ${RED}Failed: $FAILED${NC}"
    done
done

END_TIME=$(date +%s)
TOTAL_DURATION=$((END_TIME - START_TIME))
HOURS=$((TOTAL_DURATION / 3600))
MINUTES=$(((TOTAL_DURATION % 3600) / 60))
SECONDS=$((TOTAL_DURATION % 60))

# Final summary
echo ""
echo -e "${BLUE}╔════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║                  BENCHMARK COMPLETE                        ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "${GREEN}Summary:${NC}"
echo "  Total experiments: $TOTAL_EXPERIMENTS"
echo "  Successful: $SUCCESSFUL"
echo "  Failed: $FAILED"
echo "  Duration: ${HOURS}h ${MINUTES}m ${SECONDS}s"
echo ""
echo -e "${YELLOW}Results saved to:${NC} $RESULTS_DIR"
echo -e "${YELLOW}Summary file:${NC} $SUMMARY_FILE"
echo -e "${YELLOW}Individual logs:${NC} $RESULTS_DIR/*.log"
echo -e "${YELLOW}Result JSONs:${NC} validation/results/*.json"
echo ""

# Write final summary to file
echo "========================================" >> "$SUMMARY_FILE"
echo "Completed: $(date)" >> "$SUMMARY_FILE"
echo "Total Duration: ${HOURS}h ${MINUTES}m ${SECONDS}s" >> "$SUMMARY_FILE"
echo "Successful: $SUCCESSFUL" >> "$SUMMARY_FILE"
echo "Failed: $FAILED" >> "$SUMMARY_FILE"
echo "" >> "$SUMMARY_FILE"

if [ $FAILED -eq 0 ]; then
    echo -e "${GREEN}🎉 All experiments completed successfully!${NC}"
    echo ""
    echo -e "${YELLOW}Next steps for university collaboration:${NC}"
    echo "  1. Analyze results: python3 analyze_results.py"
    echo "  2. Generate comparison table"
    echo "  3. Create visualizations for paper"
    echo "  4. Write preliminary results document"
else
    echo -e "${YELLOW}⚠️  Some experiments failed. Check logs in:${NC}"
    echo "  $RESULTS_DIR"
fi

echo ""
echo -e "${BLUE}Ready for university collaboration! 🎓${NC}"
