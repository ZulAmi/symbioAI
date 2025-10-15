#!/bin/bash
#
# Quick Benchmark Suite - Essential Experiments Only
# Perfect for initial testing and quick university presentations
#
# Runs: CIFAR-10, CIFAR-100, TinyImageNet with Optimized strategy
# Estimated time: 1-2 hours

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${BLUE}╔════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║   Quick Benchmark Suite (Essential Only)      ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "${GREEN}Running essential benchmarks for university presentation:${NC}"
echo "  1. CIFAR-10 (optimized) - Standard benchmark"
echo "  2. CIFAR-100 (optimized) - Most cited CL benchmark"
echo "  3. TinyImageNet (optimized) - Production scale"
echo ""
echo -e "${YELLOW}Estimated time: 1-2 hours${NC}"
echo ""

RESULTS_DIR="./results/quick_benchmarks_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RESULTS_DIR"

START_TIME=$(date +%s)

# CIFAR-10
echo -e "${BLUE}[1/3] Running CIFAR-10 (optimized)...${NC}"
python3 industry_standard_benchmarks.py \
    --dataset cifar10 \
    --strategy optimized \
    2>&1 | tee "$RESULTS_DIR/cifar10_optimized.log"

# CIFAR-100
echo ""
echo -e "${BLUE}[2/3] Running CIFAR-100 (optimized)...${NC}"
python3 industry_standard_benchmarks.py \
    --dataset cifar100 \
    --strategy optimized \
    2>&1 | tee "$RESULTS_DIR/cifar100_optimized.log"

# TinyImageNet
echo ""
echo -e "${BLUE}[3/3] Running TinyImageNet (optimized)...${NC}"
python3 industry_standard_benchmarks.py \
    --dataset tiny_imagenet \
    --strategy optimized \
    2>&1 | tee "$RESULTS_DIR/tiny_imagenet_optimized.log"

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
MINUTES=$((DURATION / 60))
SECONDS=$((DURATION % 60))

echo ""
echo -e "${GREEN}✅ Quick benchmarks complete!${NC}"
echo "  Duration: ${MINUTES}m ${SECONDS}s"
echo "  Results: $RESULTS_DIR"
echo ""
echo -e "${YELLOW}Next: Analyze results and compare to published papers${NC}"
