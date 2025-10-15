#!/bin/bash

# Quick MNIST test to validate all fixes
# Should complete in ~5 minutes

cd "$(dirname "$0")"

echo "=================================="
echo "Running MNIST Benchmark Test"
echo "=================================="
echo "Start time: $(date)"
echo ""

# Run in background and save output
nohup python3 industry_standard_benchmarks.py --dataset mnist --strategy optimized > mnist_test.log 2>&1 &
PID=$!

echo "Benchmark running with PID: $PID"
echo "Monitor progress with:"
echo "  tail -f mnist_test.log"
echo ""
echo "Check status with:"
echo "  python3 check_status.py"
echo ""
echo "Expected completion: ~5 minutes"
