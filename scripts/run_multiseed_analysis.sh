#!/bin/bash
#
# Run full multi-seed stability and significance analysis.
# This script runs 100 null distribution trials followed by 100 multi-seed trials.
# Expected runtime: ~10-12 hours (depending on hardware)
#
# Usage:
#   ./scripts/run_multiseed_analysis.sh
#   nohup ./scripts/run_multiseed_analysis.sh > multiseed_analysis.log 2>&1 &
#

set -e  # Exit on error

cd "$(dirname "$0")/.."  # Change to project root

echo "=============================================="
echo "Multi-seed Stability and Significance Analysis"
echo "=============================================="
echo "Start time: $(date)"
echo ""

# Step 1: Build null distribution
echo "Step 1/2: Building null distribution (100 trials, permuted labels)..."
echo "----------------------------------------------"
python scripts/semantic_change_multiseed.py \
    --trials 100 \
    --permute-periods \
    --output-dir results/null_distribution \
    --checkpoint-interval 5

echo ""
echo "Null distribution complete: $(date)"
echo ""

# Step 2: Run multi-seed analysis
echo "Step 2/2: Running multi-seed analysis (100 trials, real labels)..."
echo "----------------------------------------------"
python scripts/semantic_change_multiseed.py \
    --trials 100 \
    --output-dir results/multiseed \
    --null-dir results/null_distribution \
    --checkpoint-interval 5

echo ""
echo "=============================================="
echo "Analysis complete!"
echo "End time: $(date)"
echo ""
echo "Results saved to:"
echo "  - results/null_distribution/"
echo "  - results/multiseed/"
echo "=============================================="
