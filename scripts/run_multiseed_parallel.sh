#!/bin/bash
#
# Run multi-seed analysis in parallel across multiple processes.
#
# Usage:
#   ./scripts/run_multiseed_parallel.sh --trials 100 --workers 4
#   ./scripts/run_multiseed_parallel.sh --trials 100 --workers 4 --epochs 3
#
# This will:
#   1. Run null distribution trials in parallel (split across workers)
#   2. Merge null results
#   3. Run multi-seed trials in parallel (split across workers)  
#   4. Merge multi-seed results with p-values from null
#   5. Clean up temporary directories
#

set -e

# Default values
TRIALS=100
WORKERS=4
EPOCHS=1
SEED_START=42
OUTPUT_NULL="results/null_distribution"
OUTPUT_MULTISEED="results/multiseed"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --trials)
            TRIALS="$2"
            shift 2
            ;;
        --workers)
            WORKERS="$2"
            shift 2
            ;;
        --epochs)
            EPOCHS="$2"
            shift 2
            ;;
        --seed-start)
            SEED_START="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 --trials N --workers N [--epochs N] [--seed-start N]"
            exit 1
            ;;
    esac
done

cd "$(dirname "$0")/.."

echo "=============================================="
echo "Parallel Multi-seed Analysis"
echo "=============================================="
echo "Total trials: $TRIALS"
echo "Workers: $WORKERS"
echo "Epochs: $EPOCHS"
echo "Seed start: $SEED_START"
echo "Start time: $(date)"
echo ""

# Calculate trials per worker
TRIALS_PER_WORKER=$((TRIALS / WORKERS))
REMAINDER=$((TRIALS % WORKERS))

# Function to run parallel trials
# Arguments: MODE ("null" or "multiseed"), OUTPUT_BASE
# Sets global: RESULT_PART_DIRS (array of part directories)
run_parallel_trials() {
    local MODE=$1
    local OUTPUT_BASE=$2
    
    echo "----------------------------------------------"
    echo "Running $MODE trials in parallel..."
    echo "----------------------------------------------"
    
    local PIDS=()
    RESULT_PART_DIRS=()
    
    local CURRENT_SEED=$SEED_START
    
    for ((i=0; i<WORKERS; i++)); do
        # Distribute remainder across first workers
        local WORKER_TRIALS=$TRIALS_PER_WORKER
        if [ $i -lt $REMAINDER ]; then
            WORKER_TRIALS=$((TRIALS_PER_WORKER + 1))
        fi
        
        local PART_DIR="${OUTPUT_BASE}_part$((i+1))"
        RESULT_PART_DIRS+=("$PART_DIR")
        
        echo "  Worker $((i+1)): $WORKER_TRIALS trials (seeds $CURRENT_SEED-$((CURRENT_SEED + WORKER_TRIALS - 1))) -> $PART_DIR"
        
        mkdir -p "$PART_DIR"
        
        if [ "$MODE" == "null" ]; then
            python scripts/semantic_change_multiseed.py \
                --trials "$WORKER_TRIALS" \
                --permute-periods \
                --output-dir "$PART_DIR" \
                --seed-start "$CURRENT_SEED" \
                --epochs "$EPOCHS" \
                --checkpoint-interval "$WORKER_TRIALS" \
                --skip-csv \
                > "${PART_DIR}.log" 2>&1 &
        else
            python scripts/semantic_change_multiseed.py \
                --trials "$WORKER_TRIALS" \
                --output-dir "$PART_DIR" \
                --seed-start "$CURRENT_SEED" \
                --epochs "$EPOCHS" \
                --checkpoint-interval "$WORKER_TRIALS" \
                --skip-csv \
                > "${PART_DIR}.log" 2>&1 &
        fi
        
        PIDS+=($!)
        CURRENT_SEED=$((CURRENT_SEED + WORKER_TRIALS))
    done
    
    echo ""
    echo "  Waiting for workers to complete..."
    echo "  Monitor progress: tail -f ${OUTPUT_BASE}_part*.log"
    echo ""
    
    # Wait for all workers
    local FAILED=0
    for i in "${!PIDS[@]}"; do
        if ! wait "${PIDS[$i]}"; then
            echo "  ERROR: Worker $((i+1)) failed! Check ${RESULT_PART_DIRS[$i]}.log"
            FAILED=1
        fi
    done
    
    if [ $FAILED -eq 1 ]; then
        echo "Some workers failed. Aborting."
        exit 1
    fi
    
    echo "  All workers completed."
    echo ""
}

# Step 1: Run null distribution in parallel
echo ""
echo "Step 1/4: Running null distribution trials..."
run_parallel_trials "null" "$OUTPUT_NULL"
NULL_PART_DIRS=("${RESULT_PART_DIRS[@]}")

# Step 2: Merge null results
echo "Step 2/4: Merging null distribution results..."
python scripts/merge_multiseed_results.py "${NULL_PART_DIRS[@]}" --output "$OUTPUT_NULL" --is-null

echo ""
echo "Step 3/4: Running multi-seed trials..."
run_parallel_trials "multiseed" "$OUTPUT_MULTISEED"
MULTISEED_PART_DIRS=("${RESULT_PART_DIRS[@]}")

# Step 4: Merge multiseed results with null distribution
echo "Step 4/4: Merging multi-seed results..."
python scripts/merge_multiseed_results.py "${MULTISEED_PART_DIRS[@]}" --output "$OUTPUT_MULTISEED" --null-dir "$OUTPUT_NULL"

# Cleanup temporary directories and logs
echo ""
echo "Cleaning up temporary files..."
for PART_DIR in "${NULL_PART_DIRS[@]}"; do
    rm -rf "$PART_DIR" "${PART_DIR}.log"
done
for PART_DIR in "${MULTISEED_PART_DIRS[@]}"; do
    rm -rf "$PART_DIR" "${PART_DIR}.log"
done

echo ""
echo "=============================================="
echo "Analysis complete!"
echo "End time: $(date)"
echo ""
echo "Results saved to:"
echo "  - $OUTPUT_NULL/"
echo "  - $OUTPUT_MULTISEED/"
echo "=============================================="
