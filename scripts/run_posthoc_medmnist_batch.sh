#!/bin/bash

# Run post-hoc evaluation across all trained MedMNIST datasets

CHECKPOINT_DIR="./checkpoints"
RESULTS_DIR="./results/posthoc_medmnist"
PYTHON="$(which python3 || which python)"

# Create results directory
mkdir -p "$RESULTS_DIR"

# Array of trained datasets
datasets=("chestmnist" "bloodmnist" "dermamnist" "pathmnist")

echo "Running post-hoc robustness evaluation for MedMNIST datasets..."
echo ""

for dataset in "${datasets[@]}"; do
    echo "=========================================="
    echo "Evaluating: $dataset"
    echo "=========================================="
    
    # Find the latest checkpoint for this dataset
    latest_ckpt=$(ls -t "$CHECKPOINT_DIR"/mixup_${dataset}_resnet18_best_auroc_epoch_*.pt 2>/dev/null | head -1)
    
    if [ -z "$latest_ckpt" ]; then
        echo "Warning: No checkpoints found for $dataset, skipping..."
        continue
    fi
    
    echo "Using checkpoint: $(basename $latest_ckpt)"
    
    # Run posthoc evaluation with the latest 5 checkpoints for this dataset
    $PYTHON scripts/run_posthoc_robustness.py \
        --dataset "$dataset" \
        --checkpoint-dir "$CHECKPOINT_DIR" \
        --checkpoint-glob "mixup_${dataset}_resnet18_best_auroc_epoch_*.pt" \
        --max-checkpoints 5 \
        --batch-size 256 \
        --seed 42 \
        --output "$RESULTS_DIR/posthoc_${dataset}.json" \
        --per-run-dir "$RESULTS_DIR/runs_${dataset}"
    
    if [ $? -eq 0 ]; then
        echo "✓ Successfully evaluated $dataset"
    else
        echo "✗ Failed to evaluate $dataset"
    fi
    echo ""
done

echo "=========================================="
echo "Post-hoc evaluation complete!"
echo "Results saved to: $RESULTS_DIR"
echo "=========================================="
