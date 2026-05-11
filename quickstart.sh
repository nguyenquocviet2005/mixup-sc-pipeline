#!/bin/bash

# Quick start script for running the pipeline

set -e

echo "=== Mixup SC Pipeline - Quick Start ==="
echo ""

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Step 1: Environment setup
echo -e "${BLUE}Step 1: Setting up environment...${NC}"
if [ ! -d "venv" ]; then
    python -m venv venv
fi

source venv/bin/activate

echo -e "${BLUE}Step 2: Installing dependencies...${NC}"
pip install -r requirements.txt -q

# Step 3: Create directories
echo -e "${BLUE}Step 3: Creating directories...${NC}"
mkdir -p data
mkdir -p checkpoints
mkdir -p logs

# Step 4: Run experiments
echo -e "${BLUE}Step 4: Running experiments...${NC}"
echo ""

# Run baseline
echo -e "${GREEN}Running: Standard (Baseline) on CIFAR-10${NC}"
python scripts/main.py \
    --config experiments/configs/standard_cifar10_resnet50.yaml \
    --exp-name quickstart_standard

echo ""

# Run mixup
echo -e "${GREEN}Running: Mixup on CIFAR-10${NC}"
python scripts/main.py \
    --config experiments/configs/mixup_cifar10_resnet50.yaml \
    --exp-name quickstart_mixup

echo ""
echo -e "${GREEN}✓ Quickstart complete!${NC}"
echo ""
echo "Next steps:"
echo "1. Check W&B dashboard: https://wandb.ai/your-username/mixup-sc"
echo "2. Compare metrics (AUROC, AURC, E-AURC)"
echo "3. Review detailed design: cat DESIGN.md"
echo "4. Try custom experiments: python scripts/main.py --help"
