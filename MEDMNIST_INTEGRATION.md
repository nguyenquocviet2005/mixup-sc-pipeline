# MedMNIST Integration Guide

## Overview

This pipeline now supports multiple MedMNIST datasets in addition to CIFAR-10 and CIFAR-100. MedMNIST is a collection of large-scale benchmark medical image datasets.

## Supported Datasets

The following MedMNIST datasets are available:

| Dataset | Classes | Description |
|---------|---------|-------------|
| **pathmnist** | 9 | Histopathological tissue images |
| **chestmnist** | 2 | Chest X-ray classification |
| **dermamnist** | 7 | Dermoscopy skin disease images |
| **bloodmnist** | 8 | Blood cell morphology images |
| **tissuesmnist** | 8 | Tissue images |
| **organamnist** | 11 | Organ (axial view) CT images |
| **organcmnist** | 11 | Organ (coronal view) CT images |
| **organsmnist** | 11 | Organ (sagittal view) CT images |
| **octmnist** | 4 | OCT ophthalmic disease classification |
| **pneumoniamnist** | 2 | Pneumonia classification from chest X-rays |
| **retinamnist** | 5 | Diabetic retinopathy severity levels |
| **breastmnist** | 2 | Breast cancer histology images |

## Installation

Install medmnist dependency:

```bash
pip install medmnist
```

Or install all requirements including medmnist:

```bash
pip install -r requirements.txt
```

## Quick Start

### Train on MedMNIST with Mixup

```bash
# Train on PathMNIST with Mixup
python scripts/main.py --dataset pathmnist --method mixup --epochs 100 --exp-name mixup_pathmnist

# Train on DermaMNIST with ResNet-50
python scripts/main.py --dataset dermamnist --arch resnet50 --method mixup --epochs 100

# Train on ChestMNIST with a specific batch size
python scripts/main.py --dataset chestmnist --method mixup --batch-size 64 --epochs 100
```

### Evaluate Post-hoc Methods on MedMNIST

```bash
# Evaluate 3 post-hoc methods on best PathMNIST checkpoint
python scripts/evaluate_posthoc_methods.py --dataset pathmnist --arch resnet18

# Evaluate with custom checkpoint
python scripts/evaluate_posthoc_methods.py --dataset dermamnist --checkpoint ./checkpoints/my_checkpoint.pt

# Custom output location
python scripts/evaluate_posthoc_methods.py --dataset chestmnist --output ./results/chest_posthoc.json
```

## Data Handling

### Official Train/Val/Test Splits

MedMNIST datasets come with official train/val/test splits. The pipeline respects these splits:

- **Training samples**: From official training split, optionally further split into train/validation
- **Validation samples**: Official validation split (or portion of training if official val is empty)
- **Test samples**: Official test split (held out for final evaluation)

### Image Properties

- **Resolution**: 28×28 pixels (grayscale) or 3-channel (varies by dataset)
- **Normalization**: Images are normalized to [0, 1] range
- **Augmentation**: Light augmentation for MedMNIST training (rotation, affine, color jitter)

### Data Directory Structure

Data will automatically be organized as:

```
./data/
├── medmnist/
│   ├── pathmnist/
│   ├── dermamnist/
│   ├── chestmnist/
│   └── ... (other datasets)
```

## Configuration

### Default Parameters for MedMNIST

The pipeline automatically adjusts some parameters for MedMNIST:

- **Batch size**: 128 (smaller images allow larger batches)
- **Learning rate**: 0.001 (default, adjustable)
- **Epochs**: 100 (typical for medical datasets)
- **Data augmentation**: Rotation (10°), affine transformation (±10%), color jitter

### Custom Configuration via YAML

Create a config file for custom MedMNIST training:

```yaml
# configs/medmnist_pathmnist.yaml
data:
  dataset: pathmnist
  batch_size: 64
  augmentation: true

model:
  arch: resnet18
  num_classes: 9

method:
  name: mixup
  mixup_alpha: 1.0
  prob: 1.0

training:
  epochs: 100
  learning_rate: 0.001
  seed: 42
```

Then run with:

```bash
python scripts/main.py --config configs/medmnist_pathmnist.yaml
```

## Post-hoc Methods on MedMNIST

All 3 post-hoc evaluation methods work seamlessly with MedMNIST datasets:

### Method 1: Class-wise Temperature Scaling
Optimizes per-class logit temperatures on validation NLL.
- Works well for multi-class medical datasets (e.g., PathMNIST with 9 classes)

### Method 2: Feature-kNN / Logit Probability Blending
Uses penultimate layer features and k-nearest neighbors in feature space.
- Particularly useful for datasets with learned rich feature representations
- Grid-searches optimal k and blending parameter α

### Method 3: Prototype Conformal Confidence
Uses class prototype distances in feature space.
- Beneficial for distinguishing in-distribution vs out-of-distribution predictions

See `CALIBRATION_INTEGRATION.md` for detailed method descriptions.

## Architecture Compatibility

MedMNIST supports all ResNet architectures:

- **ResNet-18** (lightweight, good for small datasets)
- **ResNet-34** (balanced)
- **ResNet-50** (larger capacity, for larger datasets)
- **ResNet-101** (highest capacity)

Note: The standard ResNet architectures are adapted for 28×28 MedMNIST images (vs. 32×32 CIFAR).

## Selective Classification on Medical Data

MedMNIST evaluation uses the same selective classification (SC) metrics as CIFAR:

- **AUROC**: Confidence-based ranking of correct vs incorrect predictions
- **AURC**: Area under Risk-Coverage curve (lower is better)
- **E-AURC**: Oracle-normalized AURC gap (accounts for classifier difficulty)

### Why SC Matters for Medical Imaging

In medical applications, **abstention is valuable** — the model should express uncertainty on difficult cases rather than make confident wrong predictions. Selective classification measures this capability.

## Example: Full Pipeline on DermaMNIST

```bash
# 1. Train Mixup variant 2 on DermaMNIST
python scripts/main.py \
  --dataset dermamnist \
  --method mixup_variant2 \
  --arch resnet50 \
  --epochs 100 \
  --batch-size 64 \
  --learning-rate 0.001 \
  --exp-name mixup_variant2_dermamnist

# 2. Evaluate 3 post-hoc methods
python scripts/evaluate_posthoc_methods.py \
  --dataset dermamnist \
  --arch resnet50 \
  --output ./results/dermamnist_posthoc_summary.json

# 3. Review results
cat ./results/dermamnist_posthoc_summary.json
```

## Troubleshooting

### MedMNIST Download Issues

If you encounter download issues:

1. **Check internet connection**: MedMNIST datasets are downloaded on first use
2. **Manual download**: Visit [MedMNIST GitHub](https://github.com/MedMNIST/MedMNIST) for manual download options
3. **Cache directory**: By default, data is cached in `./data/medmnist/`

### Memory Issues

If you run out of memory:

1. **Reduce batch size**: `--batch-size 32` (default is 128)
2. **Smaller architecture**: Use ResNet-18 instead of ResNet-50
3. **Reduce num_workers**: Set `num_workers=0` in code if needed

### Normalization Questions

MedMNIST images come pre-normalized to [0, 1]. The pipeline applies standard normalization (mean=0.5, std=0.5) for training stability.

For grayscale datasets (single channel), augmentation may convert to 3-channel. This is handled automatically by the transforms pipeline.

## Model Size Considerations

| Model | Parameters | Memory (MB) |
|-------|-----------|---------|
| ResNet-18 | 11M | ~50 |
| ResNet-34 | 22M | ~100 |
| ResNet-50 | 25M | ~150 |
| ResNet-101 | 44M | ~250 |

## Recommended Configurations by Dataset

### Small Datasets (< 10k samples)
- **Architecture**: ResNet-18
- **Batch size**: 32-64
- **Learning rate**: 0.001
- **Epochs**: 50-100

### Medium Datasets (10k-100k samples)
- **Architecture**: ResNet-34 or ResNet-50
- **Batch size**: 64-128
- **Learning rate**: 0.001
- **Epochs**: 100-200

### Large Datasets (> 100k samples)
- **Architecture**: ResNet-50 or ResNet-101
- **Batch size**: 128-256
- **Learning rate**: 0.001
- **Epochs**: 100-200

## Research Applications

MedMNIST support enables research on:

1. **Medical Image Classification**: Benchmarking SC methods on real medical data
2. **Domain Adaptation**: Training on one MedMNIST dataset, evaluating on another
3. **Uncertainty Quantification**: Studying how models express uncertainty in medical contexts
4. **Interpretability**: Analyzing what features drive selective predictions in medical imaging

## References

- MedMNIST GitHub: https://github.com/MedMNIST/MedMNIST
- MedMNIST Paper: https://arxiv.org/abs/2010.14925
- Selective Classification Background: See CALIBRATION_INTEGRATION.md

## Citing MedMNIST

If you use MedMNIST datasets, please cite:

```bibtex
@article{yang2021medmnist,
  title={MedMNIST v2-a large-scale lightweight benchmark for 2d and 3d biomedical image classification},
  author={Yang, Jiancheng and Shi, Rui and Wei, Dong and Liu, Zhenguo and Ye, Liyong},
  journal={Scientific Data},
  year={2023}
}
```
