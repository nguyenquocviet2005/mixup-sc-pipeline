# MedMNIST Quick Reference

## Installation

```bash
pip install medmnist
# Or update requirements
pip install -r requirements.txt
```

## Basic Usage

### 1. List Available Datasets

```python
from data import MEDMNIST_DATASETS

for name, info in MEDMNIST_DATASETS.items():
    print(f"{name}: {info['num_classes']} classes - {info['info']}")
```

### 2. Train on MedMNIST

```bash
# Train on PathMNIST with Mixup
python scripts/main.py --dataset pathmnist --method mixup --epochs 100

# Train on DermaMNIST with ResNet-50
python scripts/main.py --dataset dermamnist --arch resnet50 --method mixup --epochs 100

# Train on ChestMNIST with custom batch size
python scripts/main.py --dataset chestmnist --batch-size 64 --method mixup --epochs 100
```

### 3. Evaluate Post-hoc Methods

```bash
# Evaluate best checkpoint on PathMNIST
python scripts/evaluate_posthoc_methods.py --dataset pathmnist

# Evaluate with specific architecture
python scripts/evaluate_posthoc_methods.py --dataset dermamnist --arch resnet50

# Custom output path
python scripts/evaluate_posthoc_methods.py --dataset chestmnist --output ./results/chest_summary.json
```

### 4. Load Data Programmatically

```python
from data import get_dataloaders

# Automatically routes to MedMNIST loader
train_loader, val_loader, test_loader, num_classes = get_dataloaders(
    dataset="pathmnist",
    data_dir="./data",
    batch_size=128,
    augmentation=True,
)

print(f"PathMNIST: {num_classes} classes")
print(f"Train batches: {len(train_loader)}")
print(f"Validation batches: {len(val_loader)}")
print(f"Test batches: {len(test_loader)}")
```

## Supported Datasets (12 total)

| Name | Classes | Type |
|------|---------|------|
| pathmnist | 9 | Histopathology |
| chestmnist | 2 | Chest X-ray |
| dermamnist | 7 | Dermoscopy |
| bloodmnist | 8 | Blood cells |
| tissuesmnist | 8 | Tissues |
| organamnist | 11 | Organ (axial) |
| organcmnist | 11 | Organ (coronal) |
| organsmnist | 11 | Organ (sagittal) |
| octmnist | 4 | Ophthalmic OCT |
| pneumoniamnist | 2 | Pneumonia |
| retinamnist | 5 | Retinopathy |
| breastmnist | 2 | Breast cancer |

## Architecture Options

- resnet18 (11M params, lightweight)
- resnet34 (22M params)
- resnet50 (25M params)
- resnet101 (44M params, largest)

## Available Methods

- standard (no augmentation)
- mixup (standard mixup)
- mixup_variant1 (temperature scaling variant)
- mixup_variant2 (adaptive alpha variant)

## Data Organization

```
./data/
├── medmnist/
│   ├── pathmnist/
│   │   ├── train_images.npy
│   │   ├── train_labels.npy
│   │   ├── val_images.npy
│   │   ├── val_labels.npy
│   │   ├── test_images.npy
│   │   └── test_labels.npy
│   ├── dermamnist/
│   └── ... (other datasets)
```

Data is automatically downloaded on first use.

## Common Commands

```bash
# Train and evaluate pipeline for DermaMNIST
python scripts/main.py --dataset dermamnist --method mixup --epochs 100 --exp-name derm_mixup
python scripts/evaluate_posthoc_methods.py --dataset dermamnist --output ./results/derm_posthoc.json

# Batch process all MedMNIST datasets
for dataset in pathmnist chestmnist dermamnist bloodmnist tissuesmnist; do
  echo "Processing $dataset..."
  python scripts/main.py --dataset $dataset --method mixup --epochs 50
done

# Run robustness evaluation
python scripts/run_posthoc_robustness.py \
  --checkpoint-glob 'mixup_best_auroc_epoch_*.pt' \
  --max-checkpoints 0 \
  --output ./results/robustness_medmnist.json
```

## Key Differences from CIFAR

1. **Image size**: 28×28 (vs. 32×32 for CIFAR)
2. **Augmentation**: Lighter augmentation (no random crop)
3. **Normalization**: Grayscale or RGB depending on dataset
4. **Official splits**: Train/Val/Test splits are predefined
5. **Medical domain**: Different class distributions and diagnostic challenges

For detailed information, see `MEDMNIST_INTEGRATION.md`
