# MedMNIST Configuration Examples

This directory contains example YAML configurations for training on different MedMNIST datasets.

## Usage

```bash
python scripts/main.py --config experiments/configs/medmnist_pathmnist.yaml
```

## Example Configurations

### PathMNIST (Histopathology - 9 classes)
- Best for: Testing SC on multi-class medical histology
- Recommended: ResNet-34 or ResNet-50
- Epochs: 100-150

### DermaMNIST (Dermoscopy - 7 classes)
- Best for: Skin lesion classification
- Recommended: ResNet-18 or ResNet-34
- Epochs: 100

### ChestMNIST (Chest X-ray - 2 classes)
- Best for: Binary classification with class imbalance
- Recommended: ResNet-18 (lightweight for binary)
- Epochs: 50-100

### OrganMNIST variants (11 classes each)
- Best for: Multi-organ multi-view classification
- Recommended: ResNet-50 (more capacity needed)
- Epochs: 150-200

## Dataset-Specific Tips

1. **Small datasets** (ChestMNIST, PneumoniaMNIST, BreastMNIST):
   - Use ResNet-18
   - Reduce batch size to 64
   - Use early stopping

2. **Medium datasets** (PathMNIST, DermaMNIST, etc.):
   - Use ResNet-34
   - Standard batch size 128
   - 100-150 epochs

3. **Larger datasets** (OrganMNIST variants):
   - Use ResNet-50
   - Can increase batch size to 256
   - 150-200 epochs

## Memory Requirements

- ResNet-18 + 128 batch: ~2GB VRAM
- ResNet-50 + 128 batch: ~3GB VRAM
- ResNet-101 + 128 batch: ~5GB VRAM

See MEDMNIST_INTEGRATION.md for full configuration details.
