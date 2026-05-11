# Mixup for Selective Classification: Experimental Pipeline

A modular, production-ready PyTorch pipeline for testing Mixup improvements in Selective Classification (SC).

## Design Overview

### Architecture

The codebase follows a **modular, layered architecture** designed for extensibility:

```
mixup-sc-pipeline/
├── data/              # Data loading & preprocessing
│   └── loaders.py     # CIFAR-10/100 with train/val/test splits
├── models/            # Neural network architectures  
│   └── resnet.py      # ResNet-18/34/50/101
├── methods/           # Training algorithms
│   ├── base.py        # Base training method
│   └── mixup.py       # Standard Mixup + Variants (V1, V2)
├── training/          # Main training loop
│   └── trainer.py     # Trainer class with full logging
├── evaluation/        # SC-specific metrics
│   └── metrics.py     # AUROC, AURC, E-AURC computation
├── utils/             # Configuration, logging, utilities
│   ├── config.py      # Dataclass-based config system
│   ├── device.py      # GPU/CPU management
│   └── logging.py     # W&B & TensorBoard integration
├── experiments/       # Experiment configs & results
│   └── configs/       # YAML config files for reproducibility
└── scripts/
    └── main.py        # Entry point
```

### Key Design Principles

1. **Separation of Concerns**: Each module has a single responsibility
2. **Configurability**: YAML-based experiment configs for reproducibility
3. **Extensibility**: Easy to add new methods, models, or datasets
4. **Logging**: Integrated W&B + TensorBoard for experiment tracking
5. **SC-Focused Metrics**: Full suite of selective classification metrics

---

## Selective Classification Metrics Explained

### AUROC (Area Under ROC Curve)
Measures the model's ability to rank correct predictions with higher confidence than incorrect ones.
- **Range**: [0, 1]
- **Better**: Higher is better (1.0 = perfect))
- **Interpretation**: Probability that confidence is higher for correct than incorrect predictions

### AURC (Area Under Risk-Coverage Curve)
As coverage increases, the risk (error rate) decreases. AURC measures the area under this curve.
- **Range**: [0, 1]
- **Better**: Lower is better (0.0 = no error))
- **Interpretation**: Average error rate across all coverage levels
- **Formula**: Integrate risk from 0% to 100% coverage

### E-AURC (Excess AURC)
Normalized AURC: the difference between actual and optimal (oracle) classifier.
- **Range**: [0, ∞)
- **Better**: Lower is better (0.0 = optimal)
- **Interpretation**: How much the model underperforms an oracle classifier
- **Formula**: E-AURC = AURC - AURC_optimal

**SC Task**: Select confidence threshold to maximize coverage while keeping error rate low.


## Training Methods

### Standard Training
Baseline without augmentation or regularization.

### Mixup
- $\lambda \sim \text{Beta}(\alpha, \alpha)$

### Mixup Variant 1: Temperature Scaling
**Effect**: Lower T → sharper/more confident predictions → higher AUROC, potentially higher risk

### Mixup Variant 2: Adaptive Alpha
**Effect**: More aggressive mixing when model is less confident → improved calibration

---
### Installation

```bash
cd mixup-sc-pipeline
pip install -r requirements.txt

### Setup W&B (Optional)

```bash
wandb login  # Paste your API key
```

### Run from Config

```bash
python scripts/main.py --config experiments/configs/standard_cifar10_resnet50.yaml
```

### Run with Command-Line Args

```bash
python scripts/main.py \
    --dataset cifar10 \
    --method mixup \
    --epochs 200 \
    --batch-size 128 \
    --learning-rate 0.1 \
    --exp-name my_experiment
```

### Available Options

```bash
python scripts/main.py --help
```

**Key Arguments**:
- `--dataset`: `cifar10` or `cifar100`
- `--method`: `standard`, `mixup`, `mixup_variant1`, `mixup_variant2`
- `--config`: Path to YAML config file
- `--epochs`: Number of training epochs
- `--batch-size`: Training batch size
- `--learning-rate`: Learning rate
- `--exp-name`: Experiment identifier
- `--no-wandb`: Disable W&B logging

### Tiny-ImageNet training

Tiny-ImageNet is now supported as `--dataset tinyimagenet`.

Expected data layout:

- `./data/tiny-imagenet-200/train/`
- `./data/tiny-imagenet-200/val/images/`
- `./data/tiny-imagenet-200/val/val_annotations.txt`

The loader will reorganize the validation images into class folders on first use.

Example:

```bash
source /home/viet2005/workspace/Research/calibrated-selective-classification/venv/bin/activate
python scripts/main.py \
  --dataset tinyimagenet \
  --method mixup \
  --arch resnet18 \
  --epochs 100 \
  --batch-size 128 \
  --learning-rate 0.1 \
  --exp-name mixup_tinyimagenet_resnet18 \
  --no-wandb
```

Notes:

- Tiny-ImageNet has 200 classes.
- `resnet18` is the safest starting point.
- The framework uses the labeled official validation set as the held-out test split.

---

## Example Experiments

### Experiment 1: Standard vs. Mixup on CIFAR-10

```bash
# Baseline
python scripts/main.py --dataset cifar10 --method standard \
    --exp-name exp1_standard

# Mixup
python scripts/main.py --dataset cifar10 --method mixup \
    --exp-name exp1_mixup
```

Compare AUROC, AURC, and E-AURC in W&B dashboard.

### Experiment 2: Variant Comparison

```bash
for variant in standard mixup mixup_variant1 mixup_variant2; do
    python scripts/main.py --dataset cifar10 --method $variant \
        --exp-name exp2_$variant
done
```

### Experiment 3: CIFAR-100 Study

```bash
python scripts/main.py --dataset cifar100 --method mixup \
    --exp-name exp3_cifar100
```

---

## Configuration Examples

### Create Custom Config

Edit `experiments/configs/my_experiment.yaml`:

```yaml
data:
  dataset: cifar10
  batch_size: 256  # Larger batch size
  augmentation: true

model:
  arch: resnet50
  num_classes: 10

method:
  name: mixup
  mixup_alpha: 0.5  # Less aggressive mixing
  prob: 0.5         # Only 50% of batches get mixed

training:
  epochs: 100       # Shorter training
  learning_rate: 0.05
  lr_schedule: cosine

logging:
  use_wandb: true
  project_name: my_project
  experiment_name: my_custom_exp

evaluation:
  coverage_levels: [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
```

Then run:
```bash
python scripts/main.py --config experiments/configs/my_experiment.yaml
```

---

## Code Structure: Key Classes

### `Trainer` (`training/trainer.py`)
Main training orchestrator:
- `train_epoch()`: Single training epoch with forward pass, loss, and SC metrics
- `validate_epoch()`: Validation loop
- `test_epoch()`: Final test evaluation
- `train()`: Complete training loop with logging and checkpointing

### `SelectionMetrics` (`evaluation/metrics.py`)
SC metric computations:
- `compute_auroc()`: AUROC from confidences & correctness
- `compute_aurc()`: AURC across coverage levels
- `compute_eaurc()`: Excess AURC (normalized)
- `compute_all_metrics()`: All metrics at once
- `compute_confidence_distribution()`: Confidence stats

### `MixupMethod` (`methods/mixup.py`)
Mixup implementation:
- `mixup_batch()`: Apply mixup to input/targets
- `forward_and_loss()`: Forward pass with optional mixup

### `Config` (`utils/config.py`)
Configuration management:
- Dataclass-based (type-safe, IDE support)
- Can load/save YAML
- Hierarchical: DataConfig, ModelConfig, MethodConfig, etc.

---

## Logging & Experiment Tracking

### W&B Integration

Automatic logging of:
- **Metrics**: Loss, accuracy, AUROC, AURC, E-AURC every epoch
- **Hyperparameters**: Full config saved
- **Distributions**: Logits/confidence histograms
- **Checkpoints**: Best model saved to W&B

View results: https://wandb.ai/your-username/mixup-sc

### TensorBoard

Optional TensorBoard logging (set `use_tensorboard: true` in config):

```bash
tensorboard --logdir=./runs
# Visit http://localhost:6006
```

---

## Adding New Methods

1. Create new class in `methods/mixup.py`:

```python
class MyMixupVariant(MixupMethod):
    def forward_and_loss(self, inputs, targets, use_mixup=True):
        # Your custom logic
        mixed_inputs, mixed_targets, lam = self.mixup_batch(inputs, targets)
        # ... custom forward pass ...
        return logits, loss
```

2. Register in `methods/__init__.py`:

```python
def get_method(name: str, model, **kwargs):
    methods = {
        "standard": StandardMethod,
        "mixup": MixupMethod,
        "my_variant": MyMixupVariant,  # Add here
    }
    ...
```

3. Run:
```bash
python scripts/main.py --method my_variant
```

---

## Adding New Datasets

Extend `data/loaders.py`:

```python
def get_imagenet_dataloaders(...):
    # Similar to CIFAR loaders but for ImageNet
    pass

# Update get transforms, etc.
```

---

## Tips for Research

### Hyperparameter Search

Create multiple configs:
```bash
for alpha in 0.5 1.0 2.0; do
    for lr in 0.01 0.1 0.5; do
        python scripts/main.py --dataset cifar10 --method mixup \
            --exp-name grid_search_alpha${alpha}_lr${lr}
    done
done
```

### Reproducibility

- All random seeds are controlled (set in config)
- Configs are saved alongside results
- W&B tracks git commit hash

### Checkpoint Management

Models are saved to `./checkpoints/`:
- Best validation AUROC checkpoint saved automatically
- Use for resuming training or testing

---

## Troubleshooting

### CUDA Out of Memory
- Reduce `batch_size` in config
- Use smaller model (`resnet18` instead of `resnet50`)

### W&B API Errors
```bash
wandb login  # Re-authenticate
# Or set: --no-wandb flag
```

### Slow Data Loading
- Increase `num_workers` (adjust if OOM)
- Use `--no-augmentation` to check if bottleneck

---

## Performance Benchmarks

Expected accuracies on CIFAR-10 (ResNet-50, 200 epochs):
- **Standard**: ~96%
- **Mixup**: ~96.5%
- **Mixup+Temp**: ~96.2% (lower risk at some coverage)

Expected metrics:
- **AUROC**: 0.90-0.95
- **AURC**: 0.05-0.15  
- **E-AURC**: 0.01-0.05

(Exact values depend on hyperparameters and seed)

---

## References

- **Mixup**: Zhang et al., "mixup: Beyond Empirical Risk Minimization" (ICLR 2018)
- **Selective Classification**: Geifman & El-Yaniv, "Selective Prediction" (NIPS 2019)
- **AURC**: Similar to coverage-error tradeoff in OOD detection

---

## License

MIT License - Feel free to use and modify for research

---

## Questions?

Check the code: modular design means each file is self-contained and documented.
Start with `scripts/main.py` for usage examples!
