# Mixup SC Pipeline: Design & Implementation Guide

## Executive Summary

A **modular, production-ready PyTorch pipeline** for evaluating Mixup improvements in Selective Classification with:
- ✅ CIFAR-10/100 support  
- ✅ Modular methods (Standard, Mixup, Variants)
- ✅ Full SC metrics (AUROC, AURC, E-AURC)
- ✅ W&B/TensorBoard logging
- ✅ YAML config-driven experiments
- ✅ Extensible design for new methods/datasets

---

## Architecture & Modularity

### Layer 1: Data (`data/loaders.py`)
**Responsibility**: Handle data loading, augmentation, splits

```python
train_loader, val_loader, test_loader, num_classes = get_cifar_dataloaders(
    dataset="cifar10",
    batch_size=128,
    augmentation=True,
    val_split=0.1,
    seed=42
)
```

**Key Features**:
- Automatic train/val/test split
- Configurable augmentation (AutoAugment ready)
- Compatible with any torchvision Dataset

### Layer 2: Models (`models/resnet.py`)
**Responsibility**: Define network architectures

```python
model = get_model("resnet50", num_classes=10)  # 23.5M parameters
```

**Architectures**: ResNet-18, 34, 50, 101 (optimized for CIFAR)

### Layer 3: Methods (`methods/mixup.py`)
**Responsibility**: Implement training algorithms

```
BaseMethod (forward_and_loss, compute_accuracy)
├── StandardMethod (baseline)
├── MixupMethod (standard mixup)
├── MixupVariant1 (+ temperature scaling)
└── MixupVariant2 (+ adaptive alpha)
```

**Plugin System**: Easy to add new variants

### Layer 4: Training (`training/trainer.py`)
**Responsibility**: Orchestrate training, validation, logging

```python
trainer = Trainer(model, method, train_loader, val_loader, test_loader, device, config, logger)
test_metrics = trainer.train()  # Full training pipeline
```

**Key Methods**:
- `train_epoch()`: Forward/backward/metrics
- `validate_epoch()`: Evaluation loop
- `test_epoch()`: Final test
- Auto-checkpointing of best models

### Layer 5: Evaluation (`evaluation/metrics.py`)
**Responsibility**: Compute SC-specific metrics

```python
metrics = SelectionMetrics.compute_all_metrics(confidences, correctness)
# Returns: {auroc, aurc, eaurc}
```

### Layer 6: Utils (`utils/`)
**Responsibility**: Configuration, logging, device management

---

## Selective Classification Metrics (Deep Dive)

### Problem Formulation
Model outputs $f(x) = (h, c)$ where:
- $h$: prediction/hard label
- $c$: confidence (max softmax probability)

**Selective Classification Task**: Select subset $S$ to minimize error while maximizing coverage.

### AUROC: Confidence-Correctness Ranking

$$\text{AUROC} = P(c_i > c_j | \text{correct}_i = 1, \text{correct}_j = 0)$$

**Implementation**:
```python
# Correctness: 1 if prediction correct, 0 otherwise
correctness = (predictions == targets).astype(int)
confidences = probs.max(axis=1)

auroc = roc_auc_score(correctness, confidences)
```

**Interpretation**: Higher confidence for correct predictions → higher AUROC

### AURC: Risk-Coverage Tradeoff

For each coverage level $\phi \in [0, 1]$ (fraction of data rejected):
1. Select top $\phi \cdot n$ samples by confidence
2. Compute risk (error rate) on selected samples
3. Plot risk vs. coverage → integrate under curve

$$\text{AURC} = \int_0^1 \text{Risk}(\phi) \, d\phi$$

**Implementation**:
```python
# Sort by confidence (descending)
sorted_by_conf = np.argsort(-confidences)

risks = []
for coverage_level in [0, 10, 20, ..., 100]:
    n_select = int(len(data) * coverage_level / 100)
    selected = correctness[sorted_by_conf[:n_select]]
    risk = 1 - mean(selected)  # Error rate
    risks.append(risk)

aurc = auc(coverage_levels, risks)
```

**Example**:
```
Coverage  Risk
0%        -     (undefined)
10%       0.01  (select 10% → 1% error)
50%       0.03  (select 50% → 3% error)
100%      0.05  (select 100% → 5% error)

AURC ≈ area = 0.04 (average error)
```

### E-AURC: Oracle Normalization

Compares model AURC to **oracle** (perfect confidence ranker):

$$\text{E-AURC} = \text{AURC}_{\text{model}} - \text{AURC}_{\text{oracle}}$$

**Oracle Logic**:
```python
# Sort by TRUE CORRECTNESS (not confidence!)
optimal_sorted = np.argsort(-correctness)
optimal_risks = []  # Compute same way
aurc_optimal = auc(coverage_levels, optimal_risks)

eaurc = aurc - aurc_optimal  # >= 0 always
```

**Interpretation**:
- E-AURC = 0: Perfect confidence ranker
- E-AURC = 0.05: Model 5% worse than oracle

---

## Training Loop Deep Dive

### `train_epoch()`: Training Step

```python
def train_epoch(self):
    self.model.train()
    
    for batch_idx, (inputs, targets) in enumerate(self.train_loader):
        # 1. Forward + Loss
        logits, loss = self.method.forward_and_loss(inputs, targets, use_mixup=True)
        
        # 2. Backward
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # 3. Metrics
        logits, targets → confidences, correctness (via softmax)
        auroc = roc_auc_score(correctness, confidences)
        
        # 4. Log
        wandb.log({
            'train/loss': loss,
            'train/auroc': auroc,
            'train/lr': lr
        }, step=epoch)
```

**Key Detail**: SC metrics computed on **training batch** (useful for monitoring convergence)

### `validate_epoch()`: Evaluation Step

```python
def validate_epoch(self):
    self.model.eval()
    
    with torch.no_grad():
        # 1. Collect all logits+targets
        all_logits = []
        for inputs, targets in self.val_loader:
            logits, _ = self.method.forward_and_loss(inputs, targets, use_mixup=False)
            all_logits.append(logits)
        
        # 2. Compute SC metrics on FULL validation set
        confidences, correctness = compute_metrics_from_logits(all_logits, targets)
        sc_metrics = SelectionMetrics.compute_all_metrics(
            confidences, correctness, 
            coverage_levels=[0, 10, 20, ..., 100]
        )
```

**Difference from Training**: 
- No mixup during evaluation
- Uses full dataset (not single batch) for accurate metrics
- Best model saved based on AUROC

### LR Scheduling

```python
if config.training.lr_schedule == "cosine":
    scheduler = CosineAnnealingLR(optimizer, T_max=200)  # Warm restart not used
    
# Typical LR progression:
# Epoch 0: 0.1 (starting)
# Epoch 100: 0.05 (half power, cosine peak)
# Epoch 200: ~0.0001 (near zero)
```

---

## Mixup Implementation Details

### Standard Mixup Algorithm

**Input**: Batch of $(x_i, y_i)$ pairs

```python
# 1. Sample mixing ratio
λ ~ Beta(α, α)

# 2. Sample random permutation
j ~ Perm([1..N])

# 3. Mix in INPUT space
x̃ = λ*x_i + (1-λ)*x_j

# 4. Mix TARGETS (soft labels)
ỹ = λ*y_i + (1-λ)*y_j   # One-hot → probability

# 5. Train on mixed data
ℓ = -∑_k ỹ_k log(softmax(f(x̃))_k)
```

**Implementation**:
```python
def mixup_batch(self, inputs, targets):
    batch_size = inputs.size(0)
    
    # Sample lambda
    lam = np.random.beta(self.alpha, self.alpha)
    
    # Random shuffle
    index = torch.randperm(batch_size)
    
    # Mix inputs
    mixed_x = lam * inputs + (1 - lam) * inputs[index]
    
    # Mix targets (convert to one-hot first)
    targets_onehot = F.one_hot(targets, num_classes)
    targets_mixed = lam * targets_onehot + (1 - lam) * targets_onehot[index]
    
    return mixed_x, targets_mixed
```

### Variant 1: Temperature Scaling

**Motivation**: Control confidence distribution

```python
def forward_and_loss(self, inputs, targets):
    logits = self.model(mixed_inputs)
    logits = logits / self.temperature  # Scale down logits
    
    # Higher T (e.g., 1.5) → softer (lower conf)
    # Lower T (e.g., 0.5) → sharper (higher conf)
```

**Effect on Confidence**:
- T=1.0: Normal softmax
- T=2.0: Entropy ↑, confidence ↓
- T=0.5: Entropy ↓, confidence ↑

### Variant 2: Adaptive Alpha

**Motivation**: More mixing when model is uncertain

```python
def forward_and_loss(self, inputs, targets):
    # Compute confidence on unmixed batch
    with torch.no_grad():
        logits = self.model(inputs)
        conf = softmax(logits).max()
    
    # Adapt alpha
    confidence_ratio = min(1.0, conf / 0.8)  # [0, 1]
    adaptive_α = self.base_α * (2 - confidence_ratio)  # [α, 2α]
    
    # Higher confidence → less mixing
    # Lower confidence → more mixing
```

---

## Configuration System

### Hierarchical Dataclasses

```python
@dataclass
class DataConfig:
    dataset: str = "cifar10"
    batch_size: int = 128
    
@dataclass
class Config:
    data: DataConfig
    model: ModelConfig
    method: MethodConfig
    training: TrainingConfig
    ...
```

**Advantages**:
- Type hints → IDE autocomplete
- Automatic validation
- YAML serialization built-in
- Easy to override from CLI

### YAML Example

```yaml
# experiments/configs/mixup_cifar10_resnet50.yaml
data:
  dataset: cifar10
  batch_size: 128
method:
  name: mixup
  mixup_alpha: 1.0
training:
        **sc_metrics,  # Include additional metrics
        **cal_metrics,  # ECE, MCE, Brier, NLL
  learning_rate: 0.1
```

**Load & Override**:
```python
config = Config.from_yaml("mixup_cifar10_resnet50.yaml")
config.training.epochs = 300  # Override programmatically
```

---

## Logging & Experiment Tracking

### W&B Integration

```python
logger = ExperimentLogger(
    use_wandb=True,
    project_name="mixup-sc",
    experiment_name="mixup_cifar10_resnet50",
    config_dict=config.to_dict()
)

# Log scalars
logger.log_metrics({"train/auroc": 0.95}, step=epoch)

# Log histograms
logger.log_histogram("logits_dist", logits, epoch)

# Save checkpoints
logger.log_model_checkpoint(model, "best_auroc")

# Finish
logger.finish()
```

**W&B Dashboard Shows**:
- Training curves (loss, acc, AUROC, AURC, E-AURC)
- Logits distribution evolution
- Best hyperparameters
- Model artifacts

---

## Example: Running an Experiment

### Step 1: Install

```bash
cd mixup-sc-pipeline
pip install -r requirements.txt
wandb login
```

### Step 2: Run

```bash
python scripts/main.py \
    --dataset cifar10 \
    --method mixup \
    --epochs 200 \
    --batch-size 128 \
    --exp-name exp1_mixup
```

### Step 3: Monitor

- **W&B**: https://wandb.ai/your-user/mixup-sc
- **TensorBoard**: `tensorboard --logdir=./runs`
- **Checkpoints**: `./checkpoints/exp1_mixup_best_auroc_epoch_*.pt`

### Step 4: Compare

Run multiple methods:
```bash
for method in standard mixup mixup_variant1; do
    python scripts/main.py --method $method --exp-name comparison_$method
done
```

Compare AUROC, AURC, E-AURC in W&B dashboard.

---

## Adding Your Own Research

### New Training Method

1. Extend `MixupMethod`:
```python
class MyNewMixup(MixupMethod):
    def forward_and_loss(self, inputs, targets, use_mixup=True):
        # Your magic here
        return logits, loss
```

2. Register:
```python
# methods/__init__.py
def get_method(name, model, **kwargs):
    methods = {..., "my_new_mixup": MyNewMixup}
```

3. Run:
```bash
python scripts/main.py --method my_new_mixup
```

### New Dataset

```python
# data/loaders.py
def get_imagenet_dataloaders(...):
    # Return train_loader, val_loader, test_loader, num_classes
    pass

# scripts/main.py
if config.data.dataset == "imagenet":
    train_loader, ... = get_imagenet_dataloaders(...)
```

---

## Key Takeaways

| Aspect | Design Choice | Why |
|--------|---------------|-----|
| **Modularity** | Separate data/models/methods/training | Easy to swap components |
| **Configuration** | Dataclass + YAML | Type-safe + reproducible |
| **Metrics** | Full SC suite (AUROC, AURC, E-AURC) | Complete evaluation |
| **Logging** | W&B + TensorBoard | Experiment tracking + analysis |
| **Extensions** | Plugin-style methods | Add variants without modifying core |

---

## Next Steps

1. **Run baseline**: `python scripts/main.py --method standard`
2. **Run mixup**: `python scripts/main.py --method mixup`
3. **Compare**: Check W&B dashboard for AUROC/AURC/E-AURC
4. **Design variant**: Identify improvement opportunity
5. **Implement**: Add to `methods/mixup.py`
6. **Experiment**: Test on CIFAR-10/100

Good luck with your research! 🚀
