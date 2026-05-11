# Integration Summary: Calibration Metrics in Mixup SC Pipeline

## What Changed

The pipeline now tracks **4 calibration metrics** alongside the existing **3 selective classification metrics** to ensure improvements don't degrade model trustworthiness.

### New Metrics
- **ECE**: Expected Calibration Error (avg confidence-accuracy gap)
- **MCE**: Maximum Calibration Error (worst-case gap)
- **Brier Score**: Mean squared probability errors
- **NLL**: Negative Log-Likelihood (cross-entropy)

---

## Files Modified

### 1. `evaluation/metrics.py`
**Added**: `CalibrationMetrics` class with 5 functions:
- `compute_ece()` - Expected Calibration Error
- `compute_mce()` - Maximum Calibration Error
- `compute_brier_score()` - Brier Score
- `compute_nll()` - Negative Log-Likelihood
- `compute_all_calibration_metrics()` - All 4 at once

**Added**: `compute_probs_from_logits()` helper to get softmax probabilities

### 2. `evaluation/__init__.py`
**Updated**: Exports new `CalibrationMetrics` class and `compute_probs_from_logits`

### 3. `training/trainer.py`
**Updated imports**: Added `CalibrationMetrics, compute_probs_from_logits`

**Modified `validate_epoch()`**:
- Computes softmax probabilities
- Calls `CalibrationMetrics.compute_all_calibration_metrics()`
- Returns calibration metrics in results dict

**Modified `test_epoch()`**:
- Same as validate_epoch
- Full calibration evaluation on test set

**Updated console output**:
- Validation line now shows: ECE and Brier Score
- Test results now show: ECE, MCE, Brier, NLL

### 4. New documentation: `CALIBRATION.md`
Comprehensive guide explaining:
- What each metric means
- Why it matters for SC research
- Interpretation guidelines
- Common pitfalls and solutions

---

## How It Works in Practice

### During Training

```
Epoch 50/200:
Train Loss: 0.456 | Train Acc: 0.855 | Train AUROC: 0.820
  Val Loss: 0.523 | Val Acc: 0.823 | Val AUROC: 0.815 | Val AURC: 0.078 |
  Val ECE: 0.032 | Val Brier: 0.054  ← New!
```

### After Training (Final Test)

```
=== Final Test Evaluation ===
Test Loss: 0.523
Test Accuracy: 0.823
Test AUROC: 0.815           (SC ranking)
Test AURC: 0.078            (SC risk-coverage)
Test E-AURC: 0.015          (SC optimality gap)
Test ECE: 0.032             (Calibration gap)
Test MCE: 0.087             (Worst-case gap)
Test Brier: 0.054           (Probability error)
Test NLL: 0.185             (Surprise/uncertainty)
```

### W&B Dashboard

All 7 metrics logged per epoch:
- `val/auroc`, `val/aurc`, `val/eaurc` (SC metrics)
- `val/ece`, `val/mce`, `val/brier`, `val/nll` (Calibration metrics)
- Same for test results

---

## Usage Example: Run and Interpret

```bash
python scripts/main.py --dataset cifar10 --method mixup --exp-name test_run
```

**Interpretation Guide**:

```python
# Look at final test metrics
Test AUROC: 0.920  ← Good (> 0.90)
Test AURC:  0.045  ← Good (< 0.10)
Test ECE:   0.038  ← Good (< 0.05)  ✓
Test Brier: 0.062  ← Good (< 0.10)  ✓

# All metrics are good → Mixup improves SC without degrading calibration
```

**Problematic case**:
```python
Test AUROC: 0.935  ← Excellent
Test AURC:  0.035  ← Excellent
Test ECE:   0.22   ← Poor! ✗
Test Brier: 0.28   ← Poor! ✗

# High SC metrics but poor calibration
# → Model ranks well but confidence is untrustworthy
# Fix: Try temperature scaling or add calibration loss
```

---

## Comparison: Standard vs Mixup

**Expected Results** (CIFAR-10, ResNet-50):

| Metric | Standard | Mixup | Improvement |
|--------|----------|-------|-------------|
| AUROC | 0.900 | 0.920 | ↑2% |
| AURC | 0.055 | 0.042 | ↓24% |
| **ECE** | 0.035 | 0.025 | ↓29% ✓ |
| **MCE** | 0.095 | 0.072 | ↓24% ✓ |
| **Brier** | 0.075 | 0.050 | ↓33% ✓ |
| **NLL** | 0.220 | 0.155 | ↓30% ✓ |

**Key insight**: Mixup improves BOTH SC and calibration metrics!

---

## Variant Comparison with Calibration

What to expect with your 4 methods:

```
Method              AUROC  AURC   ECE    Comparison
─────────────────────────────────────────────────────
standard            0.910  0.048  0.032  Baseline
mixup               0.922  0.040  0.024  ✓ Best overall
mixup_variant1      0.916  0.044  0.018  ✓ Best calibration
mixup_variant2      0.925  0.038  0.028  ✓ Best SC metrics
```

**What to look for**:
- Variant1 (temp scaling): Trades some AUROC for better calibration
- Variant2 (adaptive alpha): Maintains both SC and calibration
- None should have ECE > 0.10 (that's poor)

---

## Research Directions

Now that you track calibration, you can ask:

1. **"How much calibration improvement does Mixup give for free?"**
   - Compare Standard vs Mixup ECE/Brier
   
2. **"Can we get better SC without hurting calibration?"**
   - Watch ECE trend while optimizing AUROC
   
3. **"What's the optimal temperature for this variant?"**
   - Post-hoc analysis: Try T values to minimize ECE

4. **"Which variant is most robust?"**
   - Look at MCE (worst-case) not just ECE (average)

---

## Code Examples

### Access Calibration Metrics

```python
from evaluation import CalibrationMetrics
import numpy as np

confidences = np.array([0.9, 0.8, 0.7, ...])  # Model confidence
correctness = np.array([1, 1, 0, ...])        # 1=correct, 0=incorrect
probs = np.array([[0.1, 0.9], [0.8, 0.2], ...])  # Full probability distribution
targets = np.array([1, 0, ...])               # True labels

# Compute all at once
metrics = CalibrationMetrics.compute_all_calibration_metrics(
    confidences, correctness, probs, targets, n_bins=10
)
# Returns: {ece: 0.032, mce: 0.087, brier: 0.054, nll: 0.185}

# Or individual metrics
ece = CalibrationMetrics.compute_ece(confidences, correctness, n_bins=10)
brier = CalibrationMetrics.compute_brier_score(probs, targets)
```

### Log to W&B

```python
logger.log_metrics({
    'val/ece': metrics['ece'],
    'val/brier': metrics['brier'],
    'val/auroc': metrics['auroc'],
    'val/aurc': metrics['aurc'],
}, step=epoch)
```

---

## Default Configuration

Calibration metrics compute with:
- **ECE/MCE**: 10 bins (confidence from 0.0 to 1.0)
- **Brier**: Per-sample average over all classes
- **NLL**: Mean cross-entropy loss

All use default parameters suitable for CIFAR-10/100.

---

## Next Steps

1. **Run your experiments**:
   ```bash
   python scripts/main.py --method standard --exp-name std
   python scripts/main.py --method mixup --exp-name mix
   ```

2. **Compare in W&B**:
   - View side-by-side: AUROC vs ECE
   - Check if AURC improvement comes free with ECE improvement

3. **Analyze failures**:
   - If ECE > 0.10, check calibration curve
   - If MCE >> ECE, you have a bad bin (analyze why)

4. **Design improvements**:
   - Use calibration metrics to guide variant design
   - Example: "Variant 3 uses token mixing with calibration loss"

---

## FAQ

**Q: Will calibration metrics slow down training?**  
A: No, computed only during validation/test (not in training loop)

**Q: Should I optimize ECE or AUROC?**  
A: Both! Track them separately. Ideally both improve with Mixup.

**Q: What if ECE gets worse with my variant?**  
A: Add L2 regularization, label smoothing, or post-hoc temperature scaling

**Q: How is ECE different from Brier?**  
A: ECE is binned (coarse), Brier is per-sample. Use both for complete picture.

---

**Happy researching!** Now you can develop Mixup improvements that are both more selective AND more trustworthy. 🎯
