# Calibration Metrics for Selective Classification

## Overview

Calibration metrics measure how well the model's **confidence aligns with its accuracy**. In the context of Selective Classification, we care about **both** improving SC metrics (AUROC, AURC) **and** maintaining good calibration—otherwise the confidence scores become unreliable.

This guide explains the four calibration metrics tracked in the pipeline.

---

## Calibration Metrics Explained

### **1. Expected Calibration Error (ECE)** ✅ **Most Important**

**Definition**: Average difference between confidence and accuracy across confidence bins.

$$\text{ECE} = \sum_{i=1}^{M} \frac{|B_i|}{N} \left| \text{acc}(B_i) - \text{conf}(B_i) \right|$$

Where:
- $B_i$ = set of predictions in bin $i$ (e.g., confidence 0.0-0.1)
- $\text{acc}(B_i)$ = accuracy of samples in bin $i$
- $\text{conf}(B_i)$ = average confidence of samples in bin $i$
- $M$ = number of bins (default: 10)

**Range**: [0, 1]

**Better**: Lower is better (0 = perfectly calibrated)

**Interpretation**:
- **ECE = 0.05**: When model says 80% confidence, accuracy is ~75% on average across bins
- **ECE = 0.20**: Large gaps between confidence and accuracy (poorly calibrated)

**Example**:
```
Bin 1 (Conf 0.0-0.1): Accuracy=5%, Avg Conf=7%  → Gap = 2%
Bin 2 (Conf 0.1-0.2): Accuracy=15%, Avg Conf=12% → Gap = 3%
Bin 3 (Conf 0.9-1.0): Accuracy=92%, Avg Conf=95% → Gap = 3%
...
ECE = weighted average of gaps = 3.5%
```

**Why it matters for SC**:
- High ECE means confidence isn't trustworthy for selective classification
- You can't set reliable rejection thresholds
- A model might claim 95% confidence but only be 70% correct

---

### **2. Maximum Calibration Error (MCE)**

**Definition**: Worst-case calibration error (maximum gap in any bin).

$$\text{MCE} = \max_i \left| \text{acc}(B_i) - \text{conf}(B_i) \right|$$

**Range**: [0, 1]

**Better**: Lower is better

**Interpretation**:
- **MCE = 0.10**: No bin has a confidence-accuracy gap > 10%
- **MCE = 0.40**: Some bin is wildly miscalibrated (e.g., 40% confident but only 10% accurate)

**Why it matters**:
- ECE gives you the average, MCE catches worst-case failures
- A model with ECE=0.05 but MCE=0.30 has a hidden problem zone
- For safety-critical SC, MCE should be low

**Example**:
```
Bin 1: Gap = 2%
Bin 2: Gap = 3%
Bin 3: Gap = 15%  ← Highest
Bin 4: Gap = 1%
MCE = 15%  (max gap)
ECE = 5%   (average gap)
```

---

### **3. Brier Score**

**Definition**: Mean squared difference between predicted probabilities and true labels.

$$\text{Brier} = \frac{1}{N} \sum_{i=1}^{N} \sum_{j=1}^{C} (p_{ij} - y_{ij})^2$$

Where:
- $p_{ij}$ = predicted probability for sample $i$, class $j$
- $y_{ij}$ = ground truth (1-hot encoded) for sample $i$, class $j$

**Range**: [0, 1]

**Better**: Lower is better (0 = perfect predictions)

**Interpretation**:
- **Brier = 0.05**: Squared differences from true labels average 0.05 (good)
- **Brier = 0.30**: Large probability errors (poor)

**Typical values**:
- Random classifier (all classes equal): 0.5
- Well-trained model: 0.05-0.20
- Overfitting (100% train acc): ~0.0 (train), > 0.2 (test)

**Why it matters**:
- Directly measures probability prediction quality
- Penalizes overconfidence (saying 99% when actually wrong)
- Complements ECE (ECE is binned, Brier is per-sample)

**Example**:
```
Sample 1: [0.1, 0.8, 0.1] predicted, true class = 2 (label [0,0,1])
Error: (0.1-0)² + (0.8-0)² + (0.1-1)² = 0.01 + 0.64 + 0.81 = 1.46
Average over all samples = Brier score
```

---

### **4. Negative Log-Likelihood (NLL)** 

**Definition**: Cross-entropy loss. Punishes assigning low probability to correct class.

$$\text{NLL} = -\frac{1}{N} \sum_{i=1}^{N} \log(p_{i,y_i})$$

Where:
- $p_{i,y_i}$ = predicted probability of true class for sample $i$

**Range**: [0, ∞)

**Better**: Lower is better

**Typical values**:
- Random guess (1/C probability): $\log(C)$ = 2.3 for CIFAR-10
- Well-trained: 0.1-0.3
- Overconfident wrong predictions: > 5

**Interpretation**:
- **NLL = 0.15**: Model usually assigns ~85% probability to correct class
- **NLL = 2.0**: Model often unsure (like random guessing)
- **NLL = 10**: Model likely making overconfident wrong predictions

**Why it matters**:
- Direct training objective (cross-entropy)
- Penalizes confidence on wrong classes
- A model with NLL=0.5 and accuracy=99% is overconfident
- A model with NLL=0.1 and accuracy=99% is well-calibrated

---

## Interpreting Results: SC + Calibration Tradeoff

### Ideal Scenario
```
Test Accuracy:   96.5%
Test AUROC:      0.93   ✓ (high confidence-correctness ranking)
Test AURC:       0.05   ✓ (low risk at high coverage)
Test ECE:        0.04   ✓ (good calibration)
Test Brier:      0.06   ✓ (good probability predictions)
Test NLL:        0.15   ✓ (well-calibrated on correct class)
```
**Interpretation**: Model is accurate, can rank by confidence, and confidence is trustworthy.

---

### Warning: High AUROC but Bad Calibration
```
Test Accuracy:   96.5%
Test AUROC:      0.95   ✓ (separates correct/incorrect well)
Test AURC:       0.04   ✓ (low risk)
Test ECE:        0.25   ✗ (poor calibration!)
Test Brier:      0.30   ✗ (overconfident)
Test NLL:        1.20   ✗ (very confident wrong answers)
```
**Problem**: Model ranks predictions well but confidence is unreliable.
- When it says 90%, actual accuracy might be 65%
- **Why this happens**: Mixup or temperature scaling optimized for ranking but not calibration
- **Fix**: Add calibration loss (focal loss, label smoothing) or post-hoc temperature scaling

---

### Common Patterns with Mixup Variants

#### Standard Training
```
Typically Overconfident (high confidence, lower accuracy than claimed):
- High AUROC (confidence separates well)
- Higher ECE (gaps between confidence and accuracy)
- Higher Brier/NLL
```

#### Mixup
```
Often better calibrated (soft labels help):
- High AUROC (still good ranking)
- Lower ECE (Mixup acts as regularizer)
- Lower Brier/NLL
```

#### Temperature Scaling Post-Hoc
```
Can improve calibration without retraining:
- T > 1.0: Softer probabilities → lower confidence, better ECE
- T < 1.0: Sharper probabilities → higher confidence, worse ECE
```

---

## Using Calibration Metrics in Your Research

### 1. Check After Training
```python
metrics = trainer.test_epoch()

print(f"AUROC: {metrics['auroc']:.4f}")     # SC ranking metric
print(f"ECE: {metrics['ece']:.4f}")         # Calibration metric
print(f"Brier: {metrics['brier']:.4f}")     # Probability quality

# Good if both are low:
if metrics['auroc'] > 0.90 and metrics['ece'] < 0.05:
    print("✓ Good SC + good calibration")
elif metrics['auroc'] > 0.90 and metrics['ece'] > 0.15:
    print("⚠ Good SC but poor calibration - needs fixing")
```

### 2. Compare Methods
```
Method        AUROC  AURC   ECE    Brier
Standard      0.910  0.045  0.035  0.050
Mixup         0.920  0.040  0.025  0.035  ← Better all around
Variant1      0.915  0.042  0.020  0.030  ← Best calibration
Variant2      0.925  0.038  0.028  0.040  ← Best SC, ok calibration
```

### 3. Post-hoc Calibration
If a variant has great AUROC but poor ECE, apply temperature scaling:

```python
# Model trained to 95.2% accuracy
# Test ECE = 0.22 (poor)
# Solution: Temperature scaling

T_values = [0.5, 1.0, 1.5, 2.0, 2.5]
best_T = 1.5  # Minimizes ECE

# Apply at inference:
logits_scaled = logits / 1.5
probs_calibrated = softmax(logits_scaled)  # Better ECE, same AUROC
```

---

## Quick Reference Table

| Metric | Range | Better | What It Measures |
|--------|-------|--------|------------------|
| **ECE** | [0,1] | Lower | Avg gap between confidence & accuracy |
| **MCE** | [0,1] | Lower | Worst-case gap |
| **Brier** | [0,1] | Lower | Squared probability errors |
| **NLL** | [0,∞) | Lower | -log(prob of correct class) |
| **AUROC** | [0,1] | Higher | Confidence ranks correct/incorrect |
| **AURC** | [0,1] | Lower | Average risk across coverage |

---

## Common Issues & Solutions

### Issue: High AUROC, High ECE
**Cause**: Model ranks well but confidence is overconfident
**Solutions**:
1. Add mixup (soft labels help)
2. Label smoothing
3. Post-hoc temperature scaling (T > 1)
4. Calibration loss (focal loss)

### Issue: Low AUROC, Low ECE
**Cause**: Model is uncertain but calibrated
**Solutions**:
1. Improve accuracy first (AUROC undefined without separation)
2. Better regularization balance
3. Longer training

### Issue: ECE Good, MCE Bad
**Cause**: One or two confidence bins are completely wrong
**Solutions**:
1. More training data for underrepresented classes
2. Class-balanced sampling
3. Check for data quality issues

---

## References

- **Guo et al.** "On Calibration of Modern Neural Networks" (ICML 2017)
  - Defines ECE, MCE, temperature scaling
- **DeGroot & Fienberg** "The Comparison and Evaluation of Forecasters" (1983)
  - Original calibration concepts
- **Gupta et al.** "Calibration and Uncertainty Metrics" (2021)
  - Modern calibration for deep learning

---

## Summary

**For Selective Classification research focused on Mixup improvements:**

1. **Track AUROC/AURC**: Are you improving SC?
2. **Track ECE/Brier**: Are you degrading calibration?
3. **Goal**: Maximize AUROC, minimize AURC, keep ECE/Brier low
4. **If AUROC ↑ but ECE ↑**: Use temperature scaling post-hoc or add calibration loss
5. **Mixup defaults**: Usually improves both SC and calibration

Good luck! 📊
