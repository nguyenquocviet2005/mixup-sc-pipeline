# Method 2 — Feature-kNN / Logit Probability Blending

## Overview

Feature-kNN (logit probability blending) uses a k-nearest-neighbors model in the model's feature space to produce a class-probability estimate and blends it with the model's softmax probabilities. This improves selective classification calibrated confidence by leveraging local neighborhood structure in representation space.

## Intuition

- The network's softmax probabilities can be overconfident or miscalibrated.
- A kNN classifier on penultimate-layer features gives an alternative, local-probability estimate that captures neighborhood agreement.
- Blending the two probabilities often yields better ranking for selective classification and can reduce AURC/E-AURC.

## Algorithm (high-level)

1. For each validation example compute the feature vector f(x) from the model's penultimate layer (before the final linear + softmax).
2. Fit a KNeighborsClassifier on validation features and validation labels. Use chosen `k` and distance weighting (e.g., distance-weighted probas or uniform).
3. For each test example compute its feature vector f(x_test).
4. Get kNN class-probabilities p_knn(y | f(x_test)).
5. Get the model softmax probabilities p_soft(y | x_test).
6. Blend probabilities:

$p_{blend}(y) = \alpha \cdot p_{soft}(y) + (1-\alpha) \cdot p_{knn}(y)$

where $\alpha \in [0,1]$ is chosen by tuning on the held-out calibration split.

7. Use the confidence score for selective classification as the max class probability from $p_{blend}$:

$s(x) = \max_y p_{blend}(y)$

8. Compute selection metrics (AURC, E-AURC, AUROC, ECE, etc.) using the resulting confidence scores and predictions.

## Implementation notes

- Feature extraction: use the model in evaluation mode and read the penultimate-layer activations (the layer directly before the final classifier logits). Ensure consistent preprocessing (normalization) between validation and test.
- kNN fitting: scikit-learn's `KNeighborsClassifier` is convenient. For distance-weighted probabilities, set `weights='distance'` and `metric='euclidean'` (L2) by default.
- Scaling / normalization: you may optionally L2-normalize feature vectors before kNN; treat this as a tuning choice.
- Blending weight `alpha` and `k` are tuned on the calibration/validation split via grid search; after selecting best hyperparameters you can refit kNN on the full validation set if desired.

## Hyperparameters to tune

- `k`: typical grid: [1, 5, 10, 20, 50, 100]
- `alpha`: typical grid: [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]
- `weights`: `uniform` vs `distance`
- `feature_norm`: `none` or `l2` normalization

Tune by optimizing the target metric on the held-out calibration split (e.g., minimize AURC or E-AURC, or maximize validation AUROC depending on your objective).

## Pseudocode

```
# extract features on val
F_val = model.extract_features(X_val)
# fit kNN
knn = KNeighborsClassifier(n_neighbors=k, weights=weights)
knn.fit(F_val, y_val)

# extract features on test
F_test = model.extract_features(X_test)

# get kNN probabilities
P_knn = knn.predict_proba(F_test)

# get softmax probabilities from logits
P_soft = softmax(model.forward_logits(X_test))

# blend
P_blend = alpha * P_soft + (1 - alpha) * P_knn

# confidence
conf = P_blend.max(axis=1)
pred = P_blend.argmax(axis=1)
```

## Integration with this repo

- The project's post-hoc evaluation pipeline expects post-hoc methods to return test probabilities (or a confidence override). Implementations can be found/extended in `scripts/evaluate_posthoc_methods.py`.
- Typical flow:
  - Compute and cache validation features & logits.
  - Grid-search `k` and `alpha` on the calibration split.
  - Apply chosen hyperparameters to test set features to produce `P_blend`.

## Example CLI (conceptual)

```bash
python scripts/evaluate_posthoc_methods.py \
  --dataset cifar10 \
  --checkpoint-dir ./checkpoints \
  --method feature_knn \
  --k 50 --alpha 0.5 \
  --output ./results/posthoc_feature_knn.json
```

Note: actual CLI flags may differ; the above is an illustrative example. See `scripts/evaluate_posthoc_methods.py` for the concrete entry points and parameter parsing.

## References

- K. He et al., "Deep Residual Learning for Image Recognition" — for ResNet feature backbones used in experiments.
- FixSelectiveClassification (external implementation) — provides baseline and related selective classification measures; inspires some neighborhood-based post-hoc approaches.

## Practical tips

- Precompute and cache features to speed repeated hyperparameter searches.
- When using large `k`, ensure you have enough validation examples (k << N_val).
- If the validation set is small, consider using distance-weighted probabilities rather than large `k`.

---

For follow-up, I can: add a short example script that extracts features and runs the grid search, or add a runnable notebook demonstrating `k`/`alpha` tuning on CIFAR-10.
