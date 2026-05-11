"""Empirical verification of the 3-part theoretical analysis for Feature-kNN blending.

Experiments:
  H1 – Geometric Disentanglement (Neural Collapse Boundary Channels)
  H2 – Logit Shrinkage & Dynamic-Range Restoration
  H3 – Directional Derivative Anisotropy Correction

Each experiment produces quantitative results saved to JSON.
"""

import argparse
import json
import sys
import warnings
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent.parent))

from data import get_dataloaders
from models import get_model


# ---------- helpers (reused from evaluate_posthoc_methods) ----------

def extract_logits_and_features(model, inputs):
    """Run forward pass and expose penultimate features for ResNet and ViT."""
    if hasattr(model, 'forward_features') and hasattr(model, 'forward_head'):
        features_unpooled = model.forward_features(inputs)
        features = model.forward_head(features_unpooled, pre_logits=True)
        logits = model.head(features)
        return logits, features
    elif hasattr(model, 'features') and hasattr(model, 'classifier'):
        x = model.features(inputs)
        x = x.view(x.size(0), -1)
        features = model.classifier[:-1](x)
        logits = model.classifier[-1](features)
        return logits, features
    else:
        out = F.relu(model.bn1(model.conv1(inputs)))
        out = model.layer1(out)
        out = model.layer2(out)
        out = model.layer3(out)
        out = model.layer4(out)
        out = model.avgpool(out)
        features = out.view(out.size(0), -1)
        logits = model.fc(features)
        return logits, features


def collect_outputs(model, loader, device, max_batches=None):
    model.eval()
    L, F_list, T = [], [], []
    with torch.no_grad():
        for i, (x, t) in enumerate(loader):
            x = x.to(device)
            logits, feats = extract_logits_and_features(model, x)
            L.append(logits.cpu()); F_list.append(feats.cpu()); T.append(t.cpu())
            if max_batches and i + 1 >= max_batches:
                break
    return {
        "logits": torch.cat(L, 0),
        "features": torch.cat(F_list, 0).numpy(),
        "targets": torch.cat(T, 0).numpy().flatten(),
    }


def knn_blend(val_feats, val_targets, val_probs, test_feats, test_probs, nc, k=10, alpha=0.4):
    knn = KNeighborsClassifier(n_neighbors=k, weights="distance")
    knn.fit(val_feats, val_targets)
    partial = knn.predict_proba(test_feats)
    full = np.zeros((test_feats.shape[0], nc), dtype=np.float32)
    full[:, knn.classes_.astype(int)] = partial
    blended = (1.0 - alpha) * test_probs + alpha * full
    return blended, full


# ==================================================================
# H1: Geometric Disentanglement – Neural Collapse Boundary Channels
# ==================================================================

def experiment_h1(model, val_data, test_data, num_classes, device):
    """Verify that kNN rescues samples stuck in Mixup boundary channels.

    Key metric:  For samples whose P_base confidence is LOW, measure
      (a) their distance to the nearest class centroid (ETF vertex),
      (b) kNN probability, and
      (c) the correction magnitude (P_blend - P_base).

    If H1 holds:
      • Low-confidence test points that are geometrically CLOSE to a
        centroid should get a LARGE positive correction from kNN.
      • Low-confidence test points that are geometrically FAR should
        NOT get a spurious boost.
    """
    print("\n" + "=" * 70)
    print("H1: Geometric Disentanglement of Neural Collapse Boundary Channels")
    print("=" * 70)

    feats_val = val_data["features"]
    targets_val = val_data["targets"]
    feats_test = test_data["features"]
    targets_test = test_data["targets"]

    logits_test = test_data["logits"]
    probs_test = F.softmax(logits_test, dim=1).numpy()
    confs_base = np.max(probs_test, axis=1)
    preds_test = np.argmax(probs_test, axis=1)

    probs_val = F.softmax(val_data["logits"], dim=1).numpy()

    # 1. Compute class centroids (ETF vertex proxies) from validation set
    dim = feats_val.shape[1]
    centroids = np.zeros((num_classes, dim), dtype=np.float64)
    for c in range(num_classes):
        idx = np.where(targets_val == c)[0]
        if len(idx) > 0:
            centroids[c] = np.mean(feats_val[idx], axis=0)

    # 2. Distance of test features to nearest centroid
    dists_to_nearest = np.zeros(feats_test.shape[0])
    nearest_class = np.zeros(feats_test.shape[0], dtype=int)
    for i in range(feats_test.shape[0]):
        d = np.linalg.norm(feats_test[i] - centroids, axis=1)
        nearest_class[i] = np.argmin(d)
        dists_to_nearest[i] = d[nearest_class[i]]

    # 3. kNN blend
    blended, knn_probs = knn_blend(
        feats_val, targets_val, probs_val,
        feats_test, probs_test, num_classes,
    )
    confs_blend = np.max(blended, axis=1)
    confs_knn = np.max(knn_probs, axis=1)
    correction = confs_blend - confs_base

    # 4. Stratify by confidence quartile
    q25, q50, q75 = np.percentile(confs_base, [25, 50, 75])
    bins = [
        ("Q1 (lowest 25%)", confs_base <= q25),
        ("Q2 (25-50%)", (confs_base > q25) & (confs_base <= q50)),
        ("Q3 (50-75%)", (confs_base > q50) & (confs_base <= q75)),
        ("Q4 (highest 25%)", confs_base > q75),
    ]

    results = {}
    for label, mask in bins:
        if mask.sum() == 0:
            continue
        d_near = dists_to_nearest[mask]
        corr = correction[mask]
        # Split by centroid proximity (median split within quartile)
        d_med = np.median(d_near)
        close_mask = d_near <= d_med
        far_mask = d_near > d_med

        results[label] = {
            "n_samples": int(mask.sum()),
            "mean_dist_to_centroid": float(np.mean(d_near)),
            "mean_base_conf": float(np.mean(confs_base[mask])),
            "mean_knn_conf": float(np.mean(confs_knn[mask])),
            "mean_blend_conf": float(np.mean(confs_blend[mask])),
            "mean_correction": float(np.mean(corr)),
            "close_to_centroid": {
                "n": int(close_mask.sum()),
                "mean_correction": float(np.mean(corr[close_mask])) if close_mask.any() else 0,
                "mean_dist": float(np.mean(d_near[close_mask])) if close_mask.any() else 0,
            },
            "far_from_centroid": {
                "n": int(far_mask.sum()),
                "mean_correction": float(np.mean(corr[far_mask])) if far_mask.any() else 0,
                "mean_dist": float(np.mean(d_near[far_mask])) if far_mask.any() else 0,
            },
        }
        print(f"  {label}: n={mask.sum()}, mean_corr={np.mean(corr):.4f}, "
              f"close_corr={results[label]['close_to_centroid']['mean_correction']:.4f}, "
              f"far_corr={results[label]['far_from_centroid']['mean_correction']:.4f}")

    # 5. Correlation: centroid distance vs correction magnitude (for low-conf samples)
    low_conf_mask = confs_base <= q50
    if low_conf_mask.sum() > 10:
        from scipy.stats import pearsonr, spearmanr
        rp, pp = pearsonr(dists_to_nearest[low_conf_mask], correction[low_conf_mask])
        rs, ps = spearmanr(dists_to_nearest[low_conf_mask], correction[low_conf_mask])
        results["correlation_low_conf"] = {
            "pearson_r": float(rp), "pearson_p": float(pp),
            "spearman_r": float(rs), "spearman_p": float(ps),
        }
        print(f"  Low-conf correlation (dist vs correction): pearson r={rp:.4f} p={pp:.4e}, "
              f"spearman r={rs:.4f} p={ps:.4e}")

    # 6. Accuracy improvement stratified
    correct_base = (preds_test == targets_test).astype(int)
    preds_blend = np.argmax(blended, axis=1)
    correct_blend = (preds_blend == targets_test).astype(int)
    results["accuracy"] = {
        "baseline": float(np.mean(correct_base)),
        "blended": float(np.mean(correct_blend)),
    }
    print(f"  Accuracy: base={np.mean(correct_base):.4f}, blend={np.mean(correct_blend):.4f}")

    return results


# ==================================================================
# H2: Overcoming Implicit Label Smoothing and Logit Shrinkage
# ==================================================================

def experiment_h2(model, val_data, test_data, num_classes, device):
    """Verify that kNN restores dynamic range crushed by Mixup label smoothing.

    Key metrics:
      • Logit dynamic range (max - min per sample)
      • Entropy of P_base vs P_kNN vs P_blend
      • Confidence spread (std of max-probs)
      • Ordinal ranking: does blending improve rank correlation with correctness?
    """
    print("\n" + "=" * 70)
    print("H2: Overcoming Implicit Label Smoothing and Logit Shrinkage")
    print("=" * 70)

    feats_val = val_data["features"]
    targets_val = val_data["targets"]
    feats_test = test_data["features"]
    targets_test = test_data["targets"]

    logits_test = test_data["logits"].numpy()
    probs_test = F.softmax(test_data["logits"], dim=1).numpy()
    probs_val = F.softmax(val_data["logits"], dim=1).numpy()

    confs_base = np.max(probs_test, axis=1)
    preds_base = np.argmax(probs_test, axis=1)
    correctness = (preds_base == targets_test).astype(int)

    blended, knn_probs = knn_blend(
        feats_val, targets_val, probs_val,
        feats_test, probs_test, num_classes,
    )
    confs_blend = np.max(blended, axis=1)
    confs_knn = np.max(knn_probs, axis=1)

    # 1. Logit dynamic range
    logit_range = np.max(logits_test, axis=1) - np.min(logits_test, axis=1)
    logit_margin = np.sort(logits_test, axis=1)[:, -1] - np.sort(logits_test, axis=1)[:, -2]

    # 2. Entropy
    def entropy(p):
        p = np.clip(p, 1e-12, 1.0)
        return -np.sum(p * np.log(p), axis=1)

    ent_base = entropy(probs_test)
    ent_knn = entropy(knn_probs)
    ent_blend = entropy(blended)

    # 3. Confidence spread
    # 4. AUROC of confidence as a selector (higher = better ordinality)
    from sklearn.metrics import roc_auc_score
    auroc_base = roc_auc_score(correctness, confs_base) if len(np.unique(correctness)) > 1 else 0.5
    auroc_knn = roc_auc_score(correctness, confs_knn) if len(np.unique(correctness)) > 1 else 0.5
    auroc_blend = roc_auc_score(correctness, confs_blend) if len(np.unique(correctness)) > 1 else 0.5

    results = {
        "logit_dynamic_range": {
            "mean": float(np.mean(logit_range)),
            "std": float(np.std(logit_range)),
            "median": float(np.median(logit_range)),
        },
        "logit_margin_top2": {
            "mean": float(np.mean(logit_margin)),
            "std": float(np.std(logit_margin)),
        },
        "confidence_stats": {
            "P_base": {"mean": float(np.mean(confs_base)), "std": float(np.std(confs_base))},
            "P_kNN": {"mean": float(np.mean(confs_knn)), "std": float(np.std(confs_knn))},
            "P_blend": {"mean": float(np.mean(confs_blend)), "std": float(np.std(confs_blend))},
        },
        "entropy": {
            "P_base": {"mean": float(np.mean(ent_base)), "std": float(np.std(ent_base))},
            "P_kNN": {"mean": float(np.mean(ent_knn)), "std": float(np.std(ent_knn))},
            "P_blend": {"mean": float(np.mean(ent_blend)), "std": float(np.std(ent_blend))},
        },
        "ordinal_quality_auroc": {
            "P_base": float(auroc_base),
            "P_kNN": float(auroc_knn),
            "P_blend": float(auroc_blend),
        },
    }

    # 5. Per-quartile analysis: does kNN preferentially boost squashed-but-correct samples?
    q25, q50, q75 = np.percentile(confs_base, [25, 50, 75])
    for label, mask in [("Q1_lowest", confs_base <= q25), ("Q4_highest", confs_base > q75)]:
        if mask.sum() == 0:
            continue
        acc_in_bin = float(np.mean(correctness[mask]))
        mean_ent_base = float(np.mean(ent_base[mask]))
        mean_ent_blend = float(np.mean(ent_blend[mask]))
        mean_conf_base = float(np.mean(confs_base[mask]))
        mean_conf_blend = float(np.mean(confs_blend[mask]))
        results[f"quartile_{label}"] = {
            "n": int(mask.sum()),
            "accuracy": acc_in_bin,
            "entropy_base": mean_ent_base,
            "entropy_blend": mean_ent_blend,
            "entropy_reduction": mean_ent_base - mean_ent_blend,
            "conf_base": mean_conf_base,
            "conf_blend": mean_conf_blend,
            "conf_boost": mean_conf_blend - mean_conf_base,
        }

    print(f"  Logit dynamic range: mean={np.mean(logit_range):.2f}±{np.std(logit_range):.2f}")
    print(f"  Conf spread (std):   P_base={np.std(confs_base):.4f}, "
          f"P_kNN={np.std(confs_knn):.4f}, P_blend={np.std(confs_blend):.4f}")
    print(f"  Entropy:  P_base={np.mean(ent_base):.4f}, "
          f"P_kNN={np.mean(ent_knn):.4f}, P_blend={np.mean(ent_blend):.4f}")
    print(f"  Selection AUROC:  P_base={auroc_base:.4f}, "
          f"P_kNN={auroc_knn:.4f}, P_blend={auroc_blend:.4f}")

    return results


# ==================================================================
# H3: Mitigating Directional Derivative Anisotropy
# ==================================================================

def experiment_h3(model, val_data, test_data, num_classes, device):
    """Verify that Mixup creates anisotropic confidence and kNN corrects it.

    Approach: For each test sample, perturb in two directions:
      (a) Interpolation direction (toward random same-batch point) – "Mixup axis"
      (b) Random orthogonal direction – "orthogonal axis"

    Measure: Confidence sensitivity (|dconf/dε|) in each direction for P_base and P_kNN.

    If H3 holds:
      • P_base is smoother (lower sensitivity) along Mixup axis
      • P_kNN sensitivity is similar in both directions (isotropic)
    """
    print("\n" + "=" * 70)
    print("H3: Mitigating Directional Derivative Anisotropy")
    print("=" * 70)

    feats_val = val_data["features"]
    targets_val = val_data["targets"]
    feats_test = test_data["features"]
    targets_test = test_data["targets"]
    probs_val = F.softmax(val_data["logits"], dim=1).numpy()

    model.eval()
    n_test = min(feats_test.shape[0], 2000)  # subsample for speed
    np.random.seed(42)
    sub_idx = np.random.choice(feats_test.shape[0], n_test, replace=False)
    feats_sub = feats_test[sub_idx]
    targets_sub = targets_test[sub_idx]

    # Fit kNN once
    knn = KNeighborsClassifier(n_neighbors=10, weights="distance")
    knn.fit(feats_val, targets_val)

    eps = 0.01  # perturbation magnitude (relative to feature norm)

    sens_base_interp = []
    sens_base_ortho = []
    sens_knn_interp = []
    sens_knn_ortho = []

    def get_base_conf(feats_batch):
        """Get P_base confidence from features via the linear classifier."""
        with torch.no_grad():
            logits = model.fc(torch.from_numpy(feats_batch).float().to(device))
            probs = F.softmax(logits, dim=1).cpu().numpy()
        return np.max(probs, axis=1), probs

    def get_knn_conf(feats_batch):
        """Get P_kNN confidence from features."""
        partial = knn.predict_proba(feats_batch)
        full = np.zeros((feats_batch.shape[0], num_classes), dtype=np.float32)
        full[:, knn.classes_.astype(int)] = partial
        return np.max(full, axis=1)

    dim = feats_sub.shape[1]

    for i in range(n_test):
        feat_i = feats_sub[i]
        feat_norm = np.linalg.norm(feat_i) + 1e-8

        # Interpolation direction: toward a random other point
        j = np.random.randint(n_test)
        while j == i:
            j = np.random.randint(n_test)
        interp_dir = feats_sub[j] - feat_i
        interp_dir = interp_dir / (np.linalg.norm(interp_dir) + 1e-8)

        # Random orthogonal direction via Gram-Schmidt
        rand_dir = np.random.randn(dim).astype(np.float32)
        rand_dir -= np.dot(rand_dir, interp_dir) * interp_dir
        rand_dir = rand_dir / (np.linalg.norm(rand_dir) + 1e-8)

        step = eps * feat_norm

        feat_plus_interp = (feat_i + step * interp_dir).reshape(1, -1)
        feat_minus_interp = (feat_i - step * interp_dir).reshape(1, -1)
        feat_plus_ortho = (feat_i + step * rand_dir).reshape(1, -1)
        feat_minus_ortho = (feat_i - step * rand_dir).reshape(1, -1)

        # P_base sensitivity
        c_pi, _ = get_base_conf(feat_plus_interp.astype(np.float32))
        c_mi, _ = get_base_conf(feat_minus_interp.astype(np.float32))
        c_po, _ = get_base_conf(feat_plus_ortho.astype(np.float32))
        c_mo, _ = get_base_conf(feat_minus_ortho.astype(np.float32))

        sens_base_interp.append(abs(c_pi[0] - c_mi[0]) / (2 * step))
        sens_base_ortho.append(abs(c_po[0] - c_mo[0]) / (2 * step))

        # P_kNN sensitivity
        ck_pi = get_knn_conf(feat_plus_interp.astype(np.float32))
        ck_mi = get_knn_conf(feat_minus_interp.astype(np.float32))
        ck_po = get_knn_conf(feat_plus_ortho.astype(np.float32))
        ck_mo = get_knn_conf(feat_minus_ortho.astype(np.float32))

        sens_knn_interp.append(abs(ck_pi[0] - ck_mi[0]) / (2 * step))
        sens_knn_ortho.append(abs(ck_po[0] - ck_mo[0]) / (2 * step))

    sens_base_interp = np.array(sens_base_interp)
    sens_base_ortho = np.array(sens_base_ortho)
    sens_knn_interp = np.array(sens_knn_interp)
    sens_knn_ortho = np.array(sens_knn_ortho)

    # Anisotropy ratio: ortho / interp sensitivity (>1 means more sensitive orthogonally)
    base_ratio = np.mean(sens_base_ortho) / (np.mean(sens_base_interp) + 1e-12)
    knn_ratio = np.mean(sens_knn_ortho) / (np.mean(sens_knn_interp) + 1e-12)

    from scipy.stats import wilcoxon
    # Test: base_ortho > base_interp ?
    try:
        stat_base, p_base = wilcoxon(sens_base_ortho, sens_base_interp, alternative="greater")
    except Exception:
        stat_base, p_base = 0.0, 1.0
    # Test: knn_ortho ≈ knn_interp ?  (two-sided)
    try:
        stat_knn, p_knn = wilcoxon(sens_knn_ortho, sens_knn_interp, alternative="two-sided")
    except Exception:
        stat_knn, p_knn = 0.0, 1.0

    results = {
        "n_samples_tested": n_test,
        "epsilon_relative": eps,
        "P_base_sensitivity": {
            "interp_mean": float(np.mean(sens_base_interp)),
            "interp_std": float(np.std(sens_base_interp)),
            "ortho_mean": float(np.mean(sens_base_ortho)),
            "ortho_std": float(np.std(sens_base_ortho)),
            "anisotropy_ratio_ortho_over_interp": float(base_ratio),
            "wilcoxon_ortho_gt_interp_p": float(p_base),
        },
        "P_kNN_sensitivity": {
            "interp_mean": float(np.mean(sens_knn_interp)),
            "interp_std": float(np.std(sens_knn_interp)),
            "ortho_mean": float(np.mean(sens_knn_ortho)),
            "ortho_std": float(np.std(sens_knn_ortho)),
            "anisotropy_ratio_ortho_over_interp": float(knn_ratio),
            "wilcoxon_two_sided_p": float(p_knn),
        },
    }

    print(f"  P_base sensitivity – interp: {np.mean(sens_base_interp):.6f}, ortho: {np.mean(sens_base_ortho):.6f}")
    print(f"  P_base anisotropy ratio (ortho/interp): {base_ratio:.4f}   (Wilcoxon p={p_base:.4e})")
    print(f"  P_kNN  sensitivity – interp: {np.mean(sens_knn_interp):.6f}, ortho: {np.mean(sens_knn_ortho):.6f}")
    print(f"  P_kNN  anisotropy ratio (ortho/interp): {knn_ratio:.4f}   (Wilcoxon p={p_knn:.4e})")

    return results


# ==================================================================
# main
# ==================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="cifar10", choices=["cifar10", "cifar100", "tinyimagenet"])
    parser.add_argument("--checkpoint-dir", type=str, default="./checkpoints")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load checkpoint
    import re
    ckpt_dir = Path(args.checkpoint_dir)
    if args.dataset == "cifar10":
        glob_pat = "exp2_mixup_best_auroc_epoch_*.pt"
    elif args.dataset == "cifar100":
        glob_pat = "cifar100_mixup_variant2_best_auroc_epoch_*.pt"
    elif args.dataset == "tinyimagenet":
        glob_pat = "mixup_tinyimagenet_resnet18_best_auroc_epoch_*.pt"
    pat = re.compile(r".*_epoch_(\d+)\.pt$")
    cands = [(int(pat.search(f.name).group(1)), f) for f in ckpt_dir.glob(glob_pat) if pat.search(f.name)]
    cands.sort(key=lambda x: x[0])
    ckpt_path = cands[-1][1]

    nc = 10 if args.dataset == "cifar10" else (100 if args.dataset == "cifar100" else 200)
    model = get_model("resnet18", num_classes=nc).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"Loaded: {ckpt_path}  ({args.dataset}, {nc} classes)")

    train_loader, val_loader, test_loader, _ = get_dataloaders(
        dataset=args.dataset, data_dir="./data", batch_size=256, val_batch_size=256,
        num_workers=4, augmentation=False, seed=args.seed,
    )

    print("Collecting outputs...")
    val_data = collect_outputs(model, val_loader, device)
    test_data = collect_outputs(model, test_loader, device)

    acc = float(np.mean(np.argmax(F.softmax(test_data["logits"], dim=1).numpy(), axis=1) == test_data["targets"]))
    print(f"Test accuracy: {acc:.4f}")

    results = {
        "dataset": args.dataset,
        "checkpoint": str(ckpt_path),
        "test_accuracy": acc,
    }

    results["H1_geometric_disentanglement"] = experiment_h1(model, val_data, test_data, nc, device)
    results["H2_logit_shrinkage"] = experiment_h2(model, val_data, test_data, nc, device)
    results["H3_anisotropy"] = experiment_h3(model, val_data, test_data, nc, device)

    out = Path(args.output or f"./results/theory_verification_{args.dataset}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"\nSaved to: {out}")


if __name__ == "__main__":
    main()
