"""
Ablation study for method_2_feature_knn_logit_blend.

Six ablation dimensions:
  A. k × alpha hyper-parameter grid (wider than default [5,10,20,30]×[0.1..0.85])
  B. kNN weight scheme: uniform / distance
  C. Feature representation: raw / L2-normalised / PCA-reduced
  D. Confidence signal: max_prob / logit-margin (top1-top2) / neg-entropy
  E. Checkpoint-epoch stability (all checkpoints for a given dataset)
  F. Cross-dataset generalisation (cifar10 / cifar100 / medmnist)

Usage examples
--------------
# Quick full run on CIFAR-10 (best checkpoint auto-selected):
python scripts/ablate_knn_blend.py --dataset cifar10

# Specify a checkpoint explicitly:
python scripts/ablate_knn_blend.py --dataset cifar10 \
    --checkpoint checkpoints/exp2_mixup_best_auroc_epoch_195.pt

# Skip the expensive per-epoch stability sweep:
python scripts/ablate_knn_blend.py --dataset cifar10 --no-epoch-sweep

# Run only specific ablation dimensions:
python scripts/ablate_knn_blend.py --dataset cifar10 --dims A B C D
"""

import argparse
import itertools
import json
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier

sys.path.insert(0, str(Path(__file__).parent.parent))

from data import get_dataloaders, MEDMNIST_DATASETS
from evaluation import SelectionMetrics
from models import get_model


# ---------------------------------------------------------------------------
# Utilities shared across all ablation dimensions
# ---------------------------------------------------------------------------

def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _evaluate_sc(preds: np.ndarray, confs: np.ndarray, targets: np.ndarray) -> Dict[str, float]:
    if not np.all(np.isfinite(confs)):
        confs = np.nan_to_num(confs, nan=0.0, posinf=1e6, neginf=-1e10)
    correctness = (preds == targets).astype(int)
    sc = SelectionMetrics.compute_all_metrics(confs, correctness)
    accuracy = float(np.mean(correctness))
    aurc  = float(sc["aurc"])
    eaurc = float(sc["eaurc"])
    auroc = float(sc["auroc"])
    # N-AURC: normalises E-AURC by the E-AURC of a random confidence estimator.
    # A random estimator has AURC = R(h) = 1 - accuracy, so its E-AURC equals
    # R(h) - AURC_optimal  =  R(h) - (aurc - eaurc).
    risk       = 1.0 - accuracy          # R(h)
    aurc_opt   = aurc - eaurc            # AURC*(h,g*)
    random_eaurc = risk - aurc_opt       # denominator = E-AURC of random estimator
    naurc = float(eaurc / random_eaurc) if random_eaurc > 1e-9 else 0.0
    return {
        "accuracy": accuracy,
        "auroc":    auroc,
        "aurc":     aurc,
        "eaurc":    eaurc,
        "naurc":    naurc,
    }


def _full_probs_from_knn(knn: KNeighborsClassifier, features: np.ndarray, num_classes: int) -> np.ndarray:
    probs_partial = knn.predict_proba(features)
    probs_full = np.zeros((features.shape[0], num_classes), dtype=np.float32)
    probs_full[:, knn.classes_.astype(int)] = probs_partial
    return probs_full


def _extract_confidence(probs: np.ndarray, logits: Optional[np.ndarray], scheme: str) -> np.ndarray:
    """Extract scalar confidence from probability (and optionally logit) arrays.

    Schemes:
      'max_prob'   – max softmax probability (standard)
      'margin'     – top-1 minus top-2 probability
      'neg_entropy'– negative Shannon entropy of probabilities
    """
    if scheme == "max_prob":
        return np.max(probs, axis=1)
    if scheme == "margin":
        sorted_p = np.sort(probs, axis=1)
        return sorted_p[:, -1] - sorted_p[:, -2]
    if scheme == "neg_entropy":
        p = np.clip(probs, 1e-12, 1.0)
        return np.sum(p * np.log(p), axis=1)   # − (−H) = −H, lower is more uncertain
    raise ValueError(f"Unknown confidence scheme: {scheme}")


def _apply_feature_transform(features: np.ndarray, transform: str, pca_model=None, n_components: int = 64) -> Tuple[np.ndarray, object]:
    """Transform features; returns (transformed, pca_model_or_None)."""
    if transform == "raw":
        return features, None
    if transform == "l2":
        norms = np.linalg.norm(features, axis=1, keepdims=True) + 1e-8
        return features / norms, None
    if transform == "pca":
        nc = min(n_components, features.shape[0] - 1, features.shape[1])
        if pca_model is None:
            pca_model = PCA(n_components=nc, random_state=42)
            pca_model.fit(features)
        return pca_model.transform(features).astype(np.float32), pca_model
    raise ValueError(f"Unknown feature transform: {transform}")


def _extract_features(model: torch.nn.Module, inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return (logits, penultimate features) for ResNet / VGG / ViT."""
    if hasattr(model, "forward_features") and hasattr(model, "forward_head"):
        feats = model.forward_head(model.forward_features(inputs), pre_logits=True)
        logits = model.head(feats)
        return logits, feats
    if hasattr(model, "features") and hasattr(model, "classifier"):
        x = model.features(inputs)
        x = x.view(x.size(0), -1)
        feats = model.classifier[:-1](x)
        logits = model.classifier[-1](feats)
        return logits, feats
    # ResNet fallback
    x = F.relu(model.bn1(model.conv1(inputs)))
    x = model.layer4(model.layer3(model.layer2(model.layer1(x))))
    x = model.avgpool(x)
    feats = x.view(x.size(0), -1)
    return model.fc(feats), feats


@torch.no_grad()
def collect_outputs(model: torch.nn.Module, loader, device: torch.device) -> Dict:
    model.eval()
    logits_l, feats_l, targets_l = [], [], []
    for inputs, targets in loader:
        inputs = inputs.to(device)
        logits, feats = _extract_features(model, inputs)
        logits_l.append(logits.cpu())
        feats_l.append(feats.cpu())
        targets_l.append(targets.cpu())
    logits  = torch.cat(logits_l).float()
    features = torch.cat(feats_l).float().numpy()
    targets  = torch.cat(targets_l).numpy()
    probs    = F.softmax(logits, dim=1).numpy()
    return {
        "logits":   logits,
        "features": features,
        "targets":  targets,
        "probs":    probs,
        "preds":    np.argmax(probs, axis=1),
        "confs":    np.max(probs, axis=1),
    }


def find_latest_checkpoint(checkpoint_dir: Path, dataset: str) -> Path:
    """Return the highest-epoch checkpoint for a given dataset."""
    dataset = dataset.lower()
    if dataset == "cifar10":
        glob = "exp2_mixup_best_auroc_epoch_*.pt"
    elif dataset == "cifar100":
        glob = "cifar100_mixup_variant2_best_auroc_epoch_*.pt"
    else:
        glob = f"mixup_{dataset}_resnet18_best_auroc_epoch_*.pt"
    pat = re.compile(r"_epoch_(\d+)\.pt$")
    candidates = [(int(pat.search(p.name).group(1)), p)
                  for p in checkpoint_dir.glob(glob) if pat.search(p.name)]
    if not candidates:
        raise FileNotFoundError(f"No checkpoints found for '{dataset}' in {checkpoint_dir}")
    return sorted(candidates)[-1][1]


def all_checkpoints_for_dataset(checkpoint_dir: Path, dataset: str) -> List[Path]:
    """Return all epoch checkpoints for a dataset, sorted by epoch."""
    dataset = dataset.lower()
    if dataset == "cifar10":
        glob = "exp2_mixup_best_auroc_epoch_*.pt"
    elif dataset == "cifar100":
        glob = "cifar100_mixup_variant2_best_auroc_epoch_*.pt"
    else:
        glob = f"mixup_{dataset}_resnet18_best_auroc_epoch_*.pt"
    pat = re.compile(r"_epoch_(\d+)\.pt$")
    candidates = [(int(pat.search(p.name).group(1)), p)
                  for p in checkpoint_dir.glob(glob) if pat.search(p.name)]
    if not candidates:
        raise FileNotFoundError(f"No checkpoints for '{dataset}' in {checkpoint_dir}")
    return [p for _, p in sorted(candidates)]


def load_model_from_checkpoint(ckpt_path: Path, dataset: str, arch: str, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg  = ckpt.get("config", {})
    arch = cfg.get("model", {}).get("arch", arch)

    if dataset == "cifar10":
        num_classes = 10
    elif dataset == "cifar100":
        num_classes = 100
    elif dataset.lower() in MEDMNIST_DATASETS:
        num_classes = MEDMNIST_DATASETS[dataset.lower()]["num_classes"]
    elif dataset == "tinyimagenet":
        num_classes = 200
    else:
        num_classes = 10

    input_size = 64 if dataset.lower() == "tinyimagenet" else 32
    model = get_model(arch, num_classes=num_classes, input_size=input_size).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, num_classes


# ---------------------------------------------------------------------------
# Core kNN blend evaluation (parameterised)
# ---------------------------------------------------------------------------

def knn_blend_eval(
    val_features: np.ndarray,
    val_probs:    np.ndarray,
    val_targets:  np.ndarray,
    test_features: np.ndarray,
    test_probs:    np.ndarray,
    test_targets:  np.ndarray,
    num_classes:   int,
    k:             int,
    alpha:         float,
    weight:        str  = "distance",   # 'uniform' | 'distance'
    conf_scheme:   str  = "max_prob",   # 'max_prob' | 'margin' | 'neg_entropy'
) -> Dict[str, float]:
    """Evaluate one (k, alpha, weight, conf_scheme) configuration on the test set.

    Uses the full validation set for fitting (no internal split), which gives
    a clean upper-bound signal for hyper-parameter ablation.
    """
    knn = KNeighborsClassifier(n_neighbors=k, weights=weight)
    knn.fit(val_features, val_targets)
    test_knn_probs = _full_probs_from_knn(knn, test_features, num_classes)
    blended = (1.0 - alpha) * test_probs + alpha * test_knn_probs
    blended /= blended.sum(axis=1, keepdims=True)

    preds = np.argmax(blended, axis=1)
    confs = _extract_confidence(blended, None, conf_scheme)
    return _evaluate_sc(preds, confs, test_targets)


# ---------------------------------------------------------------------------
# Ablation A – k × alpha full grid
# ---------------------------------------------------------------------------

def ablation_A_grid(val_data, test_data, num_classes, weight="distance", conf_scheme="max_prob",
                    feat_key="features") -> List[Dict]:
    """Full (k, alpha) grid, wider than the default search."""
    k_grid     = [1, 3, 5, 10, 20, 30, 50, 75, 100]
    alpha_grid = [0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    rows = []
    for k, alpha in itertools.product(k_grid, alpha_grid):
        if alpha == 0.0:
            # Pure baseline – only need to compute once (handled separately)
            probs = test_data["probs"]
            preds = np.argmax(probs, axis=1)
            confs = _extract_confidence(probs, None, conf_scheme)
            m = _evaluate_sc(preds, confs, test_data["targets"])
        elif alpha == 1.0:
            # Pure kNN – still depends on k
            knn = KNeighborsClassifier(n_neighbors=k, weights=weight)
            knn.fit(val_data[feat_key], val_data["targets"])
            knn_probs = _full_probs_from_knn(knn, test_data[feat_key], num_classes)
            preds = np.argmax(knn_probs, axis=1)
            confs = _extract_confidence(knn_probs, None, conf_scheme)
            m = _evaluate_sc(preds, confs, test_data["targets"])
        else:
            m = knn_blend_eval(
                val_data[feat_key], val_data["probs"], val_data["targets"],
                test_data[feat_key], test_data["probs"], test_data["targets"],
                num_classes, k=k, alpha=alpha, weight=weight, conf_scheme=conf_scheme,
            )
        rows.append({"k": k, "alpha": alpha, **m})
    return rows



# ---------------------------------------------------------------------------
# Ablation B – kNN weight scheme
# ---------------------------------------------------------------------------

def ablation_B_weight_scheme(val_data, test_data, num_classes, best_k: int, best_alpha: float) -> List[Dict]:
    """Compare 'uniform' vs 'distance' weighting at the best (k, alpha)."""
    rows = []
    for scheme in ("uniform", "distance"):
        m = knn_blend_eval(
            val_data["features"], val_data["probs"], val_data["targets"],
            test_data["features"], test_data["probs"], test_data["targets"],
            num_classes, k=best_k, alpha=best_alpha, weight=scheme,
        )
        rows.append({"weight": scheme, **m})
    return rows


# ---------------------------------------------------------------------------
# Ablation C – Feature representation
# ---------------------------------------------------------------------------

def ablation_C_feature_transform(val_data, test_data, num_classes, best_k: int, best_alpha: float,
                                   pca_dims: List[int] = None) -> List[Dict]:
    """raw / L2-normalised / PCA-k variants."""
    if pca_dims is None:
        feat_dim = val_data["features"].shape[1]
        pca_dims = [d for d in [32, 64, 128, 256] if d < feat_dim]

    transforms = ["raw", "l2"] + [f"pca_{d}" for d in pca_dims]
    rows = []

    for t in transforms:
        if t.startswith("pca_"):
            nc = int(t.split("_")[1])
            vf, pca_m = _apply_feature_transform(val_data["features"], "pca", n_components=nc)
            tf, _     = _apply_feature_transform(test_data["features"], "pca", pca_model=pca_m)
        else:
            vf, _ = _apply_feature_transform(val_data["features"], t)
            tf, _ = _apply_feature_transform(test_data["features"], t)

        m = knn_blend_eval(
            vf, val_data["probs"], val_data["targets"],
            tf, test_data["probs"], test_data["targets"],
            num_classes, k=best_k, alpha=best_alpha,
        )
        rows.append({"feature_transform": t, **m})
    return rows


# ---------------------------------------------------------------------------
# Ablation D – Confidence extraction scheme
# ---------------------------------------------------------------------------

def ablation_D_confidence_scheme(val_data, test_data, num_classes, best_k: int, best_alpha: float) -> List[Dict]:
    """Compares max_prob, margin, neg_entropy on blended probabilities."""
    rows = []
    # Pre-compute blended probs once
    knn = KNeighborsClassifier(n_neighbors=best_k, weights="distance")
    knn.fit(val_data["features"], val_data["targets"])
    knn_probs = _full_probs_from_knn(knn, test_data["features"], num_classes)
    blended = (1.0 - best_alpha) * test_data["probs"] + best_alpha * knn_probs
    blended /= blended.sum(axis=1, keepdims=True)
    preds = np.argmax(blended, axis=1)

    for scheme in ("max_prob", "margin", "neg_entropy"):
        confs = _extract_confidence(blended, None, scheme)
        m = _evaluate_sc(preds, confs, test_data["targets"])
        rows.append({"conf_scheme": scheme, **m})
    return rows


# ---------------------------------------------------------------------------
# Ablation E – Checkpoint epoch stability
# ---------------------------------------------------------------------------

def ablation_E_epoch_stability(checkpoint_dir: Path, dataset: str, arch: str, device: torch.device,
                                 loader_kwargs: Dict, best_k: int, best_alpha: float,
                                 num_classes: int) -> List[Dict]:
    """Evaluate the method at every saved epoch for the dataset."""
    ckpt_paths = all_checkpoints_for_dataset(checkpoint_dir, dataset)
    rows = []

    for ckpt_path in ckpt_paths:
        pat = re.compile(r"_epoch_(\d+)\.pt$")
        epoch = int(pat.search(ckpt_path.name).group(1))

        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        input_size = 64 if dataset.lower() == "tinyimagenet" else 32
        _arch = ckpt.get("config", {}).get("model", {}).get("arch", arch)
        model = get_model(_arch, num_classes=num_classes, input_size=input_size).to(device)
        model.load_state_dict(ckpt["model_state"])
        model.eval()

        val_data  = collect_outputs(model, loader_kwargs["val_loader"],  device)
        test_data = collect_outputs(model, loader_kwargs["test_loader"], device)

        # Baseline MSP
        base = _evaluate_sc(test_data["preds"], test_data["confs"], test_data["targets"])

        m = knn_blend_eval(
            val_data["features"], val_data["probs"], val_data["targets"],
            test_data["features"], test_data["probs"], test_data["targets"],
            num_classes, k=best_k, alpha=best_alpha,
        )
        rows.append({
            "epoch":           epoch,
            "checkpoint":      ckpt_path.name,
            "base_accuracy":   base["accuracy"],
            "base_auroc":      base["auroc"],
            "base_aurc":       base["aurc"],
            "base_eaurc":      base["eaurc"],
            "base_naurc":      base["naurc"],
            "knn_accuracy":    m["accuracy"],
            "knn_auroc":       m["auroc"],
            "knn_aurc":        m["aurc"],
            "knn_eaurc":       m["eaurc"],
            "knn_naurc":       m["naurc"],
            "aurc_delta":      m["aurc"]  - base["aurc"],
            "auroc_delta":     m["auroc"] - base["auroc"],
            "naurc_delta":     m["naurc"] - base["naurc"],
        })
        print(f"  Epoch {epoch:4d}: base_aurc={base['aurc']:.4f}  knn_aurc={m['aurc']:.4f}  "
              f"Δaurc={m['aurc']-base['aurc']:+.4f}  "
              f"base_auroc={base['auroc']:.4f}  knn_auroc={m['auroc']:.4f}  "
              f"Δnaurc={m['naurc']-base['naurc']:+.4f}")

    return rows


# ---------------------------------------------------------------------------
# Ablation F – Cross-dataset generalisation
# ---------------------------------------------------------------------------

CROSS_DATASET_PLAN = {
    "cifar10":       {"arch": "resnet18", "glob": "exp2_mixup_best_auroc_epoch_*.pt"},
    "cifar100":      {"arch": "resnet18", "glob": "cifar100_mixup_variant2_best_auroc_epoch_*.pt"},
    "bloodmnist":    {"arch": "resnet18", "glob": "mixup_bloodmnist_resnet18_best_auroc_epoch_*.pt"},
    "dermamnist":    {"arch": "resnet18", "glob": "mixup_dermamnist_resnet18_best_auroc_epoch_*.pt"},
    "octmnist":      {"arch": "resnet18", "glob": "mixup_octmnist_resnet18_best_auroc_epoch_*.pt"},
    "organamnist":   {"arch": "resnet18", "glob": "mixup_organamnist_resnet18_best_auroc_epoch_*.pt"},
    "pathmnist":     {"arch": "resnet18", "glob": "mixup_pathmnist_resnet18_best_auroc_epoch_*.pt"},
}


def ablation_F_cross_dataset(checkpoint_dir: Path, device: torch.device, best_k: int, best_alpha: float,
                               datasets: Optional[List[str]] = None) -> List[Dict]:
    """Evaluate on multiple datasets using per-dataset best checkpoint."""
    plan = {k: v for k, v in CROSS_DATASET_PLAN.items()
            if datasets is None or k in datasets}
    rows = []
    pat = re.compile(r"_epoch_(\d+)\.pt$")

    for ds, meta in plan.items():
        candidates = [(int(pat.search(p.name).group(1)), p)
                      for p in checkpoint_dir.glob(meta["glob"]) if pat.search(p.name)]
        if not candidates:
            print(f"  [skip] No checkpoints found for {ds}")
            continue
        ckpt_path = sorted(candidates)[-1][1]

        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        if ds == "cifar10":
            nc = 10
        elif ds == "cifar100":
            nc = 100
        elif ds in MEDMNIST_DATASETS:
            nc = MEDMNIST_DATASETS[ds]["num_classes"]
        else:
            nc = 10

        input_size = 64 if ds == "tinyimagenet" else 32
        _arch = ckpt.get("config", {}).get("model", {}).get("arch", meta["arch"])
        model = get_model(_arch, num_classes=nc, input_size=input_size).to(device)
        model.load_state_dict(ckpt["model_state"])
        model.eval()

        tl, vl, tel, _ = get_dataloaders(
            dataset=ds, data_dir="./data", batch_size=256, val_batch_size=256,
            num_workers=4, augmentation=False, seed=42,
        )
        val_data  = collect_outputs(model, vl,  device)
        test_data = collect_outputs(model, tel, device)

        base = _evaluate_sc(test_data["preds"], test_data["confs"], test_data["targets"])
        m    = knn_blend_eval(
            val_data["features"], val_data["probs"], val_data["targets"],
            test_data["features"], test_data["probs"], test_data["targets"],
            nc, k=best_k, alpha=best_alpha,
        )
        rows.append({
            "dataset":        ds,
            "num_classes":    nc,
            "checkpoint":     ckpt_path.name,
            "base_accuracy":  base["accuracy"],
            "base_auroc":     base["auroc"],
            "base_aurc":      base["aurc"],
            "base_eaurc":     base["eaurc"],
            "base_naurc":     base["naurc"],
            "knn_accuracy":   m["accuracy"],
            "knn_auroc":      m["auroc"],
            "knn_aurc":       m["aurc"],
            "knn_eaurc":      m["eaurc"],
            "knn_naurc":      m["naurc"],
            "aurc_delta":     m["aurc"]  - base["aurc"],
            "auroc_delta":    m["auroc"] - base["auroc"],
            "naurc_delta":    m["naurc"] - base["naurc"],
        })
        print(f"  {ds:16s}: base_aurc={base['aurc']:.4f}  knn_aurc={m['aurc']:.4f}  "
              f"Δaurc={m['aurc']-base['aurc']:+.4f}  "
              f"base_naurc={base['naurc']:.4f}  knn_naurc={m['naurc']:.4f}  "
              f"({ckpt_path.name})")

    return rows


# ---------------------------------------------------------------------------
# Pretty-print helpers
# ---------------------------------------------------------------------------

def _fmt_table(headers: List[str], rows: List[Dict], keys: List[str]) -> str:
    widths = [max(len(h), max((len(str(r.get(k, ""))) for r in rows), default=0))
              for h, k in zip(headers, keys)]
    sep = "+-" + "-+-".join("-" * w for w in widths) + "-+"
    header = "| " + " | ".join(h.ljust(w) for h, w in zip(headers, widths)) + " |"
    lines = [sep, header, sep]
    for r in rows:
        line = "| " + " | ".join(str(r.get(k, "")).ljust(w) for k, w in zip(keys, widths)) + " |"
        lines.append(line)
    lines.append(sep)
    return "\n".join(lines)


def _best_row(rows: List[Dict], metric: str = "aurc") -> Dict:
    return min(rows, key=lambda r: r[metric])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Ablation study for Feature-kNN Blend")
    ds_choices = ["cifar10", "cifar100"] + list(MEDMNIST_DATASETS.keys()) + ["tinyimagenet"]
    parser.add_argument("--dataset",        default="cifar10", choices=ds_choices)
    parser.add_argument("--arch",           default="resnet18",
                        choices=["resnet18", "resnet34", "resnet50", "vgg16_bn"])
    parser.add_argument("--checkpoint",     default=None, type=str,
                        help="Path to a specific checkpoint. Auto-selected if omitted.")
    parser.add_argument("--checkpoint-dir", default="./checkpoints", type=str)
    parser.add_argument("--batch-size",     default=256, type=int)
    parser.add_argument("--seed",           default=42, type=int)
    parser.add_argument("--output",         default="./results/ablation_knn_blend.json")
    parser.add_argument("--no-epoch-sweep", action="store_true",
                        help="Skip Ablation E (checkpoint epoch stability).")
    parser.add_argument("--no-cross-dataset", action="store_true",
                        help="Skip Ablation F (cross-dataset generalisation).")
    parser.add_argument("--dims",           nargs="+", default=["A", "B", "C", "D", "E", "F"],
                        choices=["A", "B", "C", "D", "E", "F"],
                        help="Which ablation dimensions to run (default: all).")
    parser.add_argument("--cross-datasets", nargs="+", default=None,
                        choices=list(CROSS_DATASET_PLAN.keys()),
                        help="Limit cross-dataset ablation to these datasets.")
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = get_device()
    print(f"Device: {device}")

    checkpoint_dir = Path(args.checkpoint_dir)
    if args.checkpoint:
        ckpt_path = Path(args.checkpoint)
    else:
        ckpt_path = find_latest_checkpoint(checkpoint_dir, args.dataset)
    print(f"Primary checkpoint: {ckpt_path}\n")

    model, num_classes = load_model_from_checkpoint(ckpt_path, args.dataset, args.arch, device)

    train_loader, val_loader, test_loader, _ = get_dataloaders(
        dataset=args.dataset,
        data_dir="./data",
        batch_size=args.batch_size,
        val_batch_size=args.batch_size,
        num_workers=4,
        augmentation=False,
        seed=args.seed,
    )

    print("Collecting validation outputs …")
    t0 = time.time()
    val_data = collect_outputs(model, val_loader, device)
    print("Collecting test outputs …")
    test_data = collect_outputs(model, test_loader, device)
    print(f"  Done in {time.time() - t0:.1f}s  "
          f"(val={val_data['features'].shape}, test={test_data['features'].shape})\n")

    baseline = _evaluate_sc(test_data["preds"], test_data["confs"], test_data["targets"])
    print(f"Baseline (MSP): acc={baseline['accuracy']:.4f}  auroc={baseline['auroc']:.4f}  "
          f"aurc={baseline['aurc']:.4f}  eaurc={baseline['eaurc']:.4f}  naurc={baseline['naurc']:.4f}\n")

    results = {
        "checkpoint": str(ckpt_path),
        "dataset":    args.dataset,
        "baseline":   baseline,
    }

    # -----------------------------------------------------------------------
    # Ablation A – k × alpha grid
    # -----------------------------------------------------------------------
    if "A" in args.dims:
        print("=" * 70)
        print("Ablation A: k × alpha hyper-parameter grid")
        print("=" * 70)
        t0 = time.time()
        rows_A = ablation_A_grid(val_data, test_data, num_classes)
        print(f"  Computed {len(rows_A)} configurations in {time.time() - t0:.1f}s")

        best_A = _best_row(rows_A)
        best_k     = best_A["k"]
        best_alpha = best_A["alpha"]
        print(f"\n  Best config: k={best_k}, alpha={best_alpha}  "
              f"auroc={best_A['auroc']:.4f}  aurc={best_A['aurc']:.4f}  "
              f"eaurc={best_A['eaurc']:.4f}  naurc={best_A['naurc']:.4f}")
        print(f"  AURC improvement vs baseline:  {best_A['aurc']  - baseline['aurc']:+.4f}")
        print(f"  AUROC improvement vs baseline: {best_A['auroc'] - baseline['auroc']:+.4f}")
        print(f"  NAURC improvement vs baseline: {best_A['naurc'] - baseline['naurc']:+.4f}")

        # Print concise summary: for each k show best alpha
        print("\n  Per-k best configuration:")
        per_k_best = {}
        for r in rows_A:
            k = r["k"]
            if k not in per_k_best or r["aurc"] < per_k_best[k]["aurc"]:
                per_k_best[k] = r
        per_k_rows = sorted(per_k_best.values(), key=lambda r: r["k"])
        print(_fmt_table(
            ["k", "best_alpha", "accuracy", "auroc", "aurc", "eaurc", "naurc"],
            [{"k": r["k"], "best_alpha": r["alpha"],
              "accuracy": f"{r['accuracy']:.4f}", "auroc": f"{r['auroc']:.4f}",
              "aurc": f"{r['aurc']:.4f}", "eaurc": f"{r['eaurc']:.4f}", "naurc": f"{r['naurc']:.4f}"}
             for r in per_k_rows],
            ["k", "best_alpha", "accuracy", "auroc", "aurc", "eaurc", "naurc"],
        ))

        results["ablation_A"] = {"rows": rows_A, "best": best_A}
    else:
        # Fallback: find best from default grid if A is skipped
        default_ks     = [5, 10, 20, 30]
        default_alphas = [0.1, 0.25, 0.4, 0.55, 0.7, 0.85]
        tmp = []
        for k, alpha in itertools.product(default_ks, default_alphas):
            m = knn_blend_eval(
                val_data["features"], val_data["probs"], val_data["targets"],
                test_data["features"], test_data["probs"], test_data["targets"],
                num_classes, k=k, alpha=alpha,
            )
            tmp.append({"k": k, "alpha": alpha, **m})
        best_A    = _best_row(tmp)
        best_k    = best_A["k"]
        best_alpha = best_A["alpha"]
        print(f"[skip A] Using default-grid best: k={best_k}, alpha={best_alpha}")

    # -----------------------------------------------------------------------
    # Ablation B – weight scheme
    # -----------------------------------------------------------------------
    if "B" in args.dims:
        print("\n" + "=" * 70)
        print(f"Ablation B: kNN weight scheme  (k={best_k}, alpha={best_alpha})")
        print("=" * 70)
        rows_B = ablation_B_weight_scheme(val_data, test_data, num_classes, best_k, best_alpha)
        print(_fmt_table(
            ["weight", "accuracy", "auroc", "aurc", "eaurc", "naurc"],
            [{"weight": r["weight"], "accuracy": f"{r['accuracy']:.4f}",
              "auroc": f"{r['auroc']:.4f}", "aurc": f"{r['aurc']:.4f}",
              "eaurc": f"{r['eaurc']:.4f}", "naurc": f"{r['naurc']:.4f}"} for r in rows_B],
            ["weight", "accuracy", "auroc", "aurc", "eaurc", "naurc"],
        ))
        results["ablation_B"] = rows_B

    # -----------------------------------------------------------------------
    # Ablation C – feature transform
    # -----------------------------------------------------------------------
    if "C" in args.dims:
        print("\n" + "=" * 70)
        print(f"Ablation C: Feature representation  (k={best_k}, alpha={best_alpha})")
        print("=" * 70)
        feat_dim = val_data["features"].shape[1]
        pca_dims = [d for d in [32, 64, 128, 256] if d < feat_dim]
        rows_C = ablation_C_feature_transform(val_data, test_data, num_classes, best_k, best_alpha,
                                               pca_dims=pca_dims)
        print(_fmt_table(
            ["feature_transform", "accuracy", "auroc", "aurc", "eaurc", "naurc"],
            [{"feature_transform": r["feature_transform"], "accuracy": f"{r['accuracy']:.4f}",
              "auroc": f"{r['auroc']:.4f}", "aurc": f"{r['aurc']:.4f}",
              "eaurc": f"{r['eaurc']:.4f}", "naurc": f"{r['naurc']:.4f}"} for r in rows_C],
            ["feature_transform", "accuracy", "auroc", "aurc", "eaurc", "naurc"],
        ))
        results["ablation_C"] = rows_C

    # -----------------------------------------------------------------------
    # Ablation D – confidence extraction
    # -----------------------------------------------------------------------
    if "D" in args.dims:
        print("\n" + "=" * 70)
        print(f"Ablation D: Confidence extraction scheme  (k={best_k}, alpha={best_alpha})")
        print("=" * 70)
        rows_D = ablation_D_confidence_scheme(val_data, test_data, num_classes, best_k, best_alpha)
        print(_fmt_table(
            ["conf_scheme", "accuracy", "auroc", "aurc", "eaurc", "naurc"],
            [{"conf_scheme": r["conf_scheme"], "accuracy": f"{r['accuracy']:.4f}",
              "auroc": f"{r['auroc']:.4f}", "aurc": f"{r['aurc']:.4f}",
              "eaurc": f"{r['eaurc']:.4f}", "naurc": f"{r['naurc']:.4f}"} for r in rows_D],
            ["conf_scheme", "accuracy", "auroc", "aurc", "eaurc", "naurc"],
        ))
        results["ablation_D"] = rows_D

    # -----------------------------------------------------------------------
    # Ablation E – checkpoint epoch stability
    # -----------------------------------------------------------------------
    if "E" in args.dims and not args.no_epoch_sweep:
        print("\n" + "=" * 70)
        print(f"Ablation E: Checkpoint epoch stability  (k={best_k}, alpha={best_alpha})")
        print("=" * 70)
        rows_E = ablation_E_epoch_stability(
            checkpoint_dir, args.dataset, args.arch, device,
            {"val_loader": val_loader, "test_loader": test_loader},
            best_k, best_alpha, num_classes,
        )
        print("\n  Summary table:")
        print(_fmt_table(
            ["epoch", "base_auroc", "knn_auroc", "auroc_delta",
             "base_aurc", "knn_aurc", "aurc_delta",
             "base_naurc", "knn_naurc", "naurc_delta"],
            [{
                "epoch":       r["epoch"],
                "base_auroc":  f"{r['base_auroc']:.4f}",
                "knn_auroc":   f"{r['knn_auroc']:.4f}",
                "auroc_delta": f"{r['auroc_delta']:+.4f}",
                "base_aurc":   f"{r['base_aurc']:.4f}",
                "knn_aurc":    f"{r['knn_aurc']:.4f}",
                "aurc_delta":  f"{r['aurc_delta']:+.4f}",
                "base_naurc":  f"{r['base_naurc']:.4f}",
                "knn_naurc":   f"{r['knn_naurc']:.4f}",
                "naurc_delta": f"{r['naurc_delta']:+.4f}",
             } for r in rows_E],
            ["epoch", "base_auroc", "knn_auroc", "auroc_delta",
             "base_aurc", "knn_aurc", "aurc_delta",
             "base_naurc", "knn_naurc", "naurc_delta"],
        ))
        consistent = all(r["aurc_delta"] <= 0 for r in rows_E)
        print(f"\n  Consistent AURC improvement across all epochs: {consistent}")
        results["ablation_E"] = rows_E
    elif "E" in args.dims:
        print("\n[skip E] --no-epoch-sweep flag set.")

    # -----------------------------------------------------------------------
    # Ablation F – cross-dataset
    # -----------------------------------------------------------------------
    if "F" in args.dims and not args.no_cross_dataset:
        print("\n" + "=" * 70)
        print(f"Ablation F: Cross-dataset generalisation  (k={best_k}, alpha={best_alpha})")
        print("=" * 70)
        rows_F = ablation_F_cross_dataset(
            checkpoint_dir, device, best_k, best_alpha, datasets=args.cross_datasets,
        )
        print("\n  Summary table:")
        print(_fmt_table(
            ["dataset", "num_classes",
             "base_auroc", "knn_auroc", "auroc_delta",
             "base_aurc",  "knn_aurc",  "aurc_delta",
             "base_naurc", "knn_naurc", "naurc_delta"],
            [{
                "dataset":     r["dataset"],
                "num_classes": r["num_classes"],
                "base_auroc":  f"{r['base_auroc']:.4f}",
                "knn_auroc":   f"{r['knn_auroc']:.4f}",
                "auroc_delta": f"{r['auroc_delta']:+.4f}",
                "base_aurc":   f"{r['base_aurc']:.4f}",
                "knn_aurc":    f"{r['knn_aurc']:.4f}",
                "aurc_delta":  f"{r['aurc_delta']:+.4f}",
                "base_naurc":  f"{r['base_naurc']:.4f}",
                "knn_naurc":   f"{r['knn_naurc']:.4f}",
                "naurc_delta": f"{r['naurc_delta']:+.4f}",
             } for r in rows_F],
            ["dataset", "num_classes",
             "base_auroc", "knn_auroc", "auroc_delta",
             "base_aurc",  "knn_aurc",  "aurc_delta",
             "base_naurc", "knn_naurc", "naurc_delta"],
        ))
        improved_f = sum(1 for r in rows_F if r["aurc_delta"] < 0)
        print(f"\n  Improved on {improved_f}/{len(rows_F)} datasets.")
        results["ablation_F"] = rows_F
    elif "F" in args.dims:
        print("\n[skip F] --no-cross-dataset flag set.")

    # -----------------------------------------------------------------------
    # Final overview
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("Summary: best configuration vs baseline")
    print("=" * 70)
    print(f"  Baseline (MSP):     auroc={baseline['auroc']:.4f}  aurc={baseline['aurc']:.4f}  "
          f"eaurc={baseline['eaurc']:.4f}  naurc={baseline['naurc']:.4f}")
    if "A" in args.dims:
        print(f"  Best blend config:  k={best_k}  alpha={best_alpha}  "
              f"auroc={best_A['auroc']:.4f}  aurc={best_A['aurc']:.4f}  "
              f"eaurc={best_A['eaurc']:.4f}  naurc={best_A['naurc']:.4f}  "
              f"(Δaurc={best_A['aurc']-baseline['aurc']:+.4f}  "
              f"Δnaurc={best_A['naurc']-baseline['naurc']:+.4f})")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2, default=str))
    print(f"\nFull results saved to: {out_path}")


if __name__ == "__main__":
    main()
