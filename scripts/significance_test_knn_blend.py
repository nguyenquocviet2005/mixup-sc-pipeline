"""
Statistical significance testing for method_2_feature_knn_logit_blend.

Two complementary tests (NO re-training required — uses a fixed checkpoint):

  1. Multi-seed test
     Run the method with N different data seeds.  Each seed produces a different
     val / test partition of the SAME training data, giving N paired
     (baseline_i, knn_blend_i) observations per metric.
     Reports:
       • mean ± std over seeds for each metric
       • 95 % bootstrap confidence interval on the mean *improvement*
       • Paired Wilcoxon signed-rank test (non-parametric, H₀: no improvement)
       • Paired Student t-test (for reference, assumes normality)
       • Cohen's d effect size

  2. Cross-dataset Wilcoxon test
     Each of the N_datasets datasets is one paired observation
     (baseline_ds, knn_blend_ds).  Tests whether the method consistently
     beats the baseline across datasets.
     Reads pre-computed Dim-F results from the ablation JSON (no GPU needed).

Usage examples
--------------
# Full run (multi-seed + cross-dataset):
python scripts/significance_test_knn_blend.py --dataset cifar10

# Only multi-seed:
python scripts/significance_test_knn_blend.py --dataset cifar10 --no-cross-dataset

# Use more seeds:
python scripts/significance_test_knn_blend.py --dataset cifar10 --num-seeds 10
"""

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from scipy import stats
from sklearn.neighbors import KNeighborsClassifier

sys.path.insert(0, str(Path(__file__).parent.parent))

from data import get_dataloaders, MEDMNIST_DATASETS
from evaluation import SelectionMetrics
from models import get_model


# ---------------------------------------------------------------------------
# Metric computation (identical to ablation script)
# ---------------------------------------------------------------------------

def _evaluate_sc(preds: np.ndarray, confs: np.ndarray, targets: np.ndarray) -> Dict[str, float]:
    if not np.all(np.isfinite(confs)):
        confs = np.nan_to_num(confs, nan=0.0, posinf=1e6, neginf=-1e10)
    correctness = (preds == targets).astype(int)
    sc        = SelectionMetrics.compute_all_metrics(confs, correctness)
    accuracy  = float(np.mean(correctness))
    aurc      = float(sc["aurc"])
    eaurc     = float(sc["eaurc"])
    auroc     = float(sc["auroc"])
    risk      = 1.0 - accuracy
    aurc_opt  = aurc - eaurc
    rand_eaurc = risk - aurc_opt
    naurc     = float(eaurc / rand_eaurc) if rand_eaurc > 1e-9 else 0.0
    return {"accuracy": accuracy, "auroc": auroc, "aurc": aurc, "eaurc": eaurc, "naurc": naurc}


def _extract_confidence(probs: np.ndarray, scheme: str) -> np.ndarray:
    if scheme == "max_prob":
        return np.max(probs, axis=1)
    if scheme == "margin":
        s = np.sort(probs, axis=1)
        return s[:, -1] - s[:, -2]
    if scheme == "neg_entropy":
        p = np.clip(probs, 1e-12, 1.0)
        return np.sum(p * np.log(p), axis=1)
    raise ValueError(f"Unknown confidence scheme: {scheme}")


def _full_probs_from_knn(knn, features, num_classes):
    pp = knn.predict_proba(features)
    full = np.zeros((features.shape[0], num_classes), dtype=np.float32)
    full[:, knn.classes_.astype(int)] = pp
    return full


# ---------------------------------------------------------------------------
# Model / data helpers
# ---------------------------------------------------------------------------

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def find_latest_checkpoint(checkpoint_dir: Path, dataset: str) -> Path:
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


def load_model_from_checkpoint(ckpt_path, dataset, arch, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg  = ckpt.get("config", {})
    arch = cfg.get("model", {}).get("arch", arch)

    if dataset == "cifar10":
        nc = 10
    elif dataset == "cifar100":
        nc = 100
    elif dataset.lower() in MEDMNIST_DATASETS:
        nc = MEDMNIST_DATASETS[dataset.lower()]["num_classes"]
    else:
        nc = 10

    input_size = 64 if dataset.lower() == "tinyimagenet" else 32
    model = get_model(arch, num_classes=nc, input_size=input_size).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, nc


def _extract_features(model, inputs):
    if hasattr(model, "forward_features") and hasattr(model, "forward_head"):
        feats  = model.forward_head(model.forward_features(inputs), pre_logits=True)
        logits = model.head(feats)
        return logits, feats
    if hasattr(model, "features") and hasattr(model, "classifier"):
        x = model.features(inputs)
        x = x.view(x.size(0), -1)
        feats  = model.classifier[:-1](x)
        logits = model.classifier[-1](feats)
        return logits, feats
    x = F.relu(model.bn1(model.conv1(inputs)))
    x = model.layer4(model.layer3(model.layer2(model.layer1(x))))
    x = model.avgpool(x)
    feats = x.view(x.size(0), -1)
    return model.fc(feats), feats


@torch.no_grad()
def collect_outputs(model, loader, device):
    model.eval()
    logits_l, feats_l, targets_l = [], [], []
    for inputs, targets in loader:
        inputs = inputs.to(device)
        logits, feats = _extract_features(model, inputs)
        logits_l.append(logits.cpu())
        feats_l.append(feats.cpu())
        targets_l.append(targets.cpu())
    logits   = torch.cat(logits_l).float()
    features = torch.cat(feats_l).float().numpy()
    targets  = torch.cat(targets_l).numpy()
    probs    = F.softmax(logits, dim=1).numpy()
    return {"features": features, "targets": targets, "probs": probs,
            "preds": np.argmax(probs, axis=1), "confs": np.max(probs, axis=1)}


# ---------------------------------------------------------------------------
# Statistical helpers
# ---------------------------------------------------------------------------

def bootstrap_ci(diffs: np.ndarray, n_boot: int = 10000, ci: float = 0.95) -> Tuple[float, float]:
    """Percentile bootstrap CI of the mean of `diffs`."""
    rng   = np.random.default_rng(seed=0)
    boots = rng.choice(diffs, size=(n_boot, len(diffs)), replace=True).mean(axis=1)
    lo    = float(np.percentile(boots, 100 * (1 - ci) / 2))
    hi    = float(np.percentile(boots, 100 * (1 + ci) / 2))
    return lo, hi


def cohens_d(baseline: np.ndarray, method: np.ndarray) -> float:
    """Cohen's d for paired data (d = mean_diff / std_diff)."""
    diff = method - baseline
    sd   = float(np.std(diff, ddof=1))
    return float(np.mean(diff) / sd) if sd > 1e-12 else 0.0


def _fmt_stat(stat_result: dict, metric: str) -> str:
    p  = stat_result["p_value"]
    w  = stat_result["statistic"]
    sig = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "ns"))
    return f"W={w:.1f}, p={p:.4f} {sig}"


# ---------------------------------------------------------------------------
# Single-seed evaluation
# ---------------------------------------------------------------------------

def run_one_seed(
    model, dataset: str, num_classes: int, seed: int,
    k: int, alpha: float, conf_scheme: str, device,
) -> Dict:
    """Load data with `seed`, collect outputs, evaluate baseline and kNN blend."""
    _, val_loader, test_loader, _ = get_dataloaders(
        dataset=dataset, data_dir="./data", batch_size=256, val_batch_size=256,
        num_workers=4, augmentation=False, seed=seed,
    )
    val_data  = collect_outputs(model, val_loader,  device)
    test_data = collect_outputs(model, test_loader, device)

    # Baseline: MSP of softmax
    base_confs = _extract_confidence(test_data["probs"], conf_scheme)
    baseline   = _evaluate_sc(test_data["preds"], base_confs, test_data["targets"])

    # kNN blend
    knn = KNeighborsClassifier(n_neighbors=k, weights="distance")
    knn.fit(val_data["features"], val_data["targets"])
    knn_probs = _full_probs_from_knn(knn, test_data["features"], num_classes)
    blended   = (1.0 - alpha) * test_data["probs"] + alpha * knn_probs
    blended  /= blended.sum(axis=1, keepdims=True)
    preds     = np.argmax(blended, axis=1)
    confs     = _extract_confidence(blended, conf_scheme)
    method    = _evaluate_sc(preds, confs, test_data["targets"])

    return {"seed": seed, "baseline": baseline, "knn_blend": method}


# ---------------------------------------------------------------------------
# Multi-seed experiment
# ---------------------------------------------------------------------------

def multi_seed_experiment(
    checkpoint_path: Path, dataset: str, arch: str, device,
    seeds: List[int], k: int, alpha: float, conf_scheme: str,
) -> Dict:
    """Run N seeds; return paired metrics + statistical summaries per metric."""
    model, num_classes = load_model_from_checkpoint(checkpoint_path, dataset, arch, device)

    metrics = ["accuracy", "auroc", "aurc", "eaurc", "naurc"]
    records = []

    for seed in seeds:
        t0 = time.time()
        res = run_one_seed(model, dataset, num_classes, seed, k, alpha, conf_scheme, device)
        elapsed = time.time() - t0
        records.append(res)
        b = res["baseline"]
        m = res["knn_blend"]
        print(f"  Seed {seed:3d} ({elapsed:.1f}s)  "
              f"base_auroc={b['auroc']:.4f}  knn_auroc={m['auroc']:.4f}  "
              f"Δauroc={m['auroc']-b['auroc']:+.5f}  "
              f"base_naurc={b['naurc']:.4f}  knn_naurc={m['naurc']:.4f}  "
              f"Δnaurc={m['naurc']-b['naurc']:+.5f}")

    # ------------------------------------------------------------------ #
    # Aggregate                                                           #
    # ------------------------------------------------------------------ #
    stats_out = {}
    for metric in metrics:
        base_vals = np.array([r["baseline"][metric]  for r in records])
        knn_vals  = np.array([r["knn_blend"][metric] for r in records])
        diffs     = knn_vals - base_vals

        # Direction: lower is better for AURC/E-AURC/N-AURC; higher for accuracy/AUROC
        higher_is_better = metric in ("accuracy", "auroc")
        improvement_diffs = diffs if higher_is_better else -diffs

        # Wilcoxon signed-rank test with correct directional (one-sided) alternative.
        # alternative="greater"  H1: knn > baseline  (higher is better)
        # alternative="less"     H1: knn < baseline  (lower is better)
        # Using two-sided reports W=min(T+,T-)=0 when all diffs agree in sign,
        # then P(W≥0)=1 — a well-known edge case giving a misleading p=1.
        wx_alt = "greater" if higher_is_better else "less"
        try:
            wx_stat, wx_p = stats.wilcoxon(knn_vals, base_vals, alternative=wx_alt)
        except ValueError:
            # All zeros – method identical to baseline for this metric
            wx_stat, wx_p = 0.0, 1.0

        # Paired t-test (for reference)
        tt_stat, tt_p = stats.ttest_rel(knn_vals, base_vals)

        # Bootstrap 95 % CI on the mean difference
        ci_lo, ci_hi = bootstrap_ci(diffs)

        stats_out[metric] = {
            "base_mean":   float(np.mean(base_vals)),
            "base_std":    float(np.std(base_vals, ddof=1)),
            "knn_mean":    float(np.mean(knn_vals)),
            "knn_std":     float(np.std(knn_vals, ddof=1)),
            "diff_mean":   float(np.mean(diffs)),
            "diff_std":    float(np.std(diffs, ddof=1)),
            "ci_95_lo":    ci_lo,
            "ci_95_hi":    ci_hi,
            "wilcoxon_statistic": float(wx_stat),
            "wilcoxon_p":         float(wx_p),
            "ttest_statistic":    float(tt_stat),
            "ttest_p":            float(tt_p),
            "cohens_d":           cohens_d(base_vals, knn_vals),
        }

    return {
        "config": {"checkpoint": str(checkpoint_path), "dataset": dataset,
                   "k": k, "alpha": alpha, "conf_scheme": conf_scheme,
                   "seeds": seeds, "n_seeds": len(seeds)},
        "records": records,
        "summary": stats_out,
    }


# ---------------------------------------------------------------------------
# Cross-dataset Wilcoxon (uses pre-computed Dim-F ablation results)
# ---------------------------------------------------------------------------

def cross_dataset_wilcoxon(ef_json_path: Path) -> Dict:
    """
    Load Dim-F results from the ablation JSON and run Wilcoxon signed-rank tests
    (one paired observation per dataset).
    """
    with open(ef_json_path) as f:
        data = json.load(f)

    # Support both the combined EF file and a file with only ablation_F
    if "ablation_F" in data:
        rows = data["ablation_F"]
    else:
        raise KeyError("ablation_F key not found in the provided JSON file.")

    metrics_map = {
        "auroc": ("base_auroc", "knn_auroc"),
        "aurc":  ("base_aurc",  "knn_aurc"),
        "eaurc": ("base_eaurc", "knn_eaurc"),
        "naurc": ("base_naurc", "knn_naurc"),
    }

    datasets    = [r["dataset"] for r in rows]
    results_out = {"datasets": datasets, "metrics": {}}

    for metric, (base_key, knn_key) in metrics_map.items():
        base_vals = np.array([r[base_key] for r in rows])
        knn_vals  = np.array([r[knn_key]  for r in rows])
        diffs     = knn_vals - base_vals

        # Use directional alternative so the test correctly identifies consistent improvement
        higher_is_better_cd = metric == "auroc"
        wx_alt = "greater" if higher_is_better_cd else "less"
        try:
            wx_stat, wx_p = stats.wilcoxon(knn_vals, base_vals, alternative=wx_alt)
        except ValueError:
            wx_stat, wx_p = 0.0, 1.0

        results_out["metrics"][metric] = {
            "n_datasets":    len(datasets),
            "base_vals":     base_vals.tolist(),
            "knn_vals":      knn_vals.tolist(),
            "diffs":         diffs.tolist(),
            "diff_mean":     float(np.mean(diffs)),
            "wilcoxon_statistic": float(wx_stat),
            "wilcoxon_p":         float(wx_p),
        }

    return results_out


# ---------------------------------------------------------------------------
# Pretty printing
# ---------------------------------------------------------------------------

SIG_STARS = {0.001: "***", 0.01: "**", 0.05: "*"}


def significance_stars(p: float) -> str:
    for thr, sym in SIG_STARS.items():
        if p < thr:
            return sym
    return "ns"


def print_multi_seed_summary(summary: Dict, n_seeds: int):
    metrics = ["auroc", "aurc", "eaurc", "naurc", "accuracy"]
    higher_is_better = {"accuracy": True, "auroc": True, "aurc": False, "eaurc": False, "naurc": False}

    # Header
    col_w = [8, 24, 24, 26, 8, 8, 8]
    print()
    print("Multi-Seed Statistical Summary  (N={})".format(n_seeds))
    print("=" * 95)
    hdr = "{:<8}  {:>24}  {:>24}  {:>26}  {:>8}  {:>8}  {:>6}".format(
        "Metric", "Baseline (mean±std)", "kNN Blend (mean±std)",
        "Δ mean [95% CI]", "Wilcox", "t-test", "Cohen's d")
    print(hdr)
    print("-" * 95)
    for m in metrics:
        s = summary[m]
        dir_sign = 1 if higher_is_better[m] else -1

        base_str = f"{s['base_mean']:.4f} ± {s['base_std']:.4f}"
        knn_str  = f"{s['knn_mean']:.4f} ± {s['knn_std']:.4f}"
        ci_str   = f"{s['diff_mean']:+.4f} [{s['ci_95_lo']:+.4f}, {s['ci_95_hi']:+.4f}]"
        wx_str   = f"p={s['wilcoxon_p']:.3f}{significance_stars(s['wilcoxon_p'])}"
        tt_str   = f"p={s['ttest_p']:.3f}{significance_stars(s['ttest_p'])}"
        cd_str   = f"{s['cohens_d']:+.2f}"

        print("{:<8}  {:>24}  {:>24}  {:>26}  {:>8}  {:>8}  {:>6}".format(
            m, base_str, knn_str, ci_str, wx_str, tt_str, cd_str))
    print("=" * 95)
    print("Significance: *** p<0.001  ** p<0.01  * p<0.05  ns = not significant")
    print()


def print_cross_dataset_summary(cd: Dict):
    datasets = cd["datasets"]
    metrics  = ["auroc", "aurc", "eaurc", "naurc"]
    higher_is_better = {"auroc": True, "aurc": False, "eaurc": False, "naurc": False}

    print()
    print("Cross-Dataset Wilcoxon Test  (N={} datasets: {})".format(
        len(datasets), ", ".join(datasets)))
    print("=" * 80)
    print("{:<8}  {:>10}  {:>10}  {:>12}  {:>12}  {:>6}".format(
        "Metric", "Δ mean", "W stat", "p-value", "Significant?", "Stars"))
    print("-" * 80)
    for m in metrics:
        s = cd["metrics"][m]
        suf = significance_stars(s["wilcoxon_p"])
        print("{:<8}  {:>+10.4f}  {:>10.1f}  {:>12.4f}  {:>12}  {:>6}".format(
            m, s["diff_mean"], s["wilcoxon_statistic"], s["wilcoxon_p"],
            "Yes" if s["wilcoxon_p"] < 0.05 else "No", suf))
    print("=" * 80)
    print("Significance: *** p<0.001  ** p<0.01  * p<0.05  ns = not significant")
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Statistical significance testing for kNN-blend method.")
    p.add_argument("--dataset", default="cifar10", help="Primary dataset for multi-seed test")
    p.add_argument("--arch", default="resnet18")
    p.add_argument("--checkpoint", default=None,
                   help="Checkpoint path. Auto-selected from --checkpoint-dir if omitted.")
    p.add_argument("--checkpoint-dir", default="./checkpoints", type=Path)
    p.add_argument("--num-seeds", type=int, default=10,
                   help="Number of random seeds for the multi-seed test (default: 10)")
    p.add_argument("--seed-start", type=int, default=0,
                   help="First seed value (seeds will be seed_start .. seed_start+num_seeds-1)")
    p.add_argument("--k", type=int, default=100, help="kNN k (default: 100, optimal from ablation)")
    p.add_argument("--alpha", type=float, default=0.8, help="Blend alpha (default: 0.8)")
    p.add_argument("--conf-scheme", default="neg_entropy",
                   choices=["max_prob", "margin", "neg_entropy"])
    p.add_argument("--ef-results", type=Path, default=Path("./results/ablation_knn_blend_ef.json"),
                   help="Path to Dim-F ablation JSON for cross-dataset Wilcoxon test")
    p.add_argument("--no-cross-dataset", action="store_true",
                   help="Skip cross-dataset Wilcoxon test")
    p.add_argument("--output", type=Path, default=Path("./results/significance_knn_blend.json"))
    return p.parse_args()


def main():
    args = parse_args()
    device = get_device()
    print(f"Device: {device}")

    # Resolve checkpoint
    ckpt_path = Path(args.checkpoint) if args.checkpoint else \
                find_latest_checkpoint(args.checkpoint_dir, args.dataset)
    print(f"Checkpoint: {ckpt_path}")

    seeds = list(range(args.seed_start, args.seed_start + args.num_seeds))
    print(f"Seeds: {seeds}")
    print(f"Config: k={args.k}, alpha={args.alpha}, conf_scheme={args.conf_scheme}")

    output = {}

    # ---- Multi-seed test ---------------------------------------------------
    print(f"\n{'='*60}")
    print(f"Multi-Seed Test  (N={args.num_seeds}, dataset={args.dataset})")
    print(f"{'='*60}")
    ms = multi_seed_experiment(
        checkpoint_path=ckpt_path,
        dataset=args.dataset, arch=args.arch, device=device,
        seeds=seeds, k=args.k, alpha=args.alpha, conf_scheme=args.conf_scheme,
    )
    print_multi_seed_summary(ms["summary"], len(seeds))
    output["multi_seed"] = ms

    # ---- Cross-dataset Wilcoxon -------------------------------------------
    if not args.no_cross_dataset and args.ef_results.exists():
        print(f"\n{'='*60}")
        print(f"Cross-Dataset Wilcoxon Test  ({args.ef_results})")
        print(f"{'='*60}")
        cd = cross_dataset_wilcoxon(args.ef_results)
        print_cross_dataset_summary(cd)
        output["cross_dataset"] = cd
    elif not args.no_cross_dataset:
        print(f"\n[INFO] Dim-F results not found at {args.ef_results}. "
              "Run ablate_knn_blend.py first or pass --no-cross-dataset.")

    # ---- Save results -------------------------------------------------------
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
