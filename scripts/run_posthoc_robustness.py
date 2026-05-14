"""Run post-hoc evaluation across multiple Mixup checkpoints and report mean/std."""
import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from statistics import mean, stdev

sys.path.insert(0, str(Path(__file__).parent.parent))
from data import MEDMNIST_DATASETS


def list_mixup_checkpoints(checkpoint_dir: Path, checkpoint_glob: str, epoch_regex: str):
    pattern = re.compile(epoch_regex)
    rows = []
    for ckpt in checkpoint_dir.glob(checkpoint_glob):
        m = pattern.search(ckpt.name)
        if m:
            rows.append((int(m.group(1)), ckpt))
    rows.sort(key=lambda x: x[0])
    return rows


def default_checkpoint_glob(dataset: str) -> str:
    dataset = dataset.lower()
    if dataset == "cifar10":
        return "exp2_mixup_best_auroc_epoch_*.pt"
    if dataset == "cifar100":
        return "cifar100_mixup_variant2_best_auroc_epoch_*.pt"
    if dataset == "tinyimagenet":
        return "mixup_tinyimagenet_resnet18_best_auroc_epoch_*.pt"
    return f"mixup_{dataset}_resnet18_best_auroc_epoch_*.pt"


def aggregate_runs(run_payloads):
    """Aggregate metrics per method across runs."""
    by_method = {}
    for payload in run_payloads:
        for row in payload["results"]:
            method = row["method"]
            if method not in by_method:
                by_method[method] = {"accuracy": [], "aurc": [], "eaurc": [], "naurc": [], "auroc": []}
            by_method[method]["accuracy"].append(float(row["accuracy"]))
            by_method[method]["aurc"].append(float(row["aurc"]))
            by_method[method]["eaurc"].append(float(row["eaurc"]))
            by_method[method]["naurc"].append(float(row["naurc"]))
            by_method[method]["auroc"].append(float(row["auroc"]))

    summary = []
    for method, vals in by_method.items():
        acc_vals = vals["accuracy"]
        aurc_vals = vals["aurc"]
        eaurc_vals = vals["eaurc"]
        naurc_vals = vals["naurc"]
        auroc_vals = vals["auroc"]

        summary.append(
            {
                "method": method,
                "runs": len(acc_vals),
                "accuracy_mean": mean(acc_vals),
                "accuracy_std": stdev(acc_vals) if len(acc_vals) > 1 else 0.0,
                "aurc_mean": mean(aurc_vals),
                "aurc_std": stdev(aurc_vals) if len(aurc_vals) > 1 else 0.0,
                "eaurc_mean": mean(eaurc_vals),
                "eaurc_std": stdev(eaurc_vals) if len(eaurc_vals) > 1 else 0.0,
                "naurc_mean": mean(naurc_vals),
                "naurc_std": stdev(naurc_vals) if len(naurc_vals) > 1 else 0.0,
                "auroc_mean": mean(auroc_vals),
                "auroc_std": stdev(auroc_vals) if len(auroc_vals) > 1 else 0.0,
            }
        )

    # Sort with baseline first, then by NAURC mean.
    baseline = [r for r in summary if r["method"] == "Baseline Mixup"]
    others = [r for r in summary if r["method"] != "Baseline Mixup"]
    others.sort(key=lambda x: x["naurc_mean"])
    return baseline + others


def print_table(summary):
    print("\nRobustness Summary Across Checkpoints")
    print(
        f"{'Method':40s} {'Acc (mean±std)':>18s} {'AUROC (mean±std)':>20s} {'AURC (mean±std)':>20s} {'E-AURC (mean±std)':>20s} {'NAURC (mean±std)':>20s}"
    )
    print("-" * 140)

    for r in summary:
        acc = f"{r['accuracy_mean']:.4f}±{r['accuracy_std']:.4f}"
        auroc = f"{r['auroc_mean']:.4f}±{r['auroc_std']:.4f}"
        aurc = f"{r['aurc_mean']:.4f}±{r['aurc_std']:.4f}"
        eaurc = f"{r['eaurc_mean']:.4f}±{r['eaurc_std']:.4f}"
        naurc = f"{r['naurc_mean']:.4f}±{r['naurc_std']:.4f}"
        print(f"{r['method'][:40]:40s} {acc:>18s} {auroc:>20s} {aurc:>20s} {eaurc:>20s} {naurc:>20s}")


def main():
    parser = argparse.ArgumentParser(description="Robustness runner for post-hoc Mixup SC methods")
    
    # Build dataset choices
    dataset_choices = [
        "cifar10", "cifar100", "tinyimagenet", "skin_cancer_isic", "chest_xray",
        "mri_tumor", "alzheimer", "tuberculosis", "sars_cov_2_ct_scan", "chest_ct_scan"
    ] + list(MEDMNIST_DATASETS.keys())
    
    parser.add_argument("--dataset", type=str, default="cifar100", choices=dataset_choices,
                        help="Dataset to evaluate on (default: cifar100)")
    parser.add_argument("--arch", type=str, default=None,
                        choices=["resnet18", "resnet34", "resnet50", "resnet101", "resnet110", "vgg16_bn", "vit_b_16", "vit_b_4", "wrn28_10", "dense", "cmixer"],
                        help="Model architecture override.")
    parser.add_argument("--checkpoint-dir", type=str, default="./checkpoints")
    parser.add_argument(
        "--checkpoint-glob",
        type=str,
        default=None,
        help="Glob pattern used to select checkpoint files. If omitted, a dataset-specific default is used.",
    )
    parser.add_argument(
        "--epoch-regex",
        type=str,
        default=r".*_epoch_(\d+)\.pt$",
        help="Regex with one capture group for epoch number extracted from filename.",
    )
    parser.add_argument(
        "--max-checkpoints",
        type=int,
        default=5,
        help="Number of latest checkpoints to evaluate (use <=0 for all).",
    )
    parser.add_argument(
        "--start-epoch",
        type=int,
        default=None,
        help="Optional minimum checkpoint epoch to include.",
    )
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default="./results/posthoc_robustness_summary.json")
    parser.add_argument(
        "--per-run-dir",
        type=str,
        default="./results/posthoc_runs",
        help="Directory for per-checkpoint outputs from evaluator.",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    evaluator = root / "evaluate_posthoc_methods.py"

    checkpoint_glob = args.checkpoint_glob or default_checkpoint_glob(args.dataset)
    print(f"Using checkpoint glob: {checkpoint_glob}")

    checkpoint_rows = list_mixup_checkpoints(
        Path(args.checkpoint_dir), checkpoint_glob, args.epoch_regex
    )
    if args.start_epoch is not None:
        checkpoint_rows = [row for row in checkpoint_rows if row[0] >= args.start_epoch]

    if not checkpoint_rows:
        raise FileNotFoundError("No matching mixup checkpoints found for robustness run.")

    if args.max_checkpoints > 0:
        checkpoint_rows = checkpoint_rows[-args.max_checkpoints :]

    print(f"Evaluating {len(checkpoint_rows)} checkpoints...")

    run_payloads = []
    per_run_dir = Path(args.per_run_dir)
    per_run_dir.mkdir(parents=True, exist_ok=True)

    for epoch, ckpt in checkpoint_rows:
        out_file = per_run_dir / f"posthoc_epoch_{epoch}.json"
        cmd = [
            sys.executable,
            str(evaluator),
            "--dataset",
            args.dataset,
            "--checkpoint",
            str(ckpt),
            "--batch-size",
            str(args.batch_size),
            "--seed",
            str(args.seed),
            "--output",
            str(out_file),
        ]
        if args.arch is not None:
            cmd.extend(["--arch", args.arch])

        print(f"\n[Run] epoch={epoch} checkpoint={ckpt.name}")
        subprocess.run(cmd, check=True)

        payload = json.loads(out_file.read_text())
        run_payloads.append(payload)

    summary = aggregate_runs(run_payloads)
    print_table(summary)

    baseline = next((r for r in summary if r["method"] == "Baseline Mixup"), None)
    if baseline is None:
        conclusion = "Baseline row missing; cannot determine primary candidate."
    else:
        improved = [
            r
            for r in summary
            if r["method"] != "Baseline Mixup"
            and r["naurc_mean"] < baseline["naurc_mean"]
        ]
        if improved:
            best = sorted(improved, key=lambda x: x["naurc_mean"])[0]
            conclusion = (
                f"Primary candidate: {best['method']} "
                f"(NAURC {best['naurc_mean']:.4f}±{best['naurc_std']:.4f} vs "
                f"baseline {baseline['naurc_mean']:.4f}±{baseline['naurc_std']:.4f})."
            )
        else:
            conclusion = "All methods failed to robustly outperform baseline Mixup on mean NAURC."

    print("\nConclusion:")
    print(conclusion)

    out_payload = {
        "num_checkpoints": len(checkpoint_rows),
        "checkpoints": [str(ckpt) for _, ckpt in checkpoint_rows],
        "summary": summary,
        "conclusion": conclusion,
        "per_run_dir": str(per_run_dir),
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out_payload, indent=2))
    print(f"\nSaved robustness summary to: {out_path}")


if __name__ == "__main__":
    main()
