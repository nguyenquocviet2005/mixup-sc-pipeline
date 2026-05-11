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
                by_method[method] = {"accuracy": [], "aurc": [], "eaurc": []}
            by_method[method]["accuracy"].append(float(row["accuracy"]))
            by_method[method]["aurc"].append(float(row["aurc"]))
            by_method[method]["eaurc"].append(float(row["eaurc"]))

    summary = []
    for method, vals in by_method.items():
        acc_vals = vals["accuracy"]
        aurc_vals = vals["aurc"]
        eaurc_vals = vals["eaurc"]

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
            }
        )

    # Sort with baseline first, then by AURC mean.
    baseline = [r for r in summary if r["method"] == "Baseline Mixup"]
    others = [r for r in summary if r["method"] != "Baseline Mixup"]
    others.sort(key=lambda x: x["aurc_mean"])
    return baseline + others


def print_table(summary):
    print("\nRobustness Summary Across Checkpoints")
    print(
        f"{'Method':48s} {'Acc (mean±std)':>18s} {'AURC (mean±std)':>20s} {'E-AURC (mean±std)':>22s}"
    )
    print("-" * 118)

    for r in summary:
        acc = f"{r['accuracy_mean']:.4f}±{r['accuracy_std']:.4f}"
        aurc = f"{r['aurc_mean']:.4f}±{r['aurc_std']:.4f}"
        eaurc = f"{r['eaurc_mean']:.4f}±{r['eaurc_std']:.4f}"
        print(f"{r['method'][:48]:48s} {acc:>18s} {aurc:>20s} {eaurc:>22s}")


def main():
    parser = argparse.ArgumentParser(description="Robustness runner for post-hoc Mixup SC methods")
    
    # Build dataset choices
    dataset_choices = ["cifar10", "cifar100", "tinyimagenet"] + list(MEDMNIST_DATASETS.keys())
    
    parser.add_argument("--dataset", type=str, default="cifar100", choices=dataset_choices,
                        help="Dataset to evaluate on (default: cifar100)")
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
            and r["aurc_mean"] < baseline["aurc_mean"]
            and r["eaurc_mean"] <= baseline["eaurc_mean"]
        ]
        if improved:
            best = sorted(improved, key=lambda x: x["aurc_mean"])[0]
            conclusion = (
                f"Primary candidate: {best['method']} "
                f"(AURC {best['aurc_mean']:.4f}±{best['aurc_std']:.4f} vs "
                f"baseline {baseline['aurc_mean']:.4f}±{baseline['aurc_std']:.4f})."
            )
        else:
            conclusion = "All methods failed to robustly outperform baseline Mixup on mean AURC/E-AURC."

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
