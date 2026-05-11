"""Run method-2 classifier comparison across checkpoints and summarize mean/std."""
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


def aggregate_method2_runs(run_payloads):
    """Aggregate metrics per method across runs."""
    by_method = {}
    for payload in run_payloads:
        for method_name, row in payload["methods"].items():
            if method_name == "MSP":
                continue
            if method_name not in by_method:
                by_method[method_name] = {
                    "accuracy": [],
                    "aurc": [],
                    "auroc": [],
                }
            metrics = row["metrics"]
            by_method[method_name]["accuracy"].append(float(metrics["accuracy"]))
            by_method[method_name]["aurc"].append(float(metrics["aurc"]))
            by_method[method_name]["auroc"].append(float(metrics["auroc"]))

    summary = []
    for method, vals in by_method.items():
        acc_vals = vals["accuracy"]
        aurc_vals = vals["aurc"]
        auroc_vals = vals["auroc"]
        summary.append(
            {
                "method": method,
                "runs": len(acc_vals),
                "accuracy_mean": mean(acc_vals),
                "accuracy_std": stdev(acc_vals) if len(acc_vals) > 1 else 0.0,
                "aurc_mean": mean(aurc_vals),
                "aurc_std": stdev(aurc_vals) if len(aurc_vals) > 1 else 0.0,
                "auroc_mean": mean(auroc_vals),
                "auroc_std": stdev(auroc_vals) if len(auroc_vals) > 1 else 0.0,
            }
        )

    summary.sort(key=lambda x: x["aurc_mean"])
    return summary


def aggregate_msp_runs(run_payloads):
    """Aggregate MSP baseline across runs for reference."""
    acc_vals = []
    aurc_vals = []
    auroc_vals = []

    for payload in run_payloads:
        msp = payload["methods"]["MSP"]["metrics"]
        acc_vals.append(float(msp["accuracy"]))
        aurc_vals.append(float(msp["aurc"]))
        auroc_vals.append(float(msp["auroc"]))

    return {
        "method": "MSP",
        "runs": len(acc_vals),
        "accuracy_mean": mean(acc_vals),
        "accuracy_std": stdev(acc_vals) if len(acc_vals) > 1 else 0.0,
        "aurc_mean": mean(aurc_vals),
        "aurc_std": stdev(aurc_vals) if len(aurc_vals) > 1 else 0.0,
        "auroc_mean": mean(auroc_vals),
        "auroc_std": stdev(auroc_vals) if len(auroc_vals) > 1 else 0.0,
    }


def format_summary_table(dataset: str, msp_row: dict, summary: list[dict]) -> str:
    lines = []
    lines.append(f"## {dataset}")
    lines.append("")
    lines.append("| Method | Acc (mean±std) | AURC (mean±std) | AUROC (mean±std) | Runs |")
    lines.append("|---|---|---|---|---|")

    def fmt(row):
        return (
            f"| {row['method']} | "
            f"{row['accuracy_mean']:.4f}±{row['accuracy_std']:.4f} | "
            f"{row['aurc_mean']:.4f}±{row['aurc_std']:.4f} | "
            f"{row['auroc_mean']:.4f}±{row['auroc_std']:.4f} | "
            f"{row['runs']} |"
        )

    lines.append(fmt(msp_row))
    for row in summary:
        lines.append(fmt(row))

    best_aurc = min(summary, key=lambda x: x["aurc_mean"])
    best_auroc = max(summary, key=lambda x: x["auroc_mean"])

    lines.append("")
    lines.append(
        f"Best by AURC: {best_aurc['method']} ({best_aurc['aurc_mean']:.4f}±{best_aurc['aurc_std']:.4f})"
    )
    lines.append(
        f"Best by AUROC: {best_auroc['method']} ({best_auroc['auroc_mean']:.4f}±{best_auroc['auroc_std']:.4f})"
    )
    lines.append("")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Run method-2 classifier robustness across checkpoints"
    )

    dataset_choices = ["cifar10", "cifar100", "tinyimagenet"] + list(
        MEDMNIST_DATASETS.keys()
    )

    parser.add_argument(
        "--datasets",
        type=str,
        default="cifar10,cifar100,tinyimagenet",
        help="Comma-separated datasets to evaluate.",
    )
    parser.add_argument("--checkpoint-dir", type=str, default="./checkpoints")
    parser.add_argument(
        "--checkpoint-glob",
        type=str,
        default=None,
        help="Glob pattern for checkpoints (overrides dataset default).",
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
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--method2-classifiers",
        type=str,
        default="knn,logreg,gnb,centroid",
        help="Comma-separated classifier choices to compare inside method 2.",
    )
    parser.add_argument(
        "--per-run-dir",
        type=str,
        default="./results/method2_classifier_runs",
        help="Directory for per-checkpoint outputs.",
    )
    parser.add_argument(
        "--report",
        type=str,
        default="./results/method2_classifier_robustness_report.md",
        help="Path to markdown report.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Recompute outputs even if per-checkpoint JSON already exists.",
    )
    args = parser.parse_args()

    datasets = [d.strip().lower() for d in args.datasets.split(",") if d.strip()]
    for d in datasets:
        if d not in dataset_choices:
            raise ValueError(f"Unknown dataset: {d}")

    root = Path(__file__).resolve().parent
    evaluator = root / "compare_msp_vs_method2.py"

    per_run_dir = Path(args.per_run_dir)
    per_run_dir.mkdir(parents=True, exist_ok=True)

    report_lines = []
    report_lines.append("# Method 2 Classifier Robustness Report")
    report_lines.append("")
    report_lines.append("Metrics: AURC and AUROC (correct vs incorrect predictions).")
    report_lines.append(
        "Evaluation: for each checkpoint, split test set 50/50 into calibration and eval; tune alpha on calibration."
    )
    report_lines.append("")

    for dataset in datasets:
        checkpoint_glob = args.checkpoint_glob or default_checkpoint_glob(dataset)
        checkpoint_rows = list_mixup_checkpoints(
            Path(args.checkpoint_dir), checkpoint_glob, args.epoch_regex
        )
        if not checkpoint_rows:
            raise FileNotFoundError(
                f"No matching checkpoints found for dataset {dataset} with glob {checkpoint_glob}"
            )
        if args.max_checkpoints > 0:
            checkpoint_rows = checkpoint_rows[-args.max_checkpoints :]

        run_payloads = []
        dataset_run_dir = per_run_dir / dataset
        dataset_run_dir.mkdir(parents=True, exist_ok=True)

        for epoch, ckpt in checkpoint_rows:
            out_file = dataset_run_dir / f"method2_epoch_{epoch}.json"
            if out_file.exists() and not args.force:
                payload = json.loads(out_file.read_text())
                run_payloads.append(payload)
                continue
            cmd = [
                sys.executable,
                str(evaluator),
                "--dataset",
                dataset,
                "--checkpoint",
                str(ckpt),
                "--batch-size",
                str(args.batch_size),
                "--seed",
                str(args.seed),
                "--method2-classifiers",
                args.method2_classifiers,
                "--output",
                str(out_file),
            ]
            print(f"\n[Run] dataset={dataset} epoch={epoch} checkpoint={ckpt.name}")
            subprocess.run(
                cmd,
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.STDOUT,
            )

            payload = json.loads(out_file.read_text())
            run_payloads.append(payload)

        msp_row = aggregate_msp_runs(run_payloads)
        summary = aggregate_method2_runs(run_payloads)
        report_lines.append(format_summary_table(dataset, msp_row, summary))

    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(report_lines))
    print(f"\nSaved report to: {report_path}")


if __name__ == "__main__":
    main()
