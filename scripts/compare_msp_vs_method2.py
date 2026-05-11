"""Compare MSP vs Method 2 (Feature-kNN blending) using split test evaluation.

Metrics:
  - AURC: Area Under Risk-Coverage curve
  - AUROC: Area Under ROC curve (between correct/incorrect predictions)
"""
import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
from sklearn.preprocessing import StandardScaler

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from data import get_dataloaders, MEDMNIST_DATASETS
from evaluation import SelectionMetrics
from models import get_model


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def extract_logits_and_features(model: torch.nn.Module, inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Run forward pass and expose penultimate features."""
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


def collect_outputs(model: torch.nn.Module, loader, device: torch.device):
    """Collect logits, features, and targets from a dataloader."""
    model.eval()
    all_logits = []
    all_features = []
    all_targets = []
    all_inputs = []

    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device)
            logits, features = extract_logits_and_features(model, inputs)
            all_logits.append(logits.cpu())
            all_features.append(features.cpu())
            all_targets.append(targets.cpu())
            all_inputs.append(inputs.cpu())

    logits = torch.cat(all_logits, dim=0)
    features = torch.cat(all_features, dim=0)
    targets = torch.cat(all_targets, dim=0)
    inputs_all = torch.cat(all_inputs, dim=0)

    probs = F.softmax(logits, dim=1).numpy()
    preds = np.argmax(probs, axis=1)
    confs = np.max(probs, axis=1)

    return {
        "logits": logits,
        "features": features.numpy(),
        "targets": targets.numpy(),
        "inputs": inputs_all.numpy(),
        "probs": probs,
        "preds": preds,
        "confs": confs,
    }


def compute_aurc(confs: np.ndarray, correctness: np.ndarray) -> float:
    """Compute Area Under Risk-Coverage curve."""
    rc_metrics = SelectionMetrics.compute_all_metrics(confs, correctness)
    return float(rc_metrics["aurc"])


def compute_auroc(confs: np.ndarray, correctness: np.ndarray) -> float:
    """
    Compute AUROC: ability of confidence to separate correct from incorrect.
    
    correctness: 1 = correct, 0 = incorrect
    confs: confidence scores
    
    Returns AUC score (0-1, where 1 is perfect separation).
    """
    if len(np.unique(correctness)) < 2:
        # All samples are either correct or incorrect
        return np.nan
    
    try:
        auroc = roc_auc_score(correctness, confs)
        return float(auroc)
    except Exception as e:
        print(f"Warning: Could not compute AUROC: {e}")
        return np.nan


def full_probs_from_knn(knn: KNeighborsClassifier, features: np.ndarray, num_classes: int) -> np.ndarray:
    """Map KNN predict_proba output to full class dimension."""
    probs_partial = knn.predict_proba(features)
    probs_full = np.zeros((features.shape[0], num_classes), dtype=np.float32)
    probs_full[:, knn.classes_.astype(int)] = probs_partial
    return probs_full


def full_probs_from_predict_proba(model, features: np.ndarray, num_classes: int) -> np.ndarray:
    """Map predict_proba output to full class dimension."""
    probs_partial = model.predict_proba(features)
    probs_full = np.zeros((features.shape[0], num_classes), dtype=np.float32)
    probs_full[:, model.classes_.astype(int)] = probs_partial
    return probs_full


def _stable_softmax(scores: np.ndarray) -> np.ndarray:
    shift = np.max(scores, axis=1, keepdims=True)
    exps = np.exp(scores - shift)
    return exps / np.sum(exps, axis=1, keepdims=True)


def _normalize_probs(probs: np.ndarray) -> np.ndarray:
    probs = np.clip(probs, 1e-12, None)
    probs = probs / np.sum(probs, axis=1, keepdims=True)
    return probs.astype(np.float32)


def method_msp(test_probs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """MSP baseline: maximum softmax probability."""
    preds = np.argmax(test_probs, axis=1)
    confs = np.max(test_probs, axis=1)
    return preds, confs


def fit_feature_classifier(classifier_name: str, train_features: np.ndarray, train_targets: np.ndarray):
    """Fit a feature classifier on standardized features."""
    name = classifier_name.lower().strip()
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_features)

    if name == "knn":
        model = KNeighborsClassifier(n_neighbors=5, weights="distance", n_jobs=-1)
        model.fit(train_scaled, train_targets)
        params = {
            "classifier": "knn",
            "n_neighbors": 5,
            "weights": "distance",
            "n_jobs": -1,
        }
    elif name == "logreg":
        model = LogisticRegression(max_iter=2000)
        model.fit(train_scaled, train_targets)
        params = {"classifier": "logreg", "C": 1.0, "max_iter": 2000}
    elif name == "gnb":
        model = GaussianNB()
        model.fit(train_scaled, train_targets)
        params = {"classifier": "gnb"}
    elif name == "centroid":
        model = NearestCentroid()
        model.fit(train_scaled, train_targets)
        params = {"classifier": "centroid"}
    else:
        raise ValueError(f"Unknown method-2 classifier: {classifier_name}")

    return {"name": name, "scaler": scaler, "model": model, "params": params}


def predict_feature_classifier_probs(fitted, query_features: np.ndarray, num_classes: int) -> np.ndarray:
    """Predict class probabilities for a fitted feature classifier."""
    query_scaled = fitted["scaler"].transform(query_features)
    model = fitted["model"]
    name = fitted["name"]

    if name in {"knn", "logreg", "gnb"}:
        probs = full_probs_from_predict_proba(model, query_scaled, num_classes)
    elif name == "centroid":
        centroids = model.centroids_
        distances = np.linalg.norm(query_scaled[:, None, :] - centroids[None, :, :], axis=2)
        probs = _stable_softmax(-distances)
    else:
        raise ValueError(f"Unknown fitted classifier: {name}")

    return _normalize_probs(probs)


def method_2_feature_classifier_blend_split(
    train_features,
    train_targets,
    calib_features,
    calib_base_probs,
    calib_targets,
    eval_features,
    eval_base_probs,
    num_classes,
    classifier_name: str,
):
    """Method 2: feature-classifier probability blending with alpha tuned on calibration."""
    fitted = fit_feature_classifier(classifier_name, train_features, train_targets)
    calib_clf_probs = predict_feature_classifier_probs(fitted, calib_features, num_classes)

    candidate_alpha = [0.1, 0.25, 0.4, 0.55, 0.7, 0.85]
    best = None

    for alpha in candidate_alpha:
        probs = (1.0 - alpha) * calib_base_probs + alpha * calib_clf_probs
        probs = _normalize_probs(probs)
        preds = np.argmax(probs, axis=1)
        confs = np.max(probs, axis=1)
        correctness = (preds == calib_targets).astype(int)
        aurc = compute_aurc(confs, correctness)
        if best is None or aurc < best["aurc"]:
            best = {"aurc": aurc, "alpha": alpha}

    eval_clf_probs = predict_feature_classifier_probs(fitted, eval_features, num_classes)
    eval_probs_blend = (1.0 - best["alpha"]) * eval_base_probs + best["alpha"] * eval_clf_probs
    eval_probs_blend = _normalize_probs(eval_probs_blend)

    preds = np.argmax(eval_probs_blend, axis=1)
    confs = np.max(eval_probs_blend, axis=1)

    params = dict(fitted["params"])
    params["alpha"] = float(best["alpha"])

    return preds, confs, params


def eval_metrics(preds: np.ndarray, confs: np.ndarray, targets: np.ndarray) -> Dict[str, float]:
    """Compute accuracy, AURC, and AUROC."""
    correctness = (preds == targets).astype(int)
    acc = float(np.mean(correctness))
    aurc = compute_aurc(confs, correctness)
    auroc = compute_auroc(confs, correctness)
    
    return {
        "accuracy": acc,
        "aurc": aurc,
        "auroc": auroc,
    }


def main():
    parser = argparse.ArgumentParser(description="Compare MSP vs Method 2 on split test set")
    
    dataset_choices = ["cifar10", "cifar100"] + list(MEDMNIST_DATASETS.keys()) + ["tinyimagenet"]
    
    parser.add_argument("--dataset", type=str, default="cifar10", choices=dataset_choices)
    parser.add_argument("--arch", type=str, default="resnet18", choices=["resnet18", "resnet34", "resnet50", "resnet101"])
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to converted checkpoint")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--method2-classifiers",
        type=str,
        default="knn,logreg,gnb,centroid",
        help="Comma-separated classifier choices to compare inside method 2.",
    )
    parser.add_argument("--output", type=str, default="./results/msp_vs_method2.json")
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = get_device()
    print(f"Using device: {device}")

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        print(f"Error: Checkpoint not found: {checkpoint_path}")
        return
    
    print(f"Loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device)
    cfg = ckpt.get("config", {})

    # Determine architecture and dataset
    arch = cfg.get("model", {}).get("arch", args.arch)
    # Prefer checkpoint metadata, but if unknown, use command-line argument
    ckpt_dataset = cfg.get("data", {}).get("dataset", "").lower()
    dataset = ckpt_dataset if ckpt_dataset and ckpt_dataset != "unknown" else args.dataset.lower()
    
    # Auto-determine num_classes
    if dataset == "cifar10":
        num_classes = 10
    elif dataset == "cifar100":
        num_classes = 100
    elif dataset in MEDMNIST_DATASETS:
        num_classes = MEDMNIST_DATASETS[dataset]["num_classes"]
    elif dataset == "tinyimagenet":
        num_classes = 200
    else:
        num_classes = 10
        print(f"Warning: Unknown dataset '{dataset}', defaulting to 10 classes")

    print(f"Config: dataset={dataset}, arch={arch}, num_classes={num_classes}")

    # Load data
    train_loader, val_loader, test_loader, _ = get_dataloaders(
        dataset=dataset,
        data_dir="./data",
        batch_size=args.batch_size,
        val_batch_size=args.batch_size,
        num_workers=4,
        augmentation=False,
        seed=args.seed,
    )

    # Load model
    input_size = 64 if dataset == "tinyimagenet" else 32
    model = get_model(arch, num_classes=num_classes, input_size=input_size).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    # Collect outputs
    print("\nCollecting validation outputs...")
    val_data = collect_outputs(model, val_loader, device)
    
    print("Collecting test outputs...")
    test_data = collect_outputs(model, test_loader, device)

    # Split test set into calibration and evaluation
    test_size = test_data["targets"].shape[0]
    split_idx = test_size // 2
    
    calib_data = {
        "features": test_data["features"][:split_idx],
        "probs": test_data["probs"][:split_idx],
        "targets": test_data["targets"][:split_idx],
    }
    
    eval_data = {
        "features": test_data["features"][split_idx:],
        "probs": test_data["probs"][split_idx:],
        "targets": test_data["targets"][split_idx:],
    }

    print(f"\nVal set: {val_data['targets'].shape[0]} samples")
    print(f"Calib set: {calib_data['targets'].shape[0]} samples (for tuning)")
    print(f"Eval set: {eval_data['targets'].shape[0]} samples (for evaluation)")

    classifier_choices = [c.strip().lower() for c in args.method2_classifiers.split(",") if c.strip()]
    if not classifier_choices:
        raise ValueError("No method-2 classifiers specified.")

    # Method 1: MSP baseline
    print("\n" + "="*70)
    print("Method 1: MSP (Maximum Softmax Probability)")
    print("="*70)
    msp_preds, msp_confs = method_msp(eval_data["probs"])
    msp_metrics = eval_metrics(msp_preds, msp_confs, eval_data["targets"])
    print(f"Accuracy: {msp_metrics['accuracy']:.4f}")
    print(f"AURC:     {msp_metrics['aurc']:.4f}")
    print(f"AUROC:    {msp_metrics['auroc']:.4f}")

    method2_results = []
    for classifier_name in classifier_choices:
        print("\n" + "="*70)
        print(f"Method 2: Feature-classifier blending ({classifier_name})")
        print("="*70)
        m2_preds, m2_confs, m2_params = method_2_feature_classifier_blend_split(
            val_data["features"],
            val_data["targets"],
            calib_data["features"],
            calib_data["probs"],
            calib_data["targets"],
            eval_data["features"],
            eval_data["probs"],
            num_classes,
            classifier_name=classifier_name,
        )
        m2_metrics = eval_metrics(m2_preds, m2_confs, eval_data["targets"])
        print(f"Optimal params: {m2_params}")
        print(f"Accuracy: {m2_metrics['accuracy']:.4f}")
        print(f"AURC:     {m2_metrics['aurc']:.4f}")
        print(f"AUROC:    {m2_metrics['auroc']:.4f}")
        method2_results.append(
            {
                "method": f"Method 2 ({classifier_name})",
                "metrics": m2_metrics,
                "params": m2_params,
            }
        )

    # Summary comparison
    print("\n" + "="*70)
    print("COMPARISON SUMMARY")
    print("="*70)
    print(f"{'Method':28s} {'Acc':>10s} {'AURC':>10s} {'AUROC':>10s}")
    print("-" * 64)
    print(f"{'MSP':28s} {msp_metrics['accuracy']:10.4f} {msp_metrics['aurc']:10.4f} {msp_metrics['auroc']:10.4f}")
    for row in method2_results:
        met = row["metrics"]
        print(f"{row['method'][:28]:28s} {met['accuracy']:10.4f} {met['aurc']:10.4f} {met['auroc']:10.4f}")

    best_by_aurc = min(method2_results, key=lambda r: r["metrics"]["aurc"])
    best_by_auroc = max(method2_results, key=lambda r: r["metrics"]["auroc"])

    print("\nBest method-2 variants:")
    print(f"  By AURC : {best_by_aurc['method']} (AURC={best_by_aurc['metrics']['aurc']:.4f})")
    print(f"  By AUROC: {best_by_auroc['method']} (AUROC={best_by_auroc['metrics']['auroc']:.4f})")

    # Save results
    results = {
        "checkpoint": str(checkpoint_path),
        "dataset": dataset,
        "arch": arch,
        "num_classes": num_classes,
        "val_size": int(val_data["targets"].shape[0]),
        "calib_size": int(calib_data["targets"].shape[0]),
        "eval_size": int(eval_data["targets"].shape[0]),
        "method2_classifiers": classifier_choices,
        "methods": {
            "MSP": {
                "metrics": msp_metrics,
                "params": {},
            },
            **{row["method"]: {"metrics": row["metrics"], "params": row["params"]} for row in method2_results},
        },
        "best_by_aurc": best_by_aurc["method"],
        "best_by_auroc": best_by_auroc["method"],
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nResults saved to: {out_path}")


if __name__ == "__main__":
    main()
