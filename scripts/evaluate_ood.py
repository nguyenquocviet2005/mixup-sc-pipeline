"""OOD (Out-of-Distribution) Detection evaluation adapted from the ODIN pipeline.

Evaluates multiple OOD scoring methods on Mixup-trained ResNet checkpoints.
For each ID dataset (CIFAR-10 / CIFAR-100), OOD datasets are:
  - Gaussian noise
  - Uniform noise
  - SVHN (downloaded automatically)
  - Textures (DTD, downloaded automatically)

Metrics (following ODIN paper):
  - FPR at TPR 95%
  - Detection Error
  - AUROC
  - AUPR-In
  - AUPR-Out

Scoring methods evaluated:
  - MSP (Maximum Softmax Probability) – baseline
  - ODIN (temperature + input perturbation)
  - Energy Score
  - Mahalanobis Distance (feature-space)
  - kNN-Feature Blending (our Method 2)
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
from sklearn.neighbors import KNeighborsClassifier
import faiss
import scipy.stats as sstats

sys.path.insert(0, str(Path(__file__).parent.parent))

from data import get_dataloaders
from models import get_model


# ---------------------------------------------------------------------------
# Feature extraction (reused from evaluate_posthoc_methods)
# ---------------------------------------------------------------------------

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


def collect_outputs(model, loader, device, max_samples=None):
    """Collect logits, features, probs, etc. from a data loader."""
    model.eval()
    all_logits, all_features, all_targets = [], [], []

    n = 0
    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device)
            logits, features = extract_logits_and_features(model, inputs)
            all_logits.append(logits.cpu())
            all_features.append(features.cpu())
            all_targets.append(targets.cpu())
            n += inputs.size(0)
            if max_samples and n >= max_samples:
                break

    logits = torch.cat(all_logits, 0)
    features = torch.cat(all_features, 0).numpy()
    targets = torch.cat(all_targets, 0).numpy().flatten()
    probs = F.softmax(logits, dim=1).numpy()
    return {
        "logits": logits,
        "features": features,
        "targets": targets,
        "probs": probs,
        "confs": np.max(probs, axis=1),
        "preds": np.argmax(probs, axis=1),
    }


# ---------------------------------------------------------------------------
# OOD score functions
# ---------------------------------------------------------------------------

def score_msp(probs: np.ndarray) -> np.ndarray:
    """Maximum Softmax Probability (baseline)."""
    return np.max(probs, axis=1)


def score_energy(logits_np: np.ndarray, T: float = 1.0) -> np.ndarray:
    """Energy score: -T * logsumexp(logits/T).  Higher = more ID."""
    scaled = logits_np / T
    shift = np.max(scaled, axis=1, keepdims=True)
    logsumexp = np.log(np.sum(np.exp(scaled - shift), axis=1) + 1e-12) + shift.squeeze(1)
    return T * logsumexp  # higher for ID


def score_odin(model, inputs_np, device, T=1000.0, epsilon=0.0014, batch_size=64):
    """ODIN: temperature scaling + input perturbation. Returns per-sample scores."""
    model.eval()
    all_scores = []
    channel_div = [63.0 / 255.0, 62.1 / 255.0, 66.7 / 255.0]

    for start in range(0, inputs_np.shape[0], batch_size):
        end = min(start + batch_size, inputs_np.shape[0])
        batch = torch.from_numpy(inputs_np[start:end]).float().to(device)
        batch.requires_grad_(True)

        logits, _ = extract_logits_and_features(model, batch)
        pseudo = torch.argmax(logits.detach(), dim=1)
        loss = F.cross_entropy(logits / T, pseudo)
        loss.backward()

        gradient = torch.ge(batch.grad.data, 0).float()
        gradient = (gradient - 0.5) * 2
        if gradient.shape[1] == 3:
            for c in range(3):
                gradient[:, c] = gradient[:, c] / channel_div[c]

        perturbed = torch.add(batch.data, -epsilon, gradient)
        with torch.no_grad():
            logits_p, _ = extract_logits_and_features(model, perturbed)
            probs = F.softmax(logits_p / T, dim=1)
            confs = torch.max(probs, dim=1).values
        all_scores.append(confs.cpu().numpy())

    return np.concatenate(all_scores, 0)


def score_mahalanobis(features, class_means, precision):
    """Mahalanobis distance-based score. Higher = more ID."""
    n = features.shape[0]
    n_classes = class_means.shape[0]
    scores = np.zeros(n, dtype=np.float64)
    for i in range(n):
        min_dist = float("inf")
        for c in range(n_classes):
            diff = features[i] - class_means[c]
            dist = diff @ precision @ diff
            if dist < min_dist:
                min_dist = dist
        scores[i] = -min_dist  # negative distance → higher = more ID
    return scores


def fit_mahalanobis(features, targets, num_classes):
    """Compute per-class means and shared precision (inverse covariance)."""
    dim = features.shape[1]
    class_means = np.zeros((num_classes, dim), dtype=np.float64)
    for c in range(num_classes):
        idx = np.where(targets == c)[0]
        if len(idx) > 0:
            class_means[c] = np.mean(features[idx], axis=0)

    # Shared covariance
    centered = np.zeros_like(features, dtype=np.float64)
    for c in range(num_classes):
        idx = np.where(targets == c)[0]
        if len(idx) > 0:
            centered[idx] = features[idx] - class_means[c]
    cov = (centered.T @ centered) / features.shape[0]
    cov += 1e-5 * np.eye(dim, dtype=np.float64)
    precision = np.linalg.inv(cov)
    return class_means, precision


def score_knn_blend(
    val_features, val_targets, val_probs, test_features, test_probs, num_classes,
    k=10, alpha=0.4,
):
    """kNN-Feature blending confidence (our Method 2). Returns per-sample scores."""
    knn = KNeighborsClassifier(n_neighbors=k, weights="distance")
    knn.fit(val_features, val_targets)

    knn_probs_partial = knn.predict_proba(test_features)
    knn_probs = np.zeros((test_features.shape[0], num_classes), dtype=np.float32)
    knn_probs[:, knn.classes_.astype(int)] = knn_probs_partial

    blended = (1.0 - alpha) * test_probs + alpha * knn_probs
    return np.max(blended, axis=1)


# ---------------------------------------------------------------------------
# New Score Functions (Consistent with evaluate_posthoc_methods)
# ---------------------------------------------------------------------------

def get_final_layer_params(model: torch.nn.Module):
    """Helper to extract weights and bias of the final linear layer."""
    if hasattr(model, 'fc'):
        return model.fc.weight.data, model.fc.bias.data
    elif hasattr(model, 'head'):
        if isinstance(model.head, torch.nn.Linear):
            return model.head.weight.data, model.head.bias.data
        elif hasattr(model.head, 'fc'):
            return model.head.fc.weight.data, model.head.fc.bias.data
    elif hasattr(model, 'classifier') and isinstance(model.classifier[-1], torch.nn.Linear):
        return model.classifier[-1].weight.data, model.classifier[-1].bias.data
    return None, None


def score_knn_ood(val_features, test_features, k=50):
    """KNN-OOD baseline faithful to original repo (Sun et al. ICML 2022)."""
    # Normalization as per repo: x / (||x|| + 1e-10)
    v_norm = val_features / (np.linalg.norm(val_features, ord=2, axis=-1, keepdims=True) + 1e-10)
    t_norm = test_features / (np.linalg.norm(test_features, ord=2, axis=-1, keepdims=True) + 1e-10)
    
    index = faiss.IndexFlatL2(v_norm.shape[1])
    index.add(v_norm.astype(np.float32))
    D, _ = index.search(t_norm.astype(np.float32), k)
    return -D[:, -1]


def score_rl_conf_m(logits: torch.Tensor) -> np.ndarray:
    """RL_conf-M: Logit Margin (Top1 - Top2)."""
    val, _ = torch.topk(logits, 2, dim=1)
    return (val[:, 0] - val[:, 1]).cpu().numpy()


def score_rl_geo_m(logits: torch.Tensor, model: torch.nn.Module) -> np.ndarray:
    """RL_geo-M: Geometric Margin."""
    w, b = get_final_layer_params(model)
    if w is None: return None
    w_aug = torch.cat([w, b.unsqueeze(1)], dim=1)
    w_norm = torch.norm(w_aug, p=2, dim=1)
    # Ensure w_norm is on the same device as logits
    w_norm = w_norm.to(logits.device)
    geo_dist = logits / w_norm.unsqueeze(0)
    gv, _ = torch.topk(geo_dist, 2, dim=1)
    return (gv[:, 0] - gv[:, 1]).cpu().numpy()


def get_vim_sirc_params(val_features, val_logits, model, vim_dim=64):
    """Compute centroids, subspace, and scaling for ViM/SIRC."""
    w, b = get_final_layer_params(model)
    u = -torch.linalg.pinv(w) @ b
    centered_val = val_features - u.unsqueeze(0).cpu().numpy()
    
    cov = (centered_val.T @ centered_val) / centered_val.shape[0]
    eig_vals, eig_vecs = torch.linalg.eigh(torch.from_numpy(cov))
    idx = torch.argsort(eig_vals, descending=True)
    NS = eig_vecs[:, idx][:, vim_dim:].cpu().numpy()
    
    vlogit_val = np.linalg.norm(centered_val @ NS, ord=2, axis=-1)
    max_logit_val = val_logits.max(dim=1).values.cpu().numpy()
    alpha = np.mean(max_logit_val) / np.mean(vlogit_val)
    
    # SIRC params from validation residual uncertainty
    vlogit_val_scaled = vlogit_val * alpha
    a = np.mean(vlogit_val_scaled) - 3 * np.std(vlogit_val_scaled)
    b_param = 1.0 / (np.std(vlogit_val_scaled) + 1e-12)
    
    return {"u": u.cpu().numpy(), "NS": NS, "alpha": alpha, "sirc_a": a, "sirc_b": b_param}


def score_sirc(test_features, test_probs, vim_params):
    """SIRC (MSP + ViM-Res)."""
    centered_test = test_features - vim_params["u"][np.newaxis, :]
    vlogit_test = np.linalg.norm(centered_test @ vim_params["NS"], ord=2, axis=-1) * vim_params["alpha"]
    
    msp = np.max(test_probs, axis=1)
    msp = np.clip(msp, 0.0, 1.0 - 1e-12)
    soft = np.log(1.0 - msp)
    additional = np.logaddexp(0.0, -vim_params["sirc_b"] * (vlogit_test - vim_params["sirc_a"]))
    return -soft - additional


# ---------------------------------------------------------------------------
# OOD data generation / loading
# ---------------------------------------------------------------------------

def generate_gaussian_ood(n, img_shape=(3, 32, 32), mean=None, std=None):
    """Generate Gaussian noise images, optionally normalised like CIFAR."""
    imgs = np.random.randn(n, *img_shape).astype(np.float32)
    if mean is not None and std is not None:
        for c in range(img_shape[0]):
            imgs[:, c] = (imgs[:, c] - mean[c]) / std[c]
    return imgs


def generate_uniform_ood(n, img_shape=(3, 32, 32), mean=None, std=None):
    """Generate Uniform noise images."""
    imgs = np.random.rand(n, *img_shape).astype(np.float32)
    if mean is not None and std is not None:
        for c in range(img_shape[0]):
            imgs[:, c] = (imgs[:, c] - mean[c]) / std[c]
    return imgs


def collect_ood_outputs(model, ood_images_np, device, batch_size=256):
    """Forward pass OOD images (numpy array [N,C,H,W]) through model."""
    model.eval()
    all_logits, all_features = [], []
    with torch.no_grad():
        for start in range(0, ood_images_np.shape[0], batch_size):
            end = min(start + batch_size, ood_images_np.shape[0])
            batch = torch.from_numpy(ood_images_np[start:end]).float().to(device)
            logits, feats = extract_logits_and_features(model, batch)
            all_logits.append(logits.cpu())
            all_features.append(feats.cpu())

    logits = torch.cat(all_logits, 0)
    features = torch.cat(all_features, 0).numpy()
    probs = F.softmax(logits, dim=1).numpy()
    return {
        "logits": logits,
        "features": features,
        "probs": probs,
        "confs": np.max(probs, axis=1),
    }


def load_svhn_ood(data_dir, transform, max_samples=10000):
    """Load SVHN test split as OOD (downloaded automatically by torchvision)."""
    from torchvision import datasets
    from torch.utils.data import DataLoader

    try:
        svhn = datasets.SVHN(root=data_dir, split="test", download=True, transform=transform)
    except Exception as e:
        print(f"  [WARN] SVHN download failed ({e}), skipping.")
        return None
    loader = DataLoader(svhn, batch_size=256, shuffle=False, num_workers=2)
    return loader


def load_textures_ood(data_dir, transform, max_samples=5000):
    """Load DTD (Describable Textures) as OOD."""
    from torchvision import datasets
    from torch.utils.data import DataLoader

    try:
        dtd = datasets.DTD(root=data_dir, split="test", download=True, transform=transform)
    except Exception as e:
        print(f"  [WARN] DTD download failed ({e}), skipping.")
        return None
    loader = DataLoader(dtd, batch_size=256, shuffle=False, num_workers=2)
    return loader


# ---------------------------------------------------------------------------
# OOD metric computation (ODIN-style)
# ---------------------------------------------------------------------------

def compute_ood_metrics(id_scores: np.ndarray, ood_scores: np.ndarray) -> Dict[str, float]:
    """Compute standard OOD detection metrics given ID / OOD confidence scores."""
    labels = np.concatenate([np.ones(len(id_scores)), np.zeros(len(ood_scores))])
    scores = np.concatenate([id_scores, ood_scores])

    # AUROC
    auroc = roc_auc_score(labels, scores)

    # FPR at TPR 95%
    thresholds = np.sort(id_scores)
    fpr95 = 1.0
    for t in thresholds:
        tpr = np.mean(id_scores >= t)
        if 0.9495 <= tpr <= 0.9505:
            fpr95 = np.mean(ood_scores >= t)
            break
    # Fallback: search more carefully
    if fpr95 >= 0.999:
        best_diff = float("inf")
        for t in np.linspace(scores.min(), scores.max(), 100000):
            tpr = np.mean(id_scores >= t)
            if abs(tpr - 0.95) < best_diff:
                best_diff = abs(tpr - 0.95)
                fpr95 = np.mean(ood_scores >= t)

    # Detection error
    det_err = 1.0
    for t in np.linspace(scores.min(), scores.max(), 100000):
        miss = np.mean(id_scores < t)
        fa = np.mean(ood_scores >= t)
        det_err = min(det_err, 0.5 * (miss + fa))

    # AUPR-In (ID is positive)
    precision_in, recall_in, _ = precision_recall_curve(labels, scores)
    aupr_in = auc(recall_in, precision_in)

    # AUPR-Out (OOD is positive)
    labels_out = 1 - labels
    precision_out, recall_out, _ = precision_recall_curve(labels_out, -scores)
    aupr_out = auc(recall_out, precision_out)

    return {
        "fpr95": float(fpr95),
        "detection_error": float(det_err),
        "auroc": float(auroc),
        "aupr_in": float(aupr_in),
        "aupr_out": float(aupr_out),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="OOD Detection evaluation (ODIN-style) for Mixup SC pipeline")
    parser.add_argument("--dataset", type=str, default="cifar10", choices=["cifar10", "cifar100"])
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint. If omitted, auto-detect latest.")
    parser.add_argument("--checkpoint-dir", type=str, default="./checkpoints")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-ood-noise", type=int, default=10000, help="Number of noise OOD samples")
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # --- Load checkpoint ---
    import re
    ckpt_dir = Path(args.checkpoint_dir)
    if args.checkpoint:
        ckpt_path = Path(args.checkpoint)
    else:
        if args.dataset == "cifar10":
            glob_pat = "exp2_mixup_best_auroc_epoch_*.pt"
        else:
            glob_pat = "cifar100_mixup_variant2_best_auroc_epoch_*.pt"
        pattern = re.compile(r".*_epoch_(\d+)\.pt$")
        candidates = []
        for ckpt in ckpt_dir.glob(glob_pat):
            m = pattern.search(ckpt.name)
            if m:
                candidates.append((int(m.group(1)), ckpt))
        if not candidates:
            raise FileNotFoundError(f"No checkpoints found for {args.dataset}")
        candidates.sort(key=lambda x: x[0])
        ckpt_path = candidates[-1][1]

    print(f"Checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)

    num_classes = 10 if args.dataset == "cifar10" else 100
    model = get_model("resnet18", num_classes=num_classes).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    # --- Load ID data ---
    train_loader, val_loader, test_loader, _ = get_dataloaders(
        dataset=args.dataset, data_dir="./data", batch_size=args.batch_size,
        val_batch_size=args.batch_size, num_workers=4, augmentation=False, seed=args.seed,
    )

    print("Collecting ID validation outputs...")
    val_data = collect_outputs(model, val_loader, device)
    print("Collecting ID test outputs...")
    id_data = collect_outputs(model, test_loader, device)

    id_acc = float(np.mean(id_data["preds"] == id_data["targets"]))
    print(f"ID accuracy ({args.dataset}): {id_acc:.4f}")

    # Pre-fit Mahalanobis stats on validation features
    print("Fitting Mahalanobis stats...")
    class_means, precision = fit_mahalanobis(val_data["features"], val_data["targets"], num_classes)

    # CIFAR normalization constants
    if args.dataset == "cifar10":
        norm_mean = (0.4914, 0.4822, 0.4465)
        norm_std = (0.2470, 0.2435, 0.2616)
    else:
        norm_mean = (0.5071, 0.4867, 0.4408)
        norm_std = (0.2675, 0.2565, 0.2761)

    # --- Prepare OOD datasets ---
    from torchvision import transforms
    val_transform = transforms.Compose([
        transforms.Resize(32),
        transforms.CenterCrop(32),
        transforms.ToTensor(),
        transforms.Normalize(norm_mean, norm_std),
    ])

    ood_sources = {}

    # Gaussian noise
    print("Generating Gaussian OOD noise...")
    ood_sources["Gaussian"] = generate_gaussian_ood(args.n_ood_noise, mean=norm_mean, std=norm_std)

    # Uniform noise
    print("Generating Uniform OOD noise...")
    ood_sources["Uniform"] = generate_uniform_ood(args.n_ood_noise, mean=norm_mean, std=norm_std)

    # SVHN
    print("Loading SVHN as OOD...")
    svhn_loader = load_svhn_ood("./data", val_transform)
    if svhn_loader is not None:
        ood_sources["SVHN"] = svhn_loader

    # Textures / DTD
    print("Loading DTD (Textures) as OOD...")
    dtd_loader = load_textures_ood("./data", val_transform)
    if dtd_loader is not None:
        ood_sources["Textures"] = dtd_loader

    # --- Scoring methods ---
    method_names = ["MSP", "ODIN", "Energy", "Mahalanobis", "KNN-OOD", "RL_conf-M", "RL_geo-M", "SIRC", "kNN-Feature (Ours)"]

    # Pre-compute parameters for SIRC/ViM
    print("Computing ViM/SIRC parameters on validation set...")
    vim_params = get_vim_sirc_params(val_data["features"], val_data["logits"], model)

    # Pre-compute ID scores
    id_logits_np = id_data["logits"].numpy()
    id_scores = {
        "MSP": score_msp(id_data["probs"]),
        "Energy": score_energy(id_logits_np),
        "Mahalanobis": score_mahalanobis(id_data["features"], class_means, precision),
        "KNN-OOD": score_knn_ood(val_data["features"], id_data["features"], k=50),
        "RL_conf-M": score_rl_conf_m(id_data["logits"]),
        "RL_geo-M": score_rl_geo_m(id_data["logits"], model),
        "SIRC": score_sirc(id_data["features"], id_data["probs"], vim_params),
        "kNN-Feature (Ours)": score_knn_blend(
            val_data["features"], val_data["targets"], val_data["probs"],
            id_data["features"], id_data["probs"], num_classes,
        ),
    }

    # ODIN needs raw inputs → collect from test_loader
    print("Computing ODIN scores for ID data...")
    id_inputs_list = []
    for inputs, _ in test_loader:
        id_inputs_list.append(inputs.numpy())
    id_inputs_np = np.concatenate(id_inputs_list, 0)
    id_scores["ODIN"] = score_odin(model, id_inputs_np, device)

    # --- Evaluate each OOD dataset ---
    all_results = {}

    for ood_name, ood_src in ood_sources.items():
        print(f"\n{'='*60}")
        print(f"OOD dataset: {ood_name}")
        print(f"{'='*60}")

        # Get OOD outputs
        if isinstance(ood_src, np.ndarray):
            ood_out = collect_ood_outputs(model, ood_src, device, args.batch_size)
            ood_inputs_np = ood_src
        else:
            # It's a DataLoader
            ood_out = collect_outputs(model, ood_src, device, max_samples=10000)
            ood_inputs_list = []
            for inputs, _ in ood_src:
                ood_inputs_list.append(inputs.numpy())
                if sum(x.shape[0] for x in ood_inputs_list) >= 10000:
                    break
            ood_inputs_np = np.concatenate(ood_inputs_list, 0)

        ood_logits_np = ood_out["logits"].numpy()

        ood_scores = {
            "MSP": score_msp(ood_out["probs"]),
            "Energy": score_energy(ood_logits_np),
            "Mahalanobis": score_mahalanobis(ood_out["features"], class_means, precision),
            "KNN-OOD": score_knn_ood(val_data["features"], ood_out["features"], k=50),
            "RL_conf-M": score_rl_conf_m(ood_out["logits"]),
            "RL_geo-M": score_rl_geo_m(ood_out["logits"], model),
            "SIRC": score_sirc(ood_out["features"], ood_out["probs"], vim_params),
            "kNN-Feature (Ours)": score_knn_blend(
                val_data["features"], val_data["targets"], val_data["probs"],
                ood_out["features"], ood_out["probs"], num_classes,
            ),
        }

        print(f"  Computing ODIN scores for {ood_name}...")
        ood_scores["ODIN"] = score_odin(model, ood_inputs_np, device)

        # Compute metrics per method
        results_for_ood = {}
        for method in method_names:
            metrics = compute_ood_metrics(id_scores[method], ood_scores[method])
            results_for_ood[method] = metrics
            print(f"  {method:25s}  FPR95={metrics['fpr95']:.4f}  AUROC={metrics['auroc']:.4f}  AUPR-In={metrics['aupr_in']:.4f}")

        all_results[ood_name] = results_for_ood

    # --- Save results ---
    output_path = args.output or f"./results/ood_evaluation_{args.dataset}.json"
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "dataset": args.dataset,
        "checkpoint": str(ckpt_path),
        "id_accuracy": id_acc,
        "num_classes": num_classes,
        "results": all_results,
    }
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"\nSaved OOD evaluation results to: {out_path}")


if __name__ == "__main__":
    main()
