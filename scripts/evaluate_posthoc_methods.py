"""Evaluate exactly 3 post-hoc methods on the best Mixup checkpoint for SC metrics."""
import argparse
import json
import math
import re
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.neighbors import KNeighborsClassifier
from sklearn.covariance import EmpiricalCovariance
import faiss
import scipy.stats as sstats
from scipy.special import logsumexp

import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from data import get_dataloaders, MEDMNIST_DATASETS
from evaluation import SelectionMetrics
from models import get_model


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def find_best_mixup_checkpoint(checkpoint_dir: Path, dataset: str) -> Path:
    """Select latest best-auroc checkpoint for a specific dataset."""
    dataset = dataset.lower()
    if dataset == "cifar10":
        glob_pat = "exp2_mixup_best_auroc_epoch_*.pt"
    elif dataset == "cifar100":
        # Prefer explicitly labeled CIFAR-100 checkpoints when available.
        glob_pat = "cifar100_mixup_variant2_best_auroc_epoch_*.pt"
    else:
        glob_pat = f"mixup_{dataset}_resnet18_best_auroc_epoch_*.pt"

    pattern = re.compile(r".*_epoch_(\d+)\.pt$")
    candidates = []
    for ckpt in checkpoint_dir.glob(glob_pat):
        m = pattern.search(ckpt.name)
        if m:
            candidates.append((int(m.group(1)), ckpt))

    if not candidates:
        raise FileNotFoundError(
            f"No checkpoints found for dataset '{dataset}' with pattern '{glob_pat}' in {checkpoint_dir}"
        )

    candidates.sort(key=lambda x: x[0])
    return candidates[-1][1]


def extract_logits_and_features(model: torch.nn.Module, inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
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


def collect_outputs(model: torch.nn.Module, loader, device: torch.device):
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


def evaluate_sc_metrics(preds: np.ndarray, confs: np.ndarray, targets: np.ndarray) -> Dict[str, float]:
    if not np.all(np.isfinite(confs)):
        print(f"  [DEBUG] Non-finite confidence scores detected! Samples with non-finite values: {np.sum(~np.isfinite(confs))}")
        confs = np.nan_to_num(confs, nan=0.0, posinf=1e6, neginf=-1e10)
    
    correctness = (preds == targets).astype(int)
    sc = SelectionMetrics.compute_all_metrics(confs, correctness)
    acc = float(np.mean(correctness))
    return {
        "accuracy": acc,
        "aurc": float(sc["aurc"]),
        "eaurc": float(sc["eaurc"]),
    }


def full_probs_from_knn(knn: KNeighborsClassifier, features: np.ndarray, num_classes: int) -> np.ndarray:
    """Map KNN predict_proba output to full class dimension."""
    probs_partial = knn.predict_proba(features)
    probs_full = np.zeros((features.shape[0], num_classes), dtype=np.float32)
    probs_full[:, knn.classes_.astype(int)] = probs_partial
    return probs_full


def method_1_classwise_temperature_scaling(val_logits, val_targets, test_logits):
    """Method 1: class-wise temperature scaling fitted on validation NLL."""
    num_classes = val_logits.shape[1]
    device = val_logits.device

    log_t = torch.zeros(num_classes, device=device, requires_grad=True)
    optimizer = torch.optim.Adam([log_t], lr=0.05)

    best_state = None
    best_loss = float("inf")

    for _ in range(400):
        optimizer.zero_grad()
        temperatures = torch.exp(log_t).clamp(0.05, 10.0)
        scaled = val_logits / temperatures.unsqueeze(0)
        loss = F.cross_entropy(scaled, val_targets) + 1e-4 * torch.mean(log_t ** 2)
        loss.backward()
        optimizer.step()

        cur = float(loss.item())
        if cur < best_loss:
            best_loss = cur
            best_state = log_t.detach().clone()

    with torch.no_grad():
        temperatures = torch.exp(best_state).clamp(0.05, 10.0)
        test_scaled = test_logits / temperatures.unsqueeze(0)
        test_probs = F.softmax(test_scaled, dim=1).cpu().numpy()

    return {
        "name": "Method 1: Class-wise Temperature Scaling",
        "params": {
            "temperatures": temperatures.detach().cpu().numpy().round(4).tolist(),
        },
        "test_probs": test_probs,
    }


def method_2_feature_knn_logit_blend(
    val_features,
    val_probs,
    val_targets,
    test_features,
    test_probs,
    num_classes,
):
    """Method 2: Feature-space kNN probability blending."""
    n = val_features.shape[0]
    split = n // 2

    ref_features = val_features[:split]
    ref_targets = val_targets[:split]
    calib_features = val_features[split:]
    calib_targets = val_targets[split:]
    calib_base_probs = val_probs[split:]

    candidate_k = [5, 10, 20, 30]
    candidate_alpha = [0.1, 0.25, 0.4, 0.55, 0.7, 0.85]

    best = None

    for k in candidate_k:
        knn = KNeighborsClassifier(n_neighbors=k, weights="distance")
        knn.fit(ref_features, ref_targets)
        calib_knn_probs = full_probs_from_knn(knn, calib_features, num_classes)

        for alpha in candidate_alpha:
            probs = (1.0 - alpha) * calib_base_probs + alpha * calib_knn_probs
            # Ensure probabilities sum to 1 (in case kNN has missing classes)
            # probs = probs / np.sum(probs, axis=1, keepdims=True)
            preds = np.argmax(probs, axis=1)
            confs = np.max(probs, axis=1)
            metrics = evaluate_sc_metrics(preds, confs, calib_targets)
            score = metrics["aurc"]
            if best is None or score < best["score"]:
                best = {
                    "score": score,
                    "k": k,
                    "alpha": alpha,
                }

    # Refit on full validation set with selected params.
    knn = KNeighborsClassifier(n_neighbors=best["k"], weights="distance")
    knn.fit(val_features, val_targets)
    test_knn_probs = full_probs_from_knn(knn, test_features, num_classes)
    test_probs_blend = (1.0 - best["alpha"]) * test_probs + best["alpha"] * test_knn_probs
    
    # Ensure probabilities sum to 1 (in case kNN has missing classes)
    test_probs_blend = test_probs_blend / np.sum(test_probs_blend, axis=1, keepdims=True)

    return {
        "name": "Method 2: Feature-kNN / Logit Probability Blending",
        "params": {"k": int(best["k"]), "alpha": float(best["alpha"])},
        "test_probs": test_probs_blend,
    }


def _prototype_stats(features: np.ndarray, targets: np.ndarray, num_classes: int):
    dim = features.shape[1]
    centroids = np.zeros((num_classes, dim), dtype=np.float32)
    class_scores = {}

    for c in range(num_classes):
        idx = np.where(targets == c)[0]
        if len(idx) == 0:
            class_scores[c] = np.array([1e6], dtype=np.float32)
            continue
        class_feats = features[idx]
        centroids[c] = np.mean(class_feats, axis=0)

    # Distances to true class centroid as calibration nonconformity scores.
    for c in range(num_classes):
        idx = np.where(targets == c)[0]
        if len(idx) == 0:
            continue
        d = np.linalg.norm(features[idx] - centroids[c], axis=1)
        class_scores[c] = np.sort(d)

    return centroids, class_scores


def _conformal_pvalues(
    features: np.ndarray,
    pred_classes: np.ndarray,
    centroids: np.ndarray,
    class_scores: Dict[int, np.ndarray],
):
    pvals = np.zeros(features.shape[0], dtype=np.float32)
    for i in range(features.shape[0]):
        c = int(pred_classes[i])
        dist = np.linalg.norm(features[i] - centroids[c])
        scores = class_scores.get(c, np.array([1e6], dtype=np.float32))
        # p = P(S >= s_test) with smoothing.
        count = np.sum(scores >= dist)
        pvals[i] = (count + 1.0) / (len(scores) + 1.0)
    return pvals


def method_3_prototype_conformal_confidence(
    val_features,
    val_probs,
    val_targets,
    test_features,
    test_probs,
    num_classes,
):
    """Method 3: Prototype-distance conformal confidence on mixed representations."""
    n = val_features.shape[0]
    split = n // 2

    proto_features = val_features[:split]
    proto_targets = val_targets[:split]
    calib_features = val_features[split:]
    calib_targets = val_targets[split:]
    calib_probs = val_probs[split:]

    centroids, class_scores = _prototype_stats(proto_features, proto_targets, num_classes)

    calib_preds = np.argmax(calib_probs, axis=1)
    calib_softmax_conf = np.max(calib_probs, axis=1)
    calib_pvals = _conformal_pvalues(calib_features, calib_preds, centroids, class_scores)

    best = None
    for beta in [0.0, 0.25, 0.5, 0.75, 1.0]:
        conf = (calib_softmax_conf ** (1.0 - beta)) * (calib_pvals ** beta)
        metrics = evaluate_sc_metrics(calib_preds, conf, calib_targets)
        if best is None or metrics["aurc"] < best["score"]:
            best = {"score": metrics["aurc"], "beta": beta}

    # Final calibration stats from full validation set.
    centroids, class_scores = _prototype_stats(val_features, val_targets, num_classes)
    test_preds = np.argmax(test_probs, axis=1)
    test_softmax_conf = np.max(test_probs, axis=1)
    test_pvals = _conformal_pvalues(test_features, test_preds, centroids, class_scores)
    beta = best["beta"]
    test_conf = (test_softmax_conf ** (1.0 - beta)) * (test_pvals ** beta)

    # Keep class predictions unchanged; update only confidence ranking.
    test_probs_adjusted = np.copy(test_probs)
    # Inject conformal confidence into predicted-class probability for SC confidence extraction.
    rows = np.arange(test_probs_adjusted.shape[0])
    test_probs_adjusted[rows, test_preds] = test_conf

    # Re-normalize to valid probabilities.
    test_probs_adjusted = np.clip(test_probs_adjusted, 1e-8, 1.0)
    test_probs_adjusted = test_probs_adjusted / np.sum(test_probs_adjusted, axis=1, keepdims=True)

    return {
        "name": "Method 3: Prototype Conformal Confidence",
        "params": {"beta": float(beta)},
        "test_probs": test_probs_adjusted,
        "test_conf_override": test_conf,
    }


def compute_empirical_etf(features: np.ndarray, targets: np.ndarray, num_classes: int) -> np.ndarray:
    """
    Phase 1: Extract empirical class centroids from validation features.
    
    Args:
        features: [N, d] validation feature vectors
        targets: [N] class labels
        num_classes: number of classes
    
    Returns:
        empirical_W: [C, d] L2-normalized class centroids
    """
    d = features.shape[1]
    empirical_W = np.zeros((num_classes, d), dtype=np.float32)
    
    for c in range(num_classes):
        idx = np.where(targets == c)[0]
        if len(idx) > 0:
            class_mean = np.mean(features[idx], axis=0)
            class_mean = class_mean / (np.linalg.norm(class_mean) + 1e-8)
            empirical_W[c] = class_mean
    
    return empirical_W


def compute_socp_confidence(
    features: np.ndarray,
    empirical_W: np.ndarray,
    gamma: float,
) -> np.ndarray:
    """
    Phase 2: Core SOCP function - compute orthogonal distance confidence.
    
    For each feature h:
    1. Find top-2 predicted classes i, j via cosine similarity with empirical_W
    2. Construct orthonormal basis {u1, u2} for the 2D plane S_{i,j} via Gram-Schmidt
    3. Project h onto plane: h_parallel = h^T u1 * u1 + h^T u2 * u2
    4. Compute orthogonal component: h_perp = h - h_parallel
    5. Confidence = margin / (1 + gamma * ||h_perp||_2)
       (Points far from the 2D plane are more uncertain)
    
    Args:
        features: [N, d] test feature vectors
        empirical_W: [C, d] L2-normalized class centroids
        gamma: penalty scalar for orthogonal distance
    
    Returns:
        confs: [N] SOCP confidence scores
    """
    N = features.shape[0]
    C = empirical_W.shape[0]
    d = features.shape[1]
    
    # Compute cosine similarities: [N, C]
    # Normalize features to unit norm for cleaner cosine similarity
    features_norm = features / (np.linalg.norm(features, axis=1, keepdims=True) + 1e-8)
    sims = features_norm @ empirical_W.T  # [N, C]
    
    # Get top-2 predictions per sample
    top2_indices = np.argsort(sims, axis=1)[:, -2:]  # [N, 2], ascending
    i_indices = top2_indices[:, 1]  # top-1 predicted class
    j_indices = top2_indices[:, 0]  # top-2 predicted class
    
    confs = np.zeros(N, dtype=np.float32)
    
    for n in range(N):
        h = features[n]  # [d]
        i = i_indices[n]
        j = j_indices[n]
        w_i = empirical_W[i]  # [d]
        w_j = empirical_W[j]  # [d]
        
        # Gram-Schmidt orthonormalization to build basis for plane S_{i,j}
        u1 = w_i / (np.linalg.norm(w_i) + 1e-8)
        v2 = w_j - np.dot(w_j, u1) * u1
        u2 = v2 / (np.linalg.norm(v2) + 1e-8)
        
        # Project onto plane
        h_parallel = np.dot(h, u1) * u1 + np.dot(h, u2) * u2
        h_perp = h - h_parallel
        ortho_dist = np.linalg.norm(h_perp)
        
        # Margin between top-2 classes (from logits-like representation)
        margin = np.dot(w_i, h) - np.dot(w_j, h)
        
        # SOCP confidence: penalty for being far from the 2D channel
        # Higher gamma means stronger penalty for orthogonal distance
        confs[n] = margin / (1.0 + gamma * np.clip(ortho_dist, 0, 100))
    
    return confs


def method_4_socp_confidence(
    val_features,
    val_probs,
    val_targets,
    test_features,
    test_probs,
    num_classes,
):
    """Method 4: Simplex-Orthogonal Channel Projection (SOCP) confidence."""
    
    # Phase 3: Split-validation tuning
    n = val_features.shape[0]
    split = n // 2
    
    ref_features = val_features[:split]
    ref_targets = val_targets[:split]
    calib_features = val_features[split:]
    calib_targets = val_targets[split:]
    calib_probs = val_probs[split:]
    
    # Compute empirical ETF on reference set
    empirical_W = compute_empirical_etf(ref_features, ref_targets, num_classes)
    
    # Grid search for optimal gamma on calibration set
    gamma_candidates = [0.01, 0.05, 0.1, 0.2, 0.5, 1.0]
    best = None
    
    for gamma in gamma_candidates:
        calib_confs = compute_socp_confidence(calib_features, empirical_W, gamma)
        calib_preds = np.argmax(calib_probs, axis=1)
        metrics = evaluate_sc_metrics(calib_preds, calib_confs, calib_targets)
        
        if best is None or metrics["aurc"] < best["score"]:
            best = {"score": metrics["aurc"], "gamma": gamma}
    
    # Phase 4: Final evaluation
    # Recompute empirical_W on full validation set
    empirical_W = compute_empirical_etf(val_features, val_targets, num_classes)
    
    # Apply SOCP to test set with optimal gamma
    test_confs = compute_socp_confidence(test_features, empirical_W, best["gamma"])
    
    # Return confidences as override (keep original predictions)
    test_probs_copy = np.copy(test_probs)
    
    return {
        "name": "Method 4: Simplex-Orthogonal Channel Projection (SOCP)",
        "params": {"gamma": float(best["gamma"])},
        "test_probs": test_probs_copy,
        "test_conf_override": test_confs,
    }


def energy_uncertainty(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """Energy-OOD definition: E(x) = -T * logsumexp(logits / T)."""
    scaled = logits / float(temperature)
    shift = np.max(scaled, axis=1, keepdims=True)
    logsumexp = np.log(np.sum(np.exp(scaled - shift), axis=1) + 1e-12) + shift.squeeze(1)
    return -float(temperature) * logsumexp


# ==================================================================
# Baseline Integrations: KNN-OOD (Sun et al.) & DistShift-SC (Liang et al.)
# ==================================================================

def get_final_layer_params(model: torch.nn.Module):
    """Helper to extract weights and bias of the final linear layer for geometric margin."""
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


def method_knn_ood(val_features, test_features, test_probs, k=50):
    """KNN-OOD baseline faithful to original repo (Sun et al. ICML 2022)."""
    v = val_features.numpy() if torch.is_tensor(val_features) else val_features
    t = test_features.numpy() if torch.is_tensor(test_features) else test_features
    
    # Normalization as per repo: x / (||x|| + 1e-10)
    v_norm = v / (np.linalg.norm(v, ord=2, axis=-1, keepdims=True) + 1e-10)
    t_norm = t / (np.linalg.norm(t, ord=2, axis=-1, keepdims=True) + 1e-10)
    
    index = faiss.IndexFlatL2(v_norm.shape[1])
    index.add(v_norm.astype(np.float32))
    D, _ = index.search(t_norm.astype(np.float32), k)
    
    # Confidence is negative of k-th nearest neighbor distance
    # The repo uses -D[:, -1] where D is squared L2 distance.
    test_confs = -D[:, -1]
    
    return {
        "name": f"KNN-OOD (k={k})",
        "params": {"k": k},
        "test_probs": test_probs,
        "test_conf_override": test_confs,
    }


def method_rl_conf_m(test_logits, test_probs):
    """RL_conf-M: Margin-based SC (Top1 - Top2 logit) from Liang et al. 2024."""
    val, _ = torch.topk(test_logits, 2, dim=1)
    conf_m = (val[:, 0] - val[:, 1]).cpu().numpy()
    return {
        "name": "RL_conf-M (Logit Margin)",
        "params": {},
        "test_probs": test_probs,
        "test_conf_override": conf_m,
    }


def method_rl_geo_m(test_logits, test_probs, model):
    """RL_geo-M: Geometric margin normalized by weight norms from Liang et al. 2024."""
    w, b = get_final_layer_params(model)
    if w is None:
        return None
        
    # Augment weights with bias: [classes, dim+1]
    w_aug = torch.cat([w, b.unsqueeze(1)], dim=1)
    w_norm = torch.norm(w_aug, p=2, dim=1)
    
    # Geometric distance to each hyperplane: logit / ||w_aug||
    geo_dist = test_logits / w_norm.unsqueeze(0)
    gv, _ = torch.topk(geo_dist, 2, dim=1)
    geo_m = (gv[:, 0] - gv[:, 1]).cpu().numpy()
    
    return {
        "name": "RL_geo-M (Geometric Margin)",
        "params": {},
        "test_probs": test_probs,
        "test_conf_override": geo_m,
    }


def method_sr_ent(test_probs):
    """SR_ent: Negative entropy of softmax response (ACCV 2022)."""
    # test_probs: [N, C]
    ent = sstats.entropy(test_probs, axis=1)
    return {
        "name": "SR_ent (Negative Entropy)",
        "params": {},
        "test_probs": test_probs,
        "test_conf_override": -ent,
    }


def method_vim_and_sirc(val_features, val_logits, test_features, test_logits, test_probs, model, vim_dim=64):
    """ViM (Wang et al. 2022) and SIRC (Xia et al. ACCV 2022) combined implementation."""
    w, b = get_final_layer_params(model)
    if w is None:
        return None, None

    # 1. Compute ViM parameters on validation set
    u = -torch.linalg.pinv(w) @ b
    centered_val = val_features - u.unsqueeze(0)
    
    # Null Space (NS) extraction via Eigen-decomposition
    cov = (centered_val.T @ centered_val) / centered_val.shape[0]
    eig_vals, eig_vecs = torch.linalg.eigh(cov)
    idx = torch.argsort(eig_vals, descending=True)
    NS = eig_vecs[:, idx][:, vim_dim:]
    
    # alpha scaling
    vlogit_val = torch.norm(centered_val @ NS, p=2, dim=-1)
    max_logit_val = val_logits.max(dim=1).values
    alpha = max_logit_val.mean() / vlogit_val.mean()
    
    # 2. Compute ViM scores on test set
    centered_test = test_features - u.unsqueeze(0)
    vlogit_test = torch.norm(centered_test @ NS, p=2, dim=-1) * alpha
    energy_test = torch.logsumexp(test_logits, dim=1)
    vim_confs = energy_test - vlogit_test
    
    # 3. SIRC: Combined MSP and ViM residual
    vlogit_val_scaled = vlogit_val * alpha
    res_unc_val = vlogit_val_scaled.cpu().numpy()
    a = np.mean(res_unc_val) - 3 * np.std(res_unc_val)
    b = 1.0 / (np.std(res_unc_val) + 1e-12)
    
    msp = np.max(test_probs, axis=1)
    msp = np.clip(msp, 0.0, 1.0 - 1e-12) 
    soft = np.log(1.0 - msp)
    
    res_unc_test = vlogit_test.cpu().numpy()
    additional = np.logaddexp(0.0, -b * (res_unc_test - a))
    sirc_confs = -soft - additional
    
    vim_res = {
        "name": "ViM (Virtual Logit Matching)",
        "params": {"vim_dim": vim_dim},
        "test_probs": test_probs,
        "test_conf_override": vim_confs.cpu().numpy(),
    }
    
    sirc_res = {
        "name": "SIRC (MSP + ViM-Res)",
        "params": {"a": float(a), "b": float(b)},
        "test_probs": test_probs,
        "test_conf_override": sirc_confs,
    }
    
    return vim_res, sirc_res


def method_energy(val_logits, val_targets, test_logits):
    """Energy baseline faithful to energy_ood (confidence = -energy uncertainty)."""
    test_logits_np = test_logits.cpu().numpy() if isinstance(test_logits, torch.Tensor) else test_logits
    energy_u = energy_uncertainty(test_logits_np, temperature=1.0)
    test_confs = -energy_u
    
    return {
        "name": "Energy Score",
        "params": {"temperature": 1.0},
        "test_probs": test_logits_np,
        "test_conf_override": test_confs,
    }


def compute_mahalanobis_confidence(test_probs: np.ndarray, val_probs: np.ndarray, val_targets: np.ndarray, num_classes: int) -> np.ndarray:
    """DOCTOR-style Mahalanobis on class-score vectors (faithful matrix form)."""
    dim = val_probs.shape[1]
    means_by_class = np.zeros((num_classes, dim), dtype=np.float64)
    for c in range(num_classes):
        idx = np.where(val_targets == c)[0]
        if len(idx) > 0:
            means_by_class[c] = np.mean(val_probs[idx], axis=0)

    # Compute covariance and invert it to get the precision matrix (Inverse Covariance)
    # as specified in the original Mahalanobis paper.
    cov = np.cov(val_probs.T)
    if cov.ndim == 0:
        cov = np.eye(dim, dtype=np.float64)
    
    # Add a small epsilon for numerical stability before inversion
    precision = np.linalg.inv(cov + 1e-8 * np.eye(dim, dtype=np.float64))

    # Compute Mahalanobis distance: sqrt((x-mu)^T * precision * (x-mu))
    diffs = test_probs[:, None, :] - means_by_class[None, :, :]
    d2 = np.einsum("ncd,df,ncf->nc", diffs, precision, diffs)
    d = np.sqrt(np.maximum(d2, 0.0))
    min_dist = np.min(d, axis=1)
    return 1.0 / (1.0 + min_dist)


def method_mahalanobis(val_probs, val_targets, test_probs, test_logits, num_classes):
    """Mahalanobis confidence following DOCTOR score-space computation."""
    test_confs = compute_mahalanobis_confidence(test_probs, val_probs, val_targets, num_classes)
    test_logits_np = test_logits.cpu().numpy() if isinstance(test_logits, torch.Tensor) else test_logits
    
    return {
        "name": "Mahalanobis Distance",
        "params": {},
        "test_probs": test_logits_np,
        "test_conf_override": test_confs,
    }


def _centralize_np(logits: np.ndarray) -> np.ndarray:
    return logits - np.mean(logits, axis=1, keepdims=True)


def _normalize_np(logits: np.ndarray, p: int, eps: float = 1e-12) -> np.ndarray:
    if p == 0:
        denom = np.count_nonzero(logits, axis=1, keepdims=True).astype(np.float64)
        denom = np.clip(denom, 1.0, None)
        return logits / denom
    denom = np.linalg.norm(logits, ord=p, axis=1, keepdims=True)
    denom = np.clip(denom, eps, None)
    return logits / denom


def _msp_np(logits: np.ndarray) -> np.ndarray:
    shift = np.max(logits, axis=1, keepdims=True)
    probs = np.exp(logits - shift)
    probs /= np.sum(probs, axis=1, keepdims=True)
    return np.max(probs, axis=1)


def method_5_maxlogit_pnorm(val_logits, val_targets, test_logits):
    """FixSelective MaxLogit-pNorm with p-grid optimization and MSP fallback."""
    val_logits_np = val_logits.detach().cpu().numpy() if isinstance(val_logits, torch.Tensor) else val_logits
    test_logits_np = test_logits.detach().cpu().numpy() if isinstance(test_logits, torch.Tensor) else test_logits
    val_targets_np = val_targets.detach().cpu().numpy() if isinstance(val_targets, torch.Tensor) else val_targets

    val_centered = _centralize_np(val_logits_np)
    test_centered = _centralize_np(test_logits_np)
    val_preds = np.argmax(val_logits_np, axis=1)

    # MSP fallback used in FixSelective optimize.p.
    best_p = "MSP"
    best_score = evaluate_sc_metrics(val_preds, _msp_np(val_logits_np), val_targets_np)["aurc"]

    for p in range(10):
        val_conf = np.max(_normalize_np(val_centered, p=p), axis=1)
        score = evaluate_sc_metrics(val_preds, val_conf, val_targets_np)["aurc"]
        if score < best_score:
            best_score = score
            best_p = p

    if best_p == "MSP":
        test_conf = _msp_np(test_logits_np)
    else:
        test_conf = np.max(_normalize_np(test_centered, p=int(best_p)), axis=1)

    return {
        "name": "MaxLogit pNorm",
        "params": {"p": best_p},
        "test_probs": test_logits_np,
        "test_conf_override": test_conf,
    }


def method_6_maxlogit_pnorm_temperature(val_logits, val_targets, test_logits):
    """FixSelective p-and-T optimization (pNorm+) using MSP on p-normalized logits."""
    val_logits_np = val_logits.detach().cpu().numpy() if isinstance(val_logits, torch.Tensor) else val_logits
    test_logits_np = test_logits.detach().cpu().numpy() if isinstance(test_logits, torch.Tensor) else test_logits
    val_targets_np = val_targets.detach().cpu().numpy() if isinstance(val_targets, torch.Tensor) else val_targets

    val_centered = _centralize_np(val_logits_np)
    test_centered = _centralize_np(test_logits_np)
    val_preds = np.argmax(val_logits_np, axis=1)

    t_range = np.arange(0.01, 2.0, 0.01)
    best = None

    for p in range(10):
        val_norm = np.clip(np.linalg.norm(val_centered, ord=p if p != 0 else 2, axis=1, keepdims=True), 1e-12, None)
        for t in t_range:
            # Follow FixSelective default: rescale_T=True
            t_eff = t / float(np.mean(val_norm))
            val_scaled = val_centered / (t_eff * val_norm)
            val_conf = _msp_np(val_scaled)
            score = evaluate_sc_metrics(val_preds, val_conf, val_targets_np)["aurc"]
            if best is None or score < best["score"]:
                best = {"score": score, "p": p, "temperature": float(t_eff)}

    p_best = int(best["p"])
    test_norm = np.clip(np.linalg.norm(test_centered, ord=p_best if p_best != 0 else 2, axis=1, keepdims=True), 1e-12, None)
    test_scaled = test_centered / (best["temperature"] * test_norm)
    test_conf = _msp_np(test_scaled)

    return {
        "name": "MaxLogit pNorm+",
        "params": {"p": p_best, "temperature": float(best["temperature"])},
        "test_probs": test_logits_np,
        "test_conf_override": test_conf,
    }


def method_odin(model: torch.nn.Module, test_inputs, device: torch.device, temperature: float = 1000.0, epsilon: float = 0.0014, odin_batch_size: int = 64):
    """ODIN implementation faithful to original code (temperature + input gradient perturbation)."""
    if test_inputs is None:
        return None

    model.eval()
    all_probs = []
    all_confs = []

    if isinstance(test_inputs, np.ndarray):
        inputs_np = test_inputs
    else:
        inputs_np = test_inputs.cpu().numpy()

    # CIFAR normalization constants used by original ODIN/energy_ood code.
    channel_div = [63.0 / 255.0, 62.1 / 255.0, 66.7 / 255.0]

    for start in range(0, inputs_np.shape[0], odin_batch_size):
        end = min(start + odin_batch_size, inputs_np.shape[0])
        batch = torch.from_numpy(inputs_np[start:end]).float().to(device)
        batch.requires_grad_(True)

        logits, _ = extract_logits_and_features(model, batch)
        pseudo = torch.argmax(logits.detach(), dim=1)
        loss = F.cross_entropy(logits / temperature, pseudo)
        loss.backward()

        gradient = torch.ge(batch.grad.data, 0).float()
        gradient = (gradient - 0.5) * 2
        if gradient.shape[1] == 3:
            for c in range(3):
                gradient[:, c] = gradient[:, c] / channel_div[c]

        perturbed = torch.add(batch.data, -epsilon, gradient)
        with torch.no_grad():
            logits_p, _ = extract_logits_and_features(model, perturbed)
            probs = F.softmax(logits_p / temperature, dim=1)
            confs = torch.max(probs, dim=1).values

        all_probs.append(probs.cpu().numpy())
        all_confs.append(confs.cpu().numpy())

    test_probs = np.concatenate(all_probs, axis=0)
    test_conf = np.concatenate(all_confs, axis=0)
    return {
        "name": "ODIN",
        "params": {
            "temperature": float(temperature),
            "epsilon": float(epsilon),
            "batch_size": int(odin_batch_size),
        },
        "test_probs": test_probs,
        "test_conf_override": test_conf,
    }


def doctor_alpha_confidence(probs: np.ndarray, alpha: float = 2.0) -> np.ndarray:
    """DOCTOR Alpha: 1 - sum(P_i^alpha) converts to ratio form."""
    g_x = 1.0 - np.sum(np.power(probs, alpha), axis=1)
    return g_x / (1.0 - g_x + 1e-8)


def doctor_beta_confidence(probs: np.ndarray) -> np.ndarray:
    """DOCTOR Beta: 1 - P(predicted_class) converts to ratio form."""
    pred_probs = np.max(probs, axis=1)
    b_x = 1.0 - pred_probs
    return b_x / (1.0 - b_x + 1e-8)


def method_doctor_alpha(val_probs, val_targets, test_probs, num_classes):
    """DOCTOR Alpha method with grid search on alpha."""
    split = val_probs.shape[0] // 2
    ref_probs = val_probs[:split]
    calib_probs = val_probs[split:]
    calib_targets = val_targets[split:]
    calib_preds = np.argmax(calib_probs, axis=1)
    
    alpha_candidates = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    best = None
    
    for alpha in alpha_candidates:
        calib_conf = doctor_alpha_confidence(calib_probs, alpha=alpha)
        metrics = evaluate_sc_metrics(calib_preds, calib_conf, calib_targets)
        if best is None or metrics["aurc"] < best["score"]:
            best = {"score": metrics["aurc"], "alpha": alpha}
    
    test_conf = doctor_alpha_confidence(test_probs, alpha=best["alpha"])
    test_probs_copy = np.copy(test_probs)
    
    return {
        "name": "DOCTOR-Alpha",
        "params": {"alpha": float(best["alpha"])},
        "test_probs": test_probs_copy,
        "test_conf_override": test_conf,
    }


def method_doctor_beta(val_probs, val_targets, test_probs, num_classes):
    """DOCTOR Beta method."""
    test_conf = doctor_beta_confidence(test_probs)
    test_probs_copy = np.copy(test_probs)
    
    return {
        "name": "DOCTOR-Beta",
        "params": {},
        "test_probs": test_probs_copy,
        "test_conf_override": test_conf,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate post-hoc baseline methods for Mixup SC")
    
    # Build dataset choices (include Tiny-ImageNet)
    dataset_choices = ["cifar10", "cifar100"] + list(MEDMNIST_DATASETS.keys()) + ["tinyimagenet"]
    
    parser.add_argument("--dataset", type=str, default="cifar10", choices=dataset_choices)
    parser.add_argument("--arch", type=str, default="resnet18", choices=["resnet18", "resnet34", "resnet50", "resnet101", "vgg16_bn", "vit_b_16", "vit_b_4"])
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--checkpoint-dir", type=str, default="./checkpoints")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default="./results/posthoc_summary.json")
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = get_device()
    print(f"Using device: {device}")

    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_path = Path(args.checkpoint) if args.checkpoint else find_best_mixup_checkpoint(checkpoint_dir, args.dataset)
    print(f"Selected checkpoint: {checkpoint_path}")

    ckpt = torch.load(checkpoint_path, map_location=device)
    cfg = ckpt.get("config", {})

    # Use requested dataset to avoid leakage across checkpoint metadata.
    arch = cfg.get("model", {}).get("arch", args.arch)
    dataset = args.dataset.lower()
    ckpt_dataset = cfg.get("data", {}).get("dataset", "unknown")
    if ckpt_dataset != "unknown" and ckpt_dataset.lower() != dataset:
        print(f"Warning: checkpoint metadata dataset='{ckpt_dataset}' differs from requested dataset='{dataset}'.")
    
    # Auto-determine num_classes based on dataset
    if dataset == "cifar10":
        num_classes = 10
    elif dataset == "cifar100":
        num_classes = 100
    elif dataset.lower() in MEDMNIST_DATASETS:
        num_classes = MEDMNIST_DATASETS[dataset.lower()]["num_classes"]
    elif dataset == "tinyimagenet":
        num_classes = 200
    else:
        num_classes = 10  # Default fallback
        print(f"Warning: Unknown dataset {dataset}, defaulting to 10 classes")

    print(f"Checkpoint metadata -> dataset: {dataset}, arch: {arch}, num_classes: {num_classes}")

    train_loader, val_loader, test_loader, _ = get_dataloaders(
        dataset=dataset,
        data_dir="./data",
        batch_size=args.batch_size,
        val_batch_size=args.batch_size,
        num_workers=4,
        augmentation=False,
        seed=args.seed,
    )

    input_size = 64 if dataset.lower() == "tinyimagenet" else 32
    model = get_model(arch, num_classes=num_classes, input_size=input_size).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    val_data = collect_outputs(model, val_loader, device)
    test_data = collect_outputs(model, test_loader, device)

    # Baseline from pre-trained Mixup checkpoint.
    baseline = evaluate_sc_metrics(test_data["preds"], test_data["confs"], test_data["targets"])

    # print("\n===== Phase 1: Hypotheses =====")
    # print("RQ1 (Logit calibration): Can class-wise temperatures reduce overconfidence and improve RC ranking?")
    # print("RQ2 (Latent distance): Can local neighborhood evidence in feature space refine uncertain predictions?")
    # print("RQ3 (Conformal features): Can prototype-distance conformal p-values improve coverage-risk ordering?")
    # print("RQ4 (Orthogonal projection): Can isolating Mixup channel artifacts via SOCP improve coverage-risk?")

    print("\n===== Phase 2: Execute Baseline Post-hoc Methods =====")

    val_logits_t = val_data["logits"].to(device)
    val_targets_t = torch.tensor(val_data["targets"], dtype=torch.long, device=device)
    test_logits_t = test_data["logits"].to(device)
    val_inputs_t = val_data["inputs"] if "inputs" in val_data else None
    test_inputs_t = test_data["inputs"] if "inputs" in test_data else None

    methods = [
        # Baseline simple methods
        method_energy(val_logits_t, val_targets_t, test_logits_t),
        method_mahalanobis(val_data["probs"], val_data["targets"], test_data["probs"], test_logits_t, num_classes),
        
        # DOCTOR methods
        method_doctor_alpha(val_data["probs"], val_data["targets"], test_data["probs"], num_classes),
        method_doctor_beta(val_data["probs"], val_data["targets"], test_data["probs"], num_classes),
        
        # pnorm and pnorm+temperature
        method_5_maxlogit_pnorm(val_logits_t, val_targets_t, test_logits_t),
        method_6_maxlogit_pnorm_temperature(val_logits_t, val_targets_t, test_logits_t),

        # ODIN
        method_odin(model, test_inputs_t, device) if test_inputs_t is not None else None,

        # kNN-OOD (Sun et al. ICML 2022)
        method_knn_ood(val_data["features"], test_data["features"], test_data["probs"], k=50),

        # Selective Classification Under Distribution Shifts (Liang et al. 2024)
        method_rl_conf_m(test_logits_t, test_data["probs"]),
        method_rl_geo_m(test_logits_t, test_data["probs"], model),
        
        # SR_ent and SIRC
        method_sr_ent(test_data["probs"]),
        *method_vim_and_sirc(torch.from_numpy(val_data["features"]).float().to(device), val_logits_t, torch.from_numpy(test_data["features"]).float().to(device), test_logits_t, test_data["probs"], model),

        # Feature-space kNN blending
        method_2_feature_knn_logit_blend(
            val_data["features"],
            val_data["probs"],
            val_data["targets"],
            test_data["features"],
            test_data["probs"],
            num_classes,
        ),
        
    ]
    
    # Remove None entries (if ODIN is not available)
    methods = [m for m in methods if m is not None]

    print("\n===== Phase 3: Evaluate (Accuracy, AURC, E-AURC) =====")
    rows = []
    rows.append({"method": "Baseline Mixup", **baseline, "params": {}})

    for method in methods:
        probs = method["test_probs"]
        preds = np.argmax(probs, axis=1)

        if "test_conf_override" in method:
            confs = method["test_conf_override"]
        else:
            confs = np.max(probs, axis=1)

        print(f"Evaluating {method['name']}...")
        metrics = evaluate_sc_metrics(preds, confs, test_data["targets"])
        rows.append({"method": method["name"], **metrics, "params": method["params"]})

    # Comparative summary sorted by AURC (lower better).
    baseline_aurc = rows[0]["aurc"]
    improved = [r for r in rows[1:] if r["aurc"] < baseline_aurc and r["eaurc"] <= rows[0]["eaurc"]]

    print("\nComparative Summary (Test):")
    print(f"{'Method':50s} {'Acc':>8s} {'AURC':>10s} {'E-AURC':>10s}")
    print("-" * 82)
    for r in rows:
        print(f"{r['method'][:50]:50s} {r['accuracy']:8.4f} {r['aurc']:10.4f} {r['eaurc']:10.4f}")

    if improved:
        best = sorted(improved, key=lambda x: x["aurc"])[0]
        conclusion = (
            f"Primary candidate: {best['method']} (AURC {best['aurc']:.4f} vs baseline {baseline_aurc:.4f})."
        )
    else:
        conclusion = "All post-hoc baseline methods evaluated on this checkpoint."

    print("\nConclusion:")
    print(conclusion)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "checkpoint": str(checkpoint_path),
        "dataset": dataset,
        "arch": arch,
        "results": rows,
        "conclusion": conclusion,
    }
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"\nSaved summary to: {out_path}")


if __name__ == "__main__":
    main()
