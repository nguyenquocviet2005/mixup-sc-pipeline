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
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import faiss
import scipy.stats as sstats
from scipy.special import logsumexp
from sklearn.neighbors import NearestNeighbors

import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from data import get_dataloaders, MEDMNIST_DATASETS
from evaluation import SelectionMetrics, CalibrationMetrics
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
    # Check if it's a standard ResNet with maxpool (like ResNet50/101 on ImageNet/Medical datasets)
    if hasattr(model, 'maxpool') and hasattr(model, 'avgpool'):
        features_list = []
        def hook(module, input, output):
            features_list.append(output)
        handle = model.avgpool.register_forward_hook(hook)
        logits = model(inputs)
        handle.remove()
        if len(features_list) > 0:
            feat = features_list[0]
            feat = feat.view(feat.size(0), -1)
            return logits, feat

    # Check if we want to use uncompressed features for ResNet110 (layer3 output before pooling)
    if hasattr(model, 'layer3') and hasattr(model, 'avgpool'):
        features_list = []
        def hook(module, input, output):
            features_list.append(output)
        handle = model.layer3.register_forward_hook(hook)
        
        # We need to handle if the model returns logits or something else
        # Most models here return logits in forward
        try:
            logits = model(inputs)
            if isinstance(logits, tuple):
                logits = logits[0]
        except:
            handle.remove()
            raise
            
        handle.remove()
        
        if len(features_list) > 0:
            feat = features_list[0]
            # For CIFAR ResNet110, channels = 64. 
            if feat.size(1) == 64:
                feat_pooled = F.adaptive_avg_pool2d(feat, (1, 1)).view(feat.size(0), -1)
                return logits, feat_pooled
    
    # Fallback to existing logic
    if hasattr(model, '_use_feature_output'):
        if model._use_feature_output:
            return model(inputs, feature_output=True)
    else:
        try:
            logits, features = model(inputs, feature_output=True)
            model._use_feature_output = True
            return logits, features
        except TypeError:
            model._use_feature_output = False

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
    elif hasattr(model, 'block1') and hasattr(model, 'block2') and hasattr(model, 'block3'):
        # WideResNet from FMFP
        out = model.conv1(inputs)
        out = model.block1(out)
        out = model.block2(out)
        out = model.block3(out)
        out = model.relu(model.bn1(out))
        out = F.adaptive_avg_pool2d(out, (1, 1))
        features = out.view(out.size(0), -1)
        logits = model.fc(features)
        return logits, features
    elif isinstance(model, torch.nn.Sequential):
        features = model[:-1](inputs)
        logits = model[-1](features)
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


def collect_outputs(model: torch.nn.Module, loader, device: torch.device, keep_inputs: bool = False):
    model.eval()
    all_logits = []
    all_features = []
    all_targets = []
    all_inputs = [] if keep_inputs else None

    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device)
            logits, features = extract_logits_and_features(model, inputs)
            all_logits.append(logits.cpu())
            all_features.append(features.cpu())
            all_targets.append(targets.cpu())
            if keep_inputs:
                all_inputs.append(inputs.cpu())

    logits = torch.cat(all_logits, dim=0)
    features = torch.cat(all_features, dim=0)
    targets = torch.cat(all_targets, dim=0)

    probs = F.softmax(logits, dim=1).numpy()
    preds = np.argmax(probs, axis=1)
    confs = np.max(probs, axis=1)

    features_np = features.numpy()
    targets_np = targets.numpy()
    del features, targets, all_features, all_targets

    result = {
        "logits": logits,
        "features": features_np,
        "targets": targets_np,
        "probs": probs,
        "preds": preds,
        "confs": confs,
    }
    if keep_inputs:
        result["inputs"] = torch.cat(all_inputs, dim=0).numpy()
    return result


def evaluate_sc_metrics(preds: np.ndarray, confs: np.ndarray, targets: np.ndarray, probs: np.ndarray = None) -> Dict[str, float]:
    if not np.all(np.isfinite(confs)):
        print(f"  [DEBUG] Non-finite confidence scores detected! Samples with non-finite values: {np.sum(~np.isfinite(confs))}")
        confs = np.nan_to_num(confs, nan=0.0, posinf=1e6, neginf=-1e10)
    
    correctness = (preds == targets).astype(int)
    sc = SelectionMetrics.compute_all_metrics(confs, correctness)
    acc = float(np.mean(correctness))
    
    res = {
        "accuracy": acc,
        "auroc": float(sc["auroc"]),
        "aurc": float(sc["aurc"]),
        "eaurc": float(sc["eaurc"]),
        "naurc": float(sc["naurc"]),
        "ece": float(CalibrationMetrics.compute_ece(confs, correctness)),
    }
    
    if probs is not None:
        res["brier"] = float(CalibrationMetrics.compute_brier_score(probs, targets))
        res["nll"] = float(CalibrationMetrics.compute_nll(probs, targets))
    else:
        res["brier"] = -1.0
        res["nll"] = -1.0
        
    return res


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
    fixed_k=None,
    fixed_alpha=None,
    confidence_type="max_prob",
):
    """Method 2: Feature-space kNN probability blending."""
    if fixed_k is not None and fixed_alpha is not None:
        best = {"k": fixed_k, "alpha": fixed_alpha}
    else:
        n = val_features.shape[0]
        split = n // 2

        ref_features = val_features[:split]
        ref_targets = val_targets[:split]
        calib_features = val_features[split:]
        calib_targets = val_targets[split:]
        calib_base_probs = val_probs[split:]

        candidate_k = [5, 10, 20, 30, 50, 75, 100, 150, 200, 250]
        candidate_k = [k for k in candidate_k if k <= split]
        candidate_alpha = [0.1, 0.25, 0.4, 0.55, 0.7, 0.8, 0.85, 0.9, 1.0]

        best = None

        for k in candidate_k:
            knn = KNeighborsClassifier(n_neighbors=k, weights="distance")
            knn.fit(ref_features, ref_targets)
            calib_knn_probs = full_probs_from_knn(knn, calib_features, num_classes)

            for alpha in candidate_alpha:
                probs = (1.0 - alpha) * calib_base_probs + alpha * calib_knn_probs
                preds = np.argmax(probs, axis=1)
                
                if confidence_type == "neg_entropy":
                    ent = sstats.entropy(probs, axis=1)
                    confs = 1.0 - (ent / np.log(num_classes))
                else:
                    confs = np.max(probs, axis=1)
                    
                metrics = evaluate_sc_metrics(preds, confs, calib_targets)
                score = metrics["auroc"]
                if best is None or score > best["score"]:
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

    res = {
        "name": "Method 2: Feature-kNN / Logit Probability Blending",
        "params": {"k": int(best["k"]), "alpha": float(best["alpha"]), "confidence_type": confidence_type},
        "test_probs": test_probs_blend,
    }
    
    if confidence_type == "neg_entropy":
        ent = sstats.entropy(test_probs_blend, axis=1)
        res["test_conf_override"] = 1.0 - (ent / np.log(num_classes))
        
    return res







def method_affine_shift_knn(val_features, val_probs, val_targets, test_features, test_probs, num_classes):
    """Affine Feature-Shift (Carratino Correction)."""
    mean_feat = np.mean(val_features, axis=0, keepdims=True)
    
    n = val_features.shape[0]
    split = n // 2

    # Grid search theta
    candidate_theta = [0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
    candidate_k = [k for k in [5, 10, 20, 30, 100] if k <= split]
    if not candidate_k and split > 0:
        candidate_k = [split]
    candidate_alpha = [0.0, 0.1, 0.25, 0.4, 0.55, 0.7, 0.8, 0.85, 1.0]

    best = None
    
    # Validation split
    ref_features_base = val_features[:split]
    ref_targets = val_targets[:split]
    calib_features_base = val_features[split:]
    calib_targets = val_targets[split:]
    calib_base_probs = val_probs[split:]
    
    def project(feat):
        norm = np.linalg.norm(feat, axis=1, keepdims=True)
        return feat / np.clip(norm, 1e-12, None)

    for theta in candidate_theta:
        ref_features = project(theta * ref_features_base + (1 - theta) * mean_feat)
        calib_features = project(theta * calib_features_base + (1 - theta) * mean_feat)
        
        for k in candidate_k:
            knn = KNeighborsClassifier(n_neighbors=k, weights="distance")
            knn.fit(ref_features, ref_targets)
            calib_knn_probs = full_probs_from_knn(knn, calib_features, num_classes)

            for alpha in candidate_alpha:
                probs = (1.0 - alpha) * calib_base_probs + alpha * calib_knn_probs
                preds = np.argmax(probs, axis=1)
                confs = np.max(probs, axis=1)
                    
                metrics = evaluate_sc_metrics(preds, confs, calib_targets)
                score = metrics["aurc"]
                if best is None or score < best["score"]:
                    best = {"score": score, "k": k, "alpha": alpha, "theta": theta}

    # Refit
    theta = best["theta"]
    val_proj = project(theta * val_features + (1 - theta) * mean_feat)
    test_proj = project(theta * test_features + (1 - theta) * mean_feat)
    
    knn = KNeighborsClassifier(n_neighbors=best["k"], weights="distance")
    knn.fit(val_proj, val_targets)
    test_knn_probs = full_probs_from_knn(knn, test_proj, num_classes)
    test_probs_blend = (1.0 - best["alpha"]) * test_probs + best["alpha"] * test_knn_probs
    test_probs_blend = test_probs_blend / np.sum(test_probs_blend, axis=1, keepdims=True)
    
    return {
        "name": "Affine Feature-Shift kNN Blend",
        "params": {"k": int(best["k"]), "alpha": float(best["alpha"]), "theta": float(best["theta"])},
        "test_probs": test_probs_blend,
    }


def method_virtual_flat_minima(val_features, val_probs, val_targets, test_features, test_probs, num_classes):
    """Virtual Flat Minima (Neighborhood Perturbation)."""
    mean_feat = np.mean(val_features, axis=0, keepdims=True)
    def project(feat):
        centered = feat - mean_feat
        norm = np.linalg.norm(centered, axis=1, keepdims=True)
        return centered / np.clip(norm, 1e-12, None)
        
    val_proj = project(val_features)
    test_proj = project(test_features)
    
    n = val_proj.shape[0]
    split = n // 2

    ref_features = val_proj[:split]
    ref_targets = val_targets[:split]
    calib_features = val_proj[split:]
    calib_targets = val_targets[split:]
    calib_base_probs = val_probs[split:]

    candidate_k = [k for k in [10, 30, 50] if k <= split]
    if not candidate_k and split > 0:
        candidate_k = [split]
    candidate_sigma = [0.01, 0.05, 0.1]
    candidate_penalty = [0.1, 0.5, 1.0, 2.0]

    best = None
    N = 10

    for k in candidate_k:
        knn = KNeighborsClassifier(n_neighbors=k, weights="distance")
        knn.fit(ref_features, ref_targets)
        calib_knn_probs = full_probs_from_knn(knn, calib_features, num_classes)

        for sigma in candidate_sigma:
            noise = np.random.normal(0, sigma, size=(N, calib_features.shape[0], calib_features.shape[1]))
            pert_calib = calib_features[np.newaxis, :, :] + noise
            norm_pert = np.linalg.norm(pert_calib, axis=2, keepdims=True)
            pert_calib_proj = pert_calib / np.clip(norm_pert, 1e-12, None)
            
            pert_calib_flat = pert_calib_proj.reshape(-1, calib_features.shape[1])
            pert_knn_probs_flat = full_probs_from_knn(knn, pert_calib_flat, num_classes)
            pert_knn_probs = pert_knn_probs_flat.reshape(N, calib_features.shape[0], num_classes)
            
            preds_calib = np.argmax(calib_knn_probs, axis=1)
            probs_of_pred = np.zeros((N, calib_features.shape[0]))
            for i in range(N):
                probs_of_pred[i] = pert_knn_probs[i, np.arange(calib_features.shape[0]), preds_calib]
                
            variance = np.var(probs_of_pred, axis=0)
            
            for penalty in candidate_penalty:
                confs = np.max(calib_knn_probs, axis=1) - penalty * variance
                metrics = evaluate_sc_metrics(preds_calib, confs, calib_targets)
                score = metrics["aurc"]
                if best is None or score < best["score"]:
                    best = {"score": score, "k": k, "sigma": sigma, "penalty": penalty}

    # Refit
    knn = KNeighborsClassifier(n_neighbors=best["k"], weights="distance")
    knn.fit(val_proj, val_targets)
    test_knn_probs = full_probs_from_knn(knn, test_proj, num_classes)
    
    sigma = best["sigma"]
    penalty = best["penalty"]
    
    noise = np.random.normal(0, sigma, size=(N, test_proj.shape[0], test_proj.shape[1]))
    pert_test = test_proj[np.newaxis, :, :] + noise
    norm_pert = np.linalg.norm(pert_test, axis=2, keepdims=True)
    pert_test_proj = pert_test / np.clip(norm_pert, 1e-12, None)
    
    pert_test_flat = pert_test_proj.reshape(-1, test_features.shape[1])
    pert_knn_probs_flat = full_probs_from_knn(knn, pert_test_flat, num_classes)
    pert_knn_probs = pert_knn_probs_flat.reshape(N, test_proj.shape[0], num_classes)
    
    test_preds = np.argmax(test_knn_probs, axis=1)
    probs_of_pred_test = np.zeros((N, test_proj.shape[0]))
    for i in range(N):
        probs_of_pred_test[i] = pert_knn_probs[i, np.arange(test_proj.shape[0]), test_preds]
        
    variance_test = np.var(probs_of_pred_test, axis=0)
    
    test_conf_override = np.max(test_knn_probs, axis=1) - penalty * variance_test
    
    return {
        "name": "Virtual Flat Minima (S-kNN + Variance Penalty)",
        "params": {"k": int(best["k"]), "sigma": float(sigma), "penalty": float(penalty)},
        "test_probs": test_knn_probs,
        "test_conf_override": test_conf_override,
    }


def method_simplex_orthogonal_knn(val_features, val_probs, val_targets, test_features, test_probs, num_classes, model):
    """Simplex-Orthogonal kNN (Applying ETF Filter)."""
    if hasattr(model, 'module'):
        model = model.module
    
    W = None
    for name, module in reversed(list(model.named_modules())):
        if isinstance(module, torch.nn.Linear):
            W = module.weight.detach().cpu().numpy()
            break
            
    if W is None:
        print("Could not find linear classifier matrix W.")
        return None

    P = W.T @ np.linalg.pinv(W @ W.T) @ W

    val_proj = val_features @ P.T
    test_proj = test_features @ P.T
    
    mean_feat = np.mean(val_proj, axis=0, keepdims=True)
    def project_sphere(feat):
        centered = feat - mean_feat
        norm = np.linalg.norm(centered, axis=1, keepdims=True)
        return centered / np.clip(norm, 1e-12, None)
        
    val_proj_sphere = project_sphere(val_proj)
    test_proj_sphere = project_sphere(test_proj)
    
    n = val_proj_sphere.shape[0]
    split = n // 2

    ref_features = val_proj_sphere[:split]
    ref_targets = val_targets[:split]
    calib_features = val_proj_sphere[split:]
    calib_targets = val_targets[split:]
    calib_base_probs = val_probs[split:]

    candidate_k = [k for k in [5, 10, 20, 30, 100] if k <= split]
    if not candidate_k and split > 0:
        candidate_k = [split]
    candidate_alpha = [0.0, 0.1, 0.25, 0.4, 0.55, 0.7, 0.8, 0.85, 1.0]

    best = None

    for k in candidate_k:
        knn = KNeighborsClassifier(n_neighbors=k, weights="distance")
        knn.fit(ref_features, ref_targets)
        calib_knn_probs = full_probs_from_knn(knn, calib_features, num_classes)

        for alpha in candidate_alpha:
            probs = (1.0 - alpha) * calib_base_probs + alpha * calib_knn_probs
            preds = np.argmax(probs, axis=1)
            confs = np.max(probs, axis=1)
                
            metrics = evaluate_sc_metrics(preds, confs, calib_targets)
            score = metrics["aurc"]
            if best is None or score < best["score"]:
                best = {"score": score, "k": k, "alpha": alpha}

    # Refit
    knn = KNeighborsClassifier(n_neighbors=best["k"], weights="distance")
    knn.fit(val_proj_sphere, val_targets)
    test_knn_probs = full_probs_from_knn(knn, test_proj_sphere, num_classes)
    test_probs_blend = (1.0 - best["alpha"]) * test_probs + best["alpha"] * test_knn_probs
    test_probs_blend = test_probs_blend / np.sum(test_probs_blend, axis=1, keepdims=True)
    
    return {
        "name": "Simplex-Orthogonal kNN Blend",
        "params": {"k": int(best["k"]), "alpha": float(best["alpha"])},
        "test_probs": test_probs_blend,
    }










def method_logit_zscore_blend(val_logits, val_probs, val_targets, test_logits, test_probs):
    """Centered Logit pNorm (Logit Z-Score) with alpha blending."""
    n = val_logits.shape[0]
    split = n // 2

    calib_logits = val_logits[split:]
    calib_targets = val_targets[split:]
    calib_base_probs = val_probs[split:]

    candidate_p = [1, 2, 4, 8]
    candidate_alpha = [0.0, 0.25, 0.5, 0.75, 1.0]

    best = None
    calib_preds = np.argmax(calib_base_probs, axis=1)
    
    for p in candidate_p:
        Z_mean = np.mean(calib_logits, axis=1, keepdims=True)
        Z_centered = calib_logits - Z_mean
        norm = np.linalg.norm(Z_centered, ord=p, axis=1)
        
        logit_conf = Z_centered[np.arange(calib_logits.shape[0]), calib_preds] / np.clip(norm, 1e-12, None)
        logit_conf = (logit_conf - np.min(logit_conf)) / (np.max(logit_conf) - np.min(logit_conf) + 1e-8)
        
        base_conf = np.max(calib_base_probs, axis=1)
        
        for alpha in candidate_alpha:
            confs = alpha * logit_conf + (1.0 - alpha) * base_conf
            metrics = evaluate_sc_metrics(calib_preds, confs, calib_targets)
            score = metrics["aurc"]
            if best is None or score < best["score"]:
                best = {"score": score, "p": p, "alpha": alpha}

    # Refit
    test_preds = np.argmax(test_probs, axis=1)
    Z_mean_test = np.mean(test_logits, axis=1, keepdims=True)
    Z_centered_test = test_logits - Z_mean_test
    norm_test = np.linalg.norm(Z_centered_test, ord=best["p"], axis=1)
    
    logit_conf_test = Z_centered_test[np.arange(test_logits.shape[0]), test_preds] / np.clip(norm_test, 1e-12, None)
    logit_conf_test = (logit_conf_test - np.min(logit_conf_test)) / (np.max(logit_conf_test) - np.min(logit_conf_test) + 1e-8)
    
    base_conf_test = np.max(test_probs, axis=1)
    test_conf_override = best["alpha"] * logit_conf_test + (1.0 - best["alpha"]) * base_conf_test
    
    return {
        "name": "Logit Z-Score Blend (Centered pNorm)",
        "params": {"p": int(best["p"]), "alpha": float(best["alpha"])},
        "test_probs": test_probs,
        "test_conf_override": test_conf_override,
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


def _sirc_vim_dim(feature_dim: int) -> int:
    """SIRC repo heuristic for the ViM null-space size."""
    dim = 1000 if feature_dim > 1500 else 512
    if feature_dim < 512:
        dim = feature_dim // 2
    return dim


def method_vim_and_sirc(train_features, train_logits, test_features, test_logits, test_probs, model):
    """ViM and SIRC using training-set parameters, following the SIRC repo."""
    w, b = get_final_layer_params(model)
    if w is None:
        return None, None

    # 1. Compute ViM parameters on the training set.
    u = -torch.linalg.pinv(w) @ b
    centered_train = train_features - u.unsqueeze(0)
    
    vim_dim = _sirc_vim_dim(centered_train.shape[-1])
    if vim_dim >= centered_train.shape[-1]:
        return None, None

    eig_vecs = torch.linalg.eigh(centered_train.T @ centered_train).eigenvectors.flip(-1)
    NS = eig_vecs[:, vim_dim:]
    
    # alpha scaling
    vlogit_train = torch.norm(centered_train @ NS, p=2, dim=-1)
    max_logit_train = train_logits.max(dim=1).values
    alpha = max_logit_train.mean() / vlogit_train.mean()
    
    # 2. Compute ViM scores on test set
    centered_test = test_features - u.unsqueeze(0)
    vlogit_test = torch.norm(centered_test @ NS, p=2, dim=-1) * alpha
    energy_test = torch.logsumexp(test_logits, dim=1)
    vim_confs = energy_test - vlogit_test
    
    # 3. SIRC_MSP_res: use training residual stats and combine MSP with -vlogit.
    vlogit_train_scaled = vlogit_train * alpha
    train_residual = vlogit_train_scaled.detach().cpu().numpy()
    a = -np.mean(train_residual) - 3 * np.std(train_residual)
    b = 1.0 / (np.std(train_residual) + 1e-12)
    
    msp = np.max(test_probs, axis=1)
    msp = np.clip(msp, 0.0, 1.0 - 1e-12) 
    soft = np.log(1.0 - msp)
    
    vim_res_test = -vlogit_test.detach().cpu().numpy()
    additional = np.logaddexp(0.0, -b * (vim_res_test - a))
    sirc_confs = -soft - additional
    
    vim_res = {
        "name": "ViM (Virtual Logit Matching)",
        "params": {"vim_dim": int(vim_dim), "param_source": "train"},
        "test_probs": test_probs,
        "test_conf_override": vim_confs.detach().cpu().numpy(),
    }
    
    sirc_res = {
        "name": "SIRC (MSP + ViM-Res)",
        "params": {"a": float(a), "b": float(b), "vim_dim": int(vim_dim), "param_source": "train"},
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


def compute_mahalanobis_confidence(test_features: np.ndarray, val_features: np.ndarray, val_targets: np.ndarray, num_classes: int) -> np.ndarray:
    """Original Mahalanobis on features (Lee et al. 2018)."""
    dim = val_features.shape[1]
    means_by_class = np.zeros((num_classes, dim), dtype=np.float64)
    
    # 1. Compute class means
    for c in range(num_classes):
        idx = np.where(val_targets == c)[0]
        if len(idx) > 0:
            means_by_class[c] = np.mean(val_features[idx], axis=0)
            
    # 2. Compute shared covariance matrix
    centered_features = []
    for c in range(num_classes):
        idx = np.where(val_targets == c)[0]
        if len(idx) > 0:
            centered_features.append(val_features[idx] - means_by_class[c])
            
    X = np.concatenate(centered_features, axis=0)
    
    cov_estimator = EmpiricalCovariance(assume_centered=True)
    cov_estimator.fit(X)
    cov = cov_estimator.covariance_
    
    # Add a small epsilon for numerical stability before inversion
    precision = np.linalg.inv(cov + 1e-6 * np.eye(dim, dtype=np.float64))
    
    # 3. Compute Mahalanobis distance
    diffs = test_features[:, None, :] - means_by_class[None, :, :]
    d2 = np.einsum("ncd,df,ncf->nc", diffs, precision, diffs)
    
    # Score is the maximum negative distance (or minimum distance)
    scores = -0.5 * d2
    test_confs = np.max(scores, axis=1)
    
    return test_confs


def method_mahalanobis(val_features, val_targets, test_features, test_probs, num_classes):
    """Mahalanobis confidence following original feature-space computation."""
    test_confs = compute_mahalanobis_confidence(test_features, val_features, val_targets, num_classes)
    
    return {
        "name": "Mahalanobis Distance",
        "params": {},
        "test_probs": test_probs,
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


def _pnorm_np(logits: np.ndarray, p: int, eps: float = 1e-12) -> np.ndarray:
    if p == 0:
        denom = np.count_nonzero(logits, axis=1, keepdims=True).astype(np.float64)
        return np.clip(denom, 1.0, None)
    denom = np.linalg.norm(logits, ord=p, axis=1, keepdims=True)
    return np.clip(denom, eps, None)


def _softmax_np(logits: np.ndarray) -> np.ndarray:
    shift = np.max(logits, axis=1, keepdims=True)
    probs = np.exp(logits - shift)
    return probs / np.sum(probs, axis=1, keepdims=True)


def _msp_np(logits: np.ndarray) -> np.ndarray:
    return np.max(_softmax_np(logits), axis=1)


def method_5_maxlogit_pnorm(val_logits, val_targets, test_logits):
    """FixSelective MaxLogit-pNorm with p-grid optimization and MSP fallback."""
    val_logits_np = val_logits.detach().cpu().numpy() if isinstance(val_logits, torch.Tensor) else val_logits
    test_logits_np = test_logits.detach().cpu().numpy() if isinstance(test_logits, torch.Tensor) else test_logits
    val_targets_np = val_targets.detach().cpu().numpy() if isinstance(val_targets, torch.Tensor) else val_targets

    val_centered = _centralize_np(val_logits_np)
    test_centered = _centralize_np(test_logits_np)
    val_preds = np.argmax(val_logits_np, axis=1)

    # MSP fallback used in FixSelective optimize.p; original selection minimizes AURC.
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
        "test_probs": _softmax_np(test_logits_np),
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
        val_norm = _pnorm_np(val_centered, p=p)
        for t in t_range:
            # Follow FixSelective default: rescale_T=True
            t_eff = t / float(np.mean(val_norm))
            val_scaled = val_centered / (t_eff * val_norm)
            val_conf = _msp_np(val_scaled)
            score = evaluate_sc_metrics(val_preds, val_conf, val_targets_np)["aurc"]
            if best is None or score < best["score"]:
                best = {"score": score, "p": p, "temperature": float(t_eff)}

    p_best = int(best["p"])
    test_norm = _pnorm_np(test_centered, p=p_best)
    test_scaled = test_centered / (best["temperature"] * test_norm)
    test_conf = _msp_np(test_scaled)

    return {
        "name": "MaxLogit pNorm+",
        "params": {"p": p_best, "temperature": float(best["temperature"])},
        "test_probs": _softmax_np(test_logits_np),
        "test_conf_override": test_conf,
    }

def method_residual_mllr_logistic(
    val_logits,
    val_targets,
    test_logits,
    test_probs,
    prob_temperature: float = 1.0,
):
    """
    Residual-MLLR Logistic Stacking.

    Safer than Delta-MLLR:
    learns P(correct | pNorm, margin, tail_leakage, endpointness)
    on validation data.

    If extra features are useless, logistic regression can mostly rely on pNorm.
    """
    print("Evaluating Residual-MLLR Logistic...")

    val_logits_np = val_logits.detach().cpu().numpy() if isinstance(val_logits, torch.Tensor) else np.asarray(val_logits)
    test_logits_np = test_logits.detach().cpu().numpy() if isinstance(test_logits, torch.Tensor) else np.asarray(test_logits)
    val_targets_np = val_targets.detach().cpu().numpy() if isinstance(val_targets, torch.Tensor) else np.asarray(val_targets)
    test_probs_np = test_probs.detach().cpu().numpy() if isinstance(test_probs, torch.Tensor) else np.asarray(test_probs)

    val_preds = np.argmax(val_logits_np, axis=1)
    val_correct = (val_preds == val_targets_np).astype(int)

    n = val_logits_np.shape[0]
    split = n // 2

    # Select p using first half only.
    p_choice = _find_best_p(val_logits_np[:split], val_targets_np[:split])

    # Reuse the MLLR statistic function if you pasted Delta-MLLR earlier.
    val_m = _compute_mllr_stats(
        val_logits_np,
        p_choice=p_choice,
        prob_temperature=prob_temperature,
    )
    test_m = _compute_mllr_stats(
        test_logits_np,
        p_choice=p_choice,
        prob_temperature=prob_temperature,
    )

    # Train logistic model on second half, standardizing using second half only.
    train_m = val_m[split:]
    train_y = val_correct[split:]

    mean = np.mean(train_m, axis=0, keepdims=True)
    std = np.std(train_m, axis=0, keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)

    train_x = (train_m - mean) / std
    test_x = (test_m - mean) / std

    # Try different regularization strengths.
    best = None
    for C in [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0]:
        clf = LogisticRegression(
            C=C,
            penalty="l2",
            solver="lbfgs",
            class_weight="balanced",
            max_iter=1000,
        )
        clf.fit(train_x, train_y)

        # Evaluate on first half as a simple cross-check.
        eval_m = val_m[:split]
        eval_y = val_correct[:split]
        eval_targets = val_targets_np[:split]
        eval_preds = val_preds[:split]

        eval_x = (eval_m - mean) / std
        eval_conf = clf.predict_proba(eval_x)[:, 1]

        metrics = evaluate_sc_metrics(eval_preds, eval_conf, eval_targets)
        score = metrics["aurc"]

        if best is None or score < best["score"]:
            best = {
                "score": float(score),
                "C": float(C),
                "model": clf,
                "auroc": float(metrics["auroc"]),
                "naurc": float(metrics["naurc"]),
            }

    test_conf = best["model"].predict_proba(test_x)[:, 1].astype(np.float32)

    return {
        "name": "Residual-MLLR Logistic",
        "params": {
            "p": p_choice if isinstance(p_choice, str) else int(p_choice),
            "C": best["C"],
            "stat_order": ["pNorm", "margin", "tail_leakage", "endpointness"],
            "valcheck_aurc": best["score"],
            "valcheck_auroc": best["auroc"],
            "valcheck_naurc": best["naurc"],
        },
        "test_probs": test_probs_np,
        "test_conf_override": test_conf,
    }

# ======================== Delta-RNLR: Row-Null Likelihood Ratio ========================

def _to_numpy_safe(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _pnorm_conf_np(logits_np: np.ndarray, p_choice):
    if p_choice == "MSP":
        return _msp_np(logits_np).astype(np.float32)
    centered = _centralize_np(logits_np)
    normed = _normalize_np(centered, p=int(p_choice))
    return np.max(normed, axis=1).astype(np.float32)


def _select_best_pnorm_p_aurc(val_logits_np: np.ndarray, val_targets_np: np.ndarray):
    val_preds = np.argmax(val_logits_np, axis=1)

    best_p = "MSP"
    best_score = evaluate_sc_metrics(
        val_preds,
        _msp_np(val_logits_np),
        val_targets_np,
    )["aurc"]

    for p in range(10):
        conf = _pnorm_conf_np(val_logits_np, p)
        score = evaluate_sc_metrics(val_preds, conf, val_targets_np)["aurc"]
        if score < best_score:
            best_score = score
            best_p = p

    return best_p, float(best_score)


def _row_space_projection_matrix(W_np: np.ndarray, eps: float = 1e-6):
    """
    W: [C, d]
    Returns P = W^T (W W^T)^dagger W, projection onto row(W).
    """
    W_np = W_np.astype(np.float64)
    WWt = W_np @ W_np.T
    P = W_np.T @ np.linalg.pinv(WWt + eps * np.eye(WWt.shape[0])) @ W_np
    return P.astype(np.float32)


def _decompose_row_null(features_np: np.ndarray, W_np: np.ndarray):
    """
    Decompose z into row-space and null-space components.
    """
    P = _row_space_projection_matrix(W_np)
    z_parallel = features_np @ P.T
    z_null = features_np - z_parallel
    return z_parallel.astype(np.float32), z_null.astype(np.float32)


def _standardize_fit(x: np.ndarray, eps: float = 1e-6):
    mean = np.mean(x, axis=0, keepdims=True)
    std = np.std(x, axis=0, keepdims=True)
    std = np.where(std < eps, 1.0, std)
    return mean.astype(np.float32), std.astype(np.float32)


def _standardize_apply(x: np.ndarray, mean: np.ndarray, std: np.ndarray):
    return ((x - mean) / std).astype(np.float32)


def _knn_density_proxy(query: np.ndarray, ref: np.ndarray, k: int, eps: float = 1e-8):
    """
    D_k(q, R) = - mean log(eps + distance to k nearest neighbors)
    Higher = denser / closer.
    """
    if ref.shape[0] == 0:
        return np.full(query.shape[0], -1e6, dtype=np.float32)

    k_eff = min(int(k), ref.shape[0])
    nn = NearestNeighbors(n_neighbors=k_eff, metric="euclidean")
    nn.fit(ref)
    dist, _ = nn.kneighbors(query)
    return -np.mean(np.log(eps + dist), axis=1).astype(np.float32)


def _null_delta_knn_score(
    query_null: np.ndarray,
    ref_null: np.ndarray,
    ref_correct: np.ndarray,
    k: int,
):
    """
    Correct-vs-wrong density score in null space.
    """
    zc = ref_null[ref_correct]
    zw = ref_null[~ref_correct]

    if len(zc) == 0 or len(zw) == 0:
        return np.zeros(query_null.shape[0], dtype=np.float32)

    k_eff = min(k, len(zc), len(zw))
    return (
        _knn_density_proxy(query_null, zc, k_eff)
        - _knn_density_proxy(query_null, zw, k_eff)
    ).astype(np.float32)


def _compute_rnlr_stats(
    logits_np: np.ndarray,
    null_features_np: np.ndarray,
    ref_null_np: np.ndarray,
    ref_correct: np.ndarray,
    p_choice,
    k_null: int,
):
    """
    Statistic vector:
        [pNorm, margin, null_norm, null_delta_knn]
    """
    logits_np = logits_np.astype(np.float64)

    pnorm_conf = _pnorm_conf_np(logits_np, p_choice)

    top2 = np.partition(logits_np, kth=-2, axis=1)[:, -2:]
    top2 = np.sort(top2, axis=1)[:, ::-1]
    margin = (top2[:, 0] - top2[:, 1]).astype(np.float32)

    null_norm = np.linalg.norm(null_features_np, axis=1).astype(np.float32)

    null_delta = _null_delta_knn_score(
        query_null=null_features_np,
        ref_null=ref_null_np,
        ref_correct=ref_correct,
        k=k_null,
    )

    stats = np.stack(
        [
            pnorm_conf.astype(np.float32),
            margin.astype(np.float32),
            null_norm.astype(np.float32),
            null_delta.astype(np.float32),
        ],
        axis=1,
    )

    return np.nan_to_num(stats, nan=0.0, posinf=1e6, neginf=-1e6).astype(np.float32)


def method_delta_rnlr_legacy(
    val_features,
    val_logits,
    val_targets,
    test_features,
    test_logits,
    test_probs,
    model,
    candidate_k_density=(3, 5, 10, 20, 30, 50, 75, 100),
    candidate_k_null=(3, 5, 10, 20, 30, 50),
):
    """
    Delta-RNLR: Row-Null Likelihood Ratio.

    Tests whether information discarded by the classifier head W helps selective classification.

    Final score:
        D_k(m(x), M_correct) - D_k(m(x), M_wrong)

    m(x):
        [
          pNorm(logits),
          top2 logit margin,
          ||z_null||,
          delta-kNN-density(z_null)
        ]
    """
    print("Evaluating Delta-RNLR...")

    W, b = get_final_layer_params(model)
    if W is None:
        print("  Delta-RNLR skipped: could not find final linear classifier weights.")
        return None

    W_np = W.detach().cpu().numpy()

    val_features_np = _to_numpy_safe(val_features).astype(np.float32)
    test_features_np = _to_numpy_safe(test_features).astype(np.float32)
    val_logits_np = _to_numpy_safe(val_logits).astype(np.float64)
    test_logits_np = _to_numpy_safe(test_logits).astype(np.float64)
    val_targets_np = _to_numpy_safe(val_targets).astype(int)
    test_probs_np = _to_numpy_safe(test_probs).astype(np.float32)

    val_preds = np.argmax(val_logits_np, axis=1)
    val_correct = val_preds == val_targets_np

    n = val_features_np.shape[0]
    split = n // 2

    if split < 5:
        print("  Delta-RNLR fallback: validation set too small.")
        p_choice, _ = _select_best_pnorm_p_aurc(val_logits_np, val_targets_np)
        test_conf = _pnorm_conf_np(test_logits_np, p_choice)
        return {
            "name": "Delta-RNLR (fallback pNorm)",
            "params": {"p": p_choice, "reason": "small_validation"},
            "test_probs": test_probs_np,
            "test_conf_override": test_conf,
        }

    # 1. Row/null decomposition.
    _, val_null = _decompose_row_null(val_features_np, W_np)
    _, test_null = _decompose_row_null(test_features_np, W_np)

    # Optional but recommended: normalize null vectors for kNN stability.
    # This keeps null direction/relative structure while reducing norm scale domination.
    null_mean = np.mean(val_null[:split], axis=0, keepdims=True)
    val_null_centered = val_null - null_mean
    test_null_centered = test_null - null_mean

    # 2. Reference/calibration split.
    ref_logits = val_logits_np[:split]
    ref_targets = val_targets_np[:split]
    ref_correct = val_correct[:split]

    calib_logits = val_logits_np[split:]
    calib_targets = val_targets_np[split:]
    calib_preds = val_preds[split:]

    ref_null = val_null_centered[:split]
    calib_null = val_null_centered[split:]

    if np.sum(ref_correct) == 0 or np.sum(~ref_correct) == 0:
        print("  Delta-RNLR fallback: reference split lacks correct or wrong samples.")
        p_choice, _ = _select_best_pnorm_p_aurc(val_logits_np, val_targets_np)
        test_conf = _pnorm_conf_np(test_logits_np, p_choice)
        return {
            "name": "Delta-RNLR (fallback pNorm)",
            "params": {"p": p_choice, "reason": "missing_correct_or_wrong"},
            "test_probs": test_probs_np,
            "test_conf_override": test_conf,
        }

    # 3. Choose pNorm p using reference split only.
    p_choice, p_ref_aurc = _select_best_pnorm_p_aurc(ref_logits, ref_targets)

    best = None

    for k_null in candidate_k_null:
        k_null_eff = min(int(k_null), np.sum(ref_correct), np.sum(~ref_correct))
        if k_null_eff < 1:
            continue

        ref_m_raw = _compute_rnlr_stats(
            logits_np=ref_logits,
            null_features_np=ref_null,
            ref_null_np=ref_null,
            ref_correct=ref_correct,
            p_choice=p_choice,
            k_null=k_null_eff,
        )

        calib_m_raw = _compute_rnlr_stats(
            logits_np=calib_logits,
            null_features_np=calib_null,
            ref_null_np=ref_null,
            ref_correct=ref_correct,
            p_choice=p_choice,
            k_null=k_null_eff,
        )

        mean, std = _standardize_fit(ref_m_raw)
        ref_m = _standardize_apply(ref_m_raw, mean, std)
        calib_m = _standardize_apply(calib_m_raw, mean, std)

        m_correct = ref_m[ref_correct]
        m_wrong = ref_m[~ref_correct]

        max_k_density = min(len(m_correct), len(m_wrong))
        k_density_grid = [k for k in candidate_k_density if k <= max_k_density]
        if not k_density_grid:
            k_density_grid = [1]

        for k_density in k_density_grid:
            calib_conf = (
                _knn_density_proxy(calib_m, m_correct, k_density)
                - _knn_density_proxy(calib_m, m_wrong, k_density)
            )

            metrics = evaluate_sc_metrics(calib_preds, calib_conf, calib_targets)
            score = metrics["aurc"]

            if best is None or score < best["score"]:
                best = {
                    "score": float(score),
                    "k_null": int(k_null_eff),
                    "k_density": int(k_density),
                    "calib_auroc": float(metrics["auroc"]),
                    "calib_naurc": float(metrics["naurc"]),
                    "p": p_choice,
                }

    if best is None:
        print("  Delta-RNLR fallback: no valid hyperparameter setting.")
        p_choice, _ = _select_best_pnorm_p_aurc(val_logits_np, val_targets_np)
        test_conf = _pnorm_conf_np(test_logits_np, p_choice)
        return {
            "name": "Delta-RNLR (fallback pNorm)",
            "params": {"p": p_choice, "reason": "no_valid_hyperparams"},
            "test_probs": test_probs_np,
            "test_conf_override": test_conf,
        }

    # 4. Refit on full validation set with selected hyperparameters.
    #    Re-select p on full validation for the final pNorm coordinate.
    p_final, p_full_aurc = _select_best_pnorm_p_aurc(val_logits_np, val_targets_np)

    full_correct = val_correct
    full_null = val_null - np.mean(val_null, axis=0, keepdims=True)
    test_null_final = test_null - np.mean(val_null, axis=0, keepdims=True)

    if np.sum(full_correct) == 0 or np.sum(~full_correct) == 0:
        print("  Delta-RNLR fallback: full validation lacks correct or wrong samples.")
        test_conf = _pnorm_conf_np(test_logits_np, p_final)
        return {
            "name": "Delta-RNLR (fallback pNorm)",
            "params": {"p": p_final, "reason": "missing_correct_or_wrong_full"},
            "test_probs": test_probs_np,
            "test_conf_override": test_conf,
        }

    k_null_final = min(best["k_null"], np.sum(full_correct), np.sum(~full_correct))

    val_m_raw = _compute_rnlr_stats(
        logits_np=val_logits_np,
        null_features_np=full_null,
        ref_null_np=full_null,
        ref_correct=full_correct,
        p_choice=p_final,
        k_null=k_null_final,
    )

    test_m_raw = _compute_rnlr_stats(
        logits_np=test_logits_np,
        null_features_np=test_null_final,
        ref_null_np=full_null,
        ref_correct=full_correct,
        p_choice=p_final,
        k_null=k_null_final,
    )

    mean, std = _standardize_fit(val_m_raw)
    val_m = _standardize_apply(val_m_raw, mean, std)
    test_m = _standardize_apply(test_m_raw, mean, std)

    m_correct = val_m[full_correct]
    m_wrong = val_m[~full_correct]

    k_density_final = min(best["k_density"], len(m_correct), len(m_wrong))

    test_conf = (
        _knn_density_proxy(test_m, m_correct, k_density_final)
        - _knn_density_proxy(test_m, m_wrong, k_density_final)
    )

    return {
        "name": "Delta-RNLR (Row-Null Likelihood Ratio)",
        "params": {
            "p": p_final if isinstance(p_final, str) else int(p_final),
            "p_ref": best["p"] if isinstance(best["p"], str) else int(best["p"]),
            "k_null": int(k_null_final),
            "k_density": int(k_density_final),
            "stat_order": ["pNorm", "margin", "null_norm", "null_delta_knn"],
            "calib_aurc": float(best["score"]),
            "calib_auroc": float(best["calib_auroc"]),
            "calib_naurc": float(best["calib_naurc"]),
        },
        "test_probs": test_probs_np,
        "test_conf_override": test_conf.astype(np.float32),
    }
# ======================== Delta-MLLR: Mixup Logit Likelihood Ratio ========================

def _to_numpy(x):
    """Convert torch.Tensor or array-like to np.ndarray."""
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _softmax_np(logits: np.ndarray, temperature: float = 1.0, eps: float = 1e-12) -> np.ndarray:
    z = logits / float(temperature)
    z = z - np.max(z, axis=1, keepdims=True)
    p = np.exp(z)
    return p / np.clip(np.sum(p, axis=1, keepdims=True), eps, None)


def _pnorm_conf_from_logits(logits_np: np.ndarray, p_choice):
    """
    Return S_pNorm coordinate.
    If p_choice == 'MSP', use MSP fallback.
    Otherwise use max centered-p-normalized logit.
    """
    if p_choice == "MSP":
        return _msp_np(logits_np).astype(np.float32)

    centered = _centralize_np(logits_np)
    normed = _normalize_np(centered, p=int(p_choice))
    return np.max(normed, axis=1).astype(np.float32)


def _select_best_pnorm_p(
    val_logits_np: np.ndarray,
    val_targets_np: np.ndarray,
    metric: str = "aurc",
):
    """
    Select p for the pNorm coordinate using validation labels only.
    This mirrors your existing pNorm logic but optimizes AURC by default.
    """
    val_preds = np.argmax(val_logits_np, axis=1)

    best_p = "MSP"
    best_score = evaluate_sc_metrics(
        val_preds,
        _msp_np(val_logits_np),
        val_targets_np,
    )[metric]

    for p in range(10):
        conf = _pnorm_conf_from_logits(val_logits_np, p)
        score = evaluate_sc_metrics(val_preds, conf, val_targets_np)[metric]

        if metric in ["aurc", "eaurc", "naurc"]:
            better = score < best_score
        else:
            better = score > best_score

        if better:
            best_score = score
            best_p = p

    return best_p, float(best_score)


def _compute_mllr_stats(
    logits,
    p_choice,
    prob_temperature: float = 1.0,
    eps: float = 1e-12,
):
    """
    Compute 4D Mixup-aware logit statistic:

        m(x) = [
            S_pNorm(x),
            top1_logit - top2_logit,
            log((1 - p1 - p2 + eps) / (p1 + p2 + eps)),
            -log(4 lambda (1 - lambda) + eps)
        ]

    where lambda = p1 / (p1 + p2).

    Shape:
        logits: [N, C]
        return: [N, 4]
    """
    logits_np = _to_numpy(logits).astype(np.float64)
    probs = _softmax_np(logits_np, temperature=prob_temperature, eps=eps)

    # Top-2 by logits. This is equivalent to top-2 by softmax probability.
    top2_idx = np.argpartition(logits_np, kth=-2, axis=1)[:, -2:]
    top2_logits_unsorted = np.take_along_axis(logits_np, top2_idx, axis=1)
    order = np.argsort(top2_logits_unsorted, axis=1)[:, ::-1]
    top2_idx = np.take_along_axis(top2_idx, order, axis=1)

    rows = np.arange(logits_np.shape[0])
    i1 = top2_idx[:, 0]
    i2 = top2_idx[:, 1]

    l1 = logits_np[rows, i1]
    l2 = logits_np[rows, i2]
    margin = l1 - l2

    p1 = probs[rows, i1]
    p2 = probs[rows, i2]

    m2 = np.clip(p1 + p2, eps, 1.0 - eps)
    tail_leakage = np.log((1.0 - m2 + eps) / (m2 + eps))

    lam = np.clip(p1 / np.clip(m2, eps, None), eps, 1.0 - eps)
    endpointness = -np.log(4.0 * lam * (1.0 - lam) + eps)

    pnorm_conf = _pnorm_conf_from_logits(logits_np, p_choice)

    stats = np.stack(
        [
            pnorm_conf,
            margin.astype(np.float32),
            tail_leakage.astype(np.float32),
            endpointness.astype(np.float32),
        ],
        axis=1,
    ).astype(np.float32)

    stats = np.nan_to_num(stats, nan=0.0, posinf=1e6, neginf=-1e6)
    return stats


def _fit_standardizer(ref_m: np.ndarray, eps: float = 1e-6):
    mean = np.mean(ref_m, axis=0, keepdims=True)
    std = np.std(ref_m, axis=0, keepdims=True)
    std = np.where(std < eps, 1.0, std)
    return mean.astype(np.float32), std.astype(np.float32)


def _apply_standardizer(m: np.ndarray, mean: np.ndarray, std: np.ndarray):
    return ((m - mean) / std).astype(np.float32)


def _knn_log_density_score(query_m: np.ndarray, ref_m: np.ndarray, k: int, eps: float = 1e-8):
    """
    kNN log-density proxy:
        D_k(m, M) = - mean_i log(eps + distance(m, m_i))
    Higher means denser / closer.
    """
    if ref_m.shape[0] == 0:
        return np.full(query_m.shape[0], -1e6, dtype=np.float32)

    k_eff = min(int(k), ref_m.shape[0])
    nn = NearestNeighbors(n_neighbors=k_eff, metric="euclidean")
    nn.fit(ref_m)

    dist, _ = nn.kneighbors(query_m)
    return -np.mean(np.log(eps + dist), axis=1).astype(np.float32)


def method_delta_mllr(
    val_logits,
    val_targets,
    test_logits,
    test_probs,
    candidate_k=(3, 5, 10, 20, 30, 50, 75, 100),
    prob_temperature: float = 1.0,
):
    """
    Delta-MLLR: Mixup Logit Likelihood Ratio.

    Logit-only NP-style selector:
        s(x) = D_k(m(x), M_correct) - D_k(m(x), M_wrong)

    where m(x) contains:
        1. pNorm confidence
        2. top-2 logit margin
        3. simplex tail leakage
        4. Mixup endpointness

    This is designed to keep pNorm's strength while adding Mixup-specific
    simplex geometry.
    """
    print("Evaluating Delta-MLLR...")

    val_logits_np = _to_numpy(val_logits).astype(np.float64)
    test_logits_np = _to_numpy(test_logits).astype(np.float64)
    val_targets_np = _to_numpy(val_targets).astype(int)
    test_probs_np = _to_numpy(test_probs).astype(np.float32)

    val_preds = np.argmax(val_logits_np, axis=1)
    val_correct = val_preds == val_targets_np

    n = val_logits_np.shape[0]
    split = n // 2

    # 1. Select pNorm coordinate using only the reference half.
    ref_logits_np = val_logits_np[:split]
    ref_targets_np = val_targets_np[:split]
    p_choice, p_score = _select_best_pnorm_p(
        ref_logits_np,
        ref_targets_np,
        metric="aurc",
    )

    # 2. Compute MLLR statistics for all val/test samples.
    val_m = _compute_mllr_stats(
        val_logits_np,
        p_choice=p_choice,
        prob_temperature=prob_temperature,
    )
    test_m = _compute_mllr_stats(
        test_logits_np,
        p_choice=p_choice,
        prob_temperature=prob_temperature,
    )

    # 3. Split val into reference and calibration for k selection.
    ref_m_raw = val_m[:split]
    calib_m_raw = val_m[split:]

    ref_correct = val_correct[:split]
    calib_targets = val_targets_np[split:]
    calib_preds = val_preds[split:]

    if np.sum(ref_correct) == 0 or np.sum(~ref_correct) == 0:
        print("  Delta-MLLR fallback: reference split lacks both correct and wrong samples.")
        test_conf = _pnorm_conf_from_logits(test_logits_np, p_choice)
        return {
            "name": "Delta-MLLR (fallback pNorm)",
            "params": {"p": p_choice, "reason": "missing_correct_or_wrong_ref_samples"},
            "test_probs": test_probs_np,
            "test_conf_override": test_conf,
        }

    # 4. Standardize using the reference split only.
    mean, std = _fit_standardizer(ref_m_raw)
    ref_m = _apply_standardizer(ref_m_raw, mean, std)
    calib_m = _apply_standardizer(calib_m_raw, mean, std)

    m_correct = ref_m[ref_correct]
    m_wrong = ref_m[~ref_correct]

    # 5. Tune density k on calibration AURC.
    max_k = min(len(m_correct), len(m_wrong))
    k_grid = [k for k in candidate_k if k <= max_k]
    if not k_grid:
        k_grid = [1]

    best = None
    for k in k_grid:
        calib_conf = (
            _knn_log_density_score(calib_m, m_correct, k)
            - _knn_log_density_score(calib_m, m_wrong, k)
        )

        metrics = evaluate_sc_metrics(calib_preds, calib_conf, calib_targets)
        score = metrics["aurc"]

        if best is None or score < best["score"]:
            best = {
                "score": float(score),
                "k_density": int(k),
                "calib_auroc": float(metrics["auroc"]),
                "calib_naurc": float(metrics["naurc"]),
            }

    # 6. Refit correct/wrong banks on full validation set.
    #    This is okay because p and k are already selected without test labels.
    p_choice_final, p_score_final = _select_best_pnorm_p(
        val_logits_np,
        val_targets_np,
        metric="aurc",
    )

    val_m_final = _compute_mllr_stats(
        val_logits_np,
        p_choice=p_choice_final,
        prob_temperature=prob_temperature,
    )
    test_m_final = _compute_mllr_stats(
        test_logits_np,
        p_choice=p_choice_final,
        prob_temperature=prob_temperature,
    )

    mean_final, std_final = _fit_standardizer(val_m_final)
    val_m_std = _apply_standardizer(val_m_final, mean_final, std_final)
    test_m_std = _apply_standardizer(test_m_final, mean_final, std_final)

    m_correct_final = val_m_std[val_correct]
    m_wrong_final = val_m_std[~val_correct]

    if np.sum(val_correct) == 0 or np.sum(~val_correct) == 0:
        print("  Delta-MLLR fallback: full validation lacks both correct and wrong samples.")
        test_conf = _pnorm_conf_from_logits(test_logits_np, p_choice_final)
        return {
            "name": "Delta-MLLR (fallback pNorm)",
            "params": {"p": p_choice_final, "reason": "missing_correct_or_wrong_full_val_samples"},
            "test_probs": test_probs_np,
            "test_conf_override": test_conf,
        }

    k_final = min(best["k_density"], len(m_correct_final), len(m_wrong_final))

    test_conf = (
        _knn_log_density_score(test_m_std, m_correct_final, k_final)
        - _knn_log_density_score(test_m_std, m_wrong_final, k_final)
    )

    return {
        "name": "Delta-MLLR (Mixup Logit Likelihood Ratio)",
        "params": {
            "p": p_choice_final if isinstance(p_choice_final, str) else int(p_choice_final),
            "p_ref": p_choice if isinstance(p_choice, str) else int(p_choice),
            "k_density": int(k_final),
            "prob_temperature": float(prob_temperature),
            "stat_order": ["pNorm", "margin", "tail_leakage", "endpointness"],
            "calib_aurc": float(best["score"]),
            "calib_auroc": float(best["calib_auroc"]),
            "calib_naurc": float(best["calib_naurc"]),
        },
        "test_probs": test_probs_np,
        "test_conf_override": test_conf,
    }

# ======================== pNorm Calibration Fixes ========================

def _find_best_p(val_logits_np, val_targets_np):
    """Find the optimal p for pNorm on the validation set. Shared by all fixes."""
    val_centered = _centralize_np(val_logits_np)
    val_preds = np.argmax(val_logits_np, axis=1)

    best_p = "MSP"
    best_score = evaluate_sc_metrics(val_preds, _msp_np(val_logits_np), val_targets_np)["aurc"]

    for p in range(10):
        val_conf = np.max(_normalize_np(val_centered, p=p), axis=1)
        score = evaluate_sc_metrics(val_preds, val_conf, val_targets_np)["aurc"]
        if score < best_score:
            best_score = score
            best_p = p

    return best_p


def _get_pnorm_logits(logits_np, p):
    """Apply centering + p-normalization to logits."""
    centered = _centralize_np(logits_np)
    return _normalize_np(centered, p=int(p))


def method_pnorm_softmax(val_logits, val_targets, test_logits):
    """
    Fix A: Softmax on pNorm-normalized logits.
    
    Idea: After centering and p-normalizing, the logits live on a unit sphere.
    Applying softmax converts them back to valid probabilities while preserving
    the argmax (and thus the ranking of the predicted class). We additionally
    search for an optimal temperature T on the val set to control sharpness.
    """
    val_logits_np = val_logits.detach().cpu().numpy() if isinstance(val_logits, torch.Tensor) else val_logits
    test_logits_np = test_logits.detach().cpu().numpy() if isinstance(test_logits, torch.Tensor) else test_logits
    val_targets_np = val_targets.detach().cpu().numpy() if isinstance(val_targets, torch.Tensor) else val_targets

    best_p = _find_best_p(val_logits_np, val_targets_np)
    if best_p == "MSP":
        # If MSP was best, just return standard softmax
        test_probs = np.exp(test_logits_np - np.max(test_logits_np, axis=1, keepdims=True))
        test_probs /= np.sum(test_probs, axis=1, keepdims=True)
        return {
            "name": "pNorm-Softmax (Fix A)",
            "params": {"p": "MSP", "T": 1.0},
            "test_probs": test_probs,
        }

    val_pnorm = _get_pnorm_logits(val_logits_np, best_p)
    test_pnorm = _get_pnorm_logits(test_logits_np, best_p)
    val_preds = np.argmax(val_logits_np, axis=1)

    # Grid search for temperature on val set (optimize AURC)
    best_T = 1.0
    best_score = float("inf")
    for T in [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0]:
        scaled = val_pnorm / T
        probs = np.exp(scaled - np.max(scaled, axis=1, keepdims=True))
        probs /= np.sum(probs, axis=1, keepdims=True)
        confs = np.max(probs, axis=1)
        score = evaluate_sc_metrics(val_preds, confs, val_targets_np)["aurc"]
        if score < best_score:
            best_score = score
            best_T = T

    test_scaled = test_pnorm / best_T
    test_probs = np.exp(test_scaled - np.max(test_scaled, axis=1, keepdims=True))
    test_probs /= np.sum(test_probs, axis=1, keepdims=True)

    return {
        "name": "pNorm-Softmax (Fix A)",
        "params": {"p": int(best_p), "T": float(best_T)},
        "test_probs": test_probs,
    }


def method_pnorm_platt(val_logits, val_targets, test_logits):
    """
    Fix B: Platt scaling on pNorm confidence scores.
    
    Idea: pNorm produces excellent ranking but the raw max-of-normalized-logit
    values are not calibrated. Platt scaling learns a logistic mapping
    sigma(a*s + b) from the scalar pNorm confidence to calibrated probability.
    This is guaranteed to preserve ranking (since sigma is monotonic).
    """
    val_logits_np = val_logits.detach().cpu().numpy() if isinstance(val_logits, torch.Tensor) else val_logits
    test_logits_np = test_logits.detach().cpu().numpy() if isinstance(test_logits, torch.Tensor) else test_logits
    val_targets_np = val_targets.detach().cpu().numpy() if isinstance(val_targets, torch.Tensor) else val_targets

    best_p = _find_best_p(val_logits_np, val_targets_np)
    if best_p == "MSP":
        val_conf = _msp_np(val_logits_np)
        test_conf = _msp_np(test_logits_np)
    else:
        val_conf = np.max(_get_pnorm_logits(val_logits_np, best_p), axis=1)
        test_conf = np.max(_get_pnorm_logits(test_logits_np, best_p), axis=1)

    val_preds = np.argmax(val_logits_np, axis=1)
    val_correct = (val_preds == val_targets_np).astype(float)

    # Fit Platt scaling: P(correct | conf) = sigma(a * conf + b)
    # Use scipy minimize to find optimal a, b
    from scipy.optimize import minimize

    def platt_nll(params):
        a, b = params
        logit = a * val_conf + b
        logit = np.clip(logit, -30, 30)  # numerical stability
        prob = 1.0 / (1.0 + np.exp(-logit))
        prob = np.clip(prob, 1e-7, 1.0 - 1e-7)
        loss = -np.mean(val_correct * np.log(prob) + (1 - val_correct) * np.log(1 - prob))
        return loss

    result = minimize(platt_nll, x0=[1.0, 0.0], method="Nelder-Mead")
    a_opt, b_opt = result.x

    # Apply Platt scaling to test
    test_logit = a_opt * test_conf + b_opt
    test_logit = np.clip(test_logit, -30, 30)
    test_calibrated_conf = 1.0 / (1.0 + np.exp(-test_logit))

    # Construct calibrated probs: use original softmax probs but override confidence
    test_probs_orig = np.exp(test_logits_np - np.max(test_logits_np, axis=1, keepdims=True))
    test_probs_orig /= np.sum(test_probs_orig, axis=1, keepdims=True)

    return {
        "name": "pNorm-Platt (Fix B)",
        "params": {"p": best_p if isinstance(best_p, str) else int(best_p), "a": float(a_opt), "b": float(b_opt)},
        "test_probs": test_probs_orig,
        "test_conf_override": test_calibrated_conf,
    }


def method_pnorm_isotonic(val_logits, val_targets, test_logits):
    """
    Fix C: Isotonic regression on pNorm confidence scores.
    
    Idea: Learn a non-parametric, monotonically increasing mapping from pNorm
    confidence to P(correct). Isotonic regression is the gold standard for
    preserving ranking while achieving calibration, since the mapping is
    monotonic by construction.
    """
    val_logits_np = val_logits.detach().cpu().numpy() if isinstance(val_logits, torch.Tensor) else val_logits
    test_logits_np = test_logits.detach().cpu().numpy() if isinstance(test_logits, torch.Tensor) else test_logits
    val_targets_np = val_targets.detach().cpu().numpy() if isinstance(val_targets, torch.Tensor) else val_targets

    best_p = _find_best_p(val_logits_np, val_targets_np)
    if best_p == "MSP":
        val_conf = _msp_np(val_logits_np)
        test_conf = _msp_np(test_logits_np)
    else:
        val_conf = np.max(_get_pnorm_logits(val_logits_np, best_p), axis=1)
        test_conf = np.max(_get_pnorm_logits(test_logits_np, best_p), axis=1)

    val_preds = np.argmax(val_logits_np, axis=1)
    val_correct = (val_preds == val_targets_np).astype(float)

    # Fit isotonic regression: maps confidence -> P(correct)
    iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
    iso.fit(val_conf, val_correct)

    test_calibrated_conf = iso.predict(test_conf)

    # Use original softmax probs for NLL/Brier
    test_probs_orig = np.exp(test_logits_np - np.max(test_logits_np, axis=1, keepdims=True))
    test_probs_orig /= np.sum(test_probs_orig, axis=1, keepdims=True)

    return {
        "name": "pNorm-Isotonic (Fix C)",
        "params": {"p": best_p if isinstance(best_p, str) else int(best_p)},
        "test_probs": test_probs_orig,
        "test_conf_override": test_calibrated_conf,
    }


def method_pnorm_blend(val_logits, val_targets, test_logits):
    """
    Fix D: Blend pNorm confidence with original MSP.
    
    Idea: pNorm provides good ranking, MSP provides calibrated probabilities.
    Blend: conf = beta * pNorm_conf + (1 - beta) * MSP_conf.
    Optimize beta on val set to find the sweet spot between ranking and calibration.
    """
    val_logits_np = val_logits.detach().cpu().numpy() if isinstance(val_logits, torch.Tensor) else val_logits
    test_logits_np = test_logits.detach().cpu().numpy() if isinstance(test_logits, torch.Tensor) else test_logits
    val_targets_np = val_targets.detach().cpu().numpy() if isinstance(val_targets, torch.Tensor) else val_targets

    best_p = _find_best_p(val_logits_np, val_targets_np)
    if best_p == "MSP":
        val_pnorm_conf = _msp_np(val_logits_np)
        test_pnorm_conf = _msp_np(test_logits_np)
    else:
        val_pnorm_conf = np.max(_get_pnorm_logits(val_logits_np, best_p), axis=1)
        test_pnorm_conf = np.max(_get_pnorm_logits(test_logits_np, best_p), axis=1)

    # Normalize pNorm confidence to [0, 1] via min-max on val set
    pnorm_min = np.min(val_pnorm_conf)
    pnorm_max = np.max(val_pnorm_conf)
    if pnorm_max - pnorm_min > 1e-12:
        val_pnorm_norm = (val_pnorm_conf - pnorm_min) / (pnorm_max - pnorm_min)
        test_pnorm_norm = np.clip((test_pnorm_conf - pnorm_min) / (pnorm_max - pnorm_min), 0, 1)
    else:
        val_pnorm_norm = val_pnorm_conf
        test_pnorm_norm = test_pnorm_conf

    val_msp = _msp_np(val_logits_np)
    test_msp = _msp_np(test_logits_np)
    val_preds = np.argmax(val_logits_np, axis=1)

    # Grid search for optimal beta
    best_beta = 0.5
    best_score = float("inf")
    for beta in np.arange(0.0, 1.05, 0.05):
        blended = beta * val_pnorm_norm + (1 - beta) * val_msp
        score = evaluate_sc_metrics(val_preds, blended, val_targets_np)["aurc"]
        if score < best_score:
            best_score = score
            best_beta = beta

    test_blended = best_beta * test_pnorm_norm + (1 - best_beta) * test_msp

    # Use original softmax probs for NLL/Brier
    test_probs_orig = np.exp(test_logits_np - np.max(test_logits_np, axis=1, keepdims=True))
    test_probs_orig /= np.sum(test_probs_orig, axis=1, keepdims=True)

    return {
        "name": "pNorm-MSP Blend (Fix D)",
        "params": {"p": best_p if isinstance(best_p, str) else int(best_p), "beta": float(best_beta)},
        "test_probs": test_probs_orig,
        "test_conf_override": test_blended,
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
    """DOCTOR Alpha confidence: negative DOCTOR uncertainty ratio."""
    g_x = 1.0 - np.sum(np.power(probs, alpha), axis=1)
    return -(g_x / (1.0 - g_x + 1e-8))


def doctor_beta_confidence(probs: np.ndarray) -> np.ndarray:
    """DOCTOR Beta confidence: negative DOCTOR uncertainty ratio."""
    pred_probs = np.max(probs, axis=1)
    b_x = 1.0 - pred_probs
    return -(b_x / (1.0 - b_x + 1e-8))


def method_doctor_alpha(val_probs, val_targets, test_probs, num_classes):
    """DOCTOR Alpha method with the original default alpha=2."""
    alpha = 2.0
    test_conf = doctor_alpha_confidence(test_probs, alpha=alpha)
    test_probs_copy = np.copy(test_probs)
    
    return {
        "name": "DOCTOR-Alpha",
        "params": {"alpha": alpha},
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
from sklearn.neighbors import KDTree


class TrustScore:
    """
    Trust Score: a measure of classifier uncertainty based on nearest neighbors.
    """

    def __init__(self, k=10, alpha=0.0, filtering="none", min_dist=1e-12):
        self.k = k
        self.filtering = filtering
        self.alpha = alpha
        self.min_dist = min_dist

    def filter_by_density(self, X: np.array):
        kdtree = KDTree(X)
        knn_radii = kdtree.query(X, k=self.k)[0][:, -1]
        eps = np.percentile(knn_radii, (1 - self.alpha) * 100)
        return X[np.where(knn_radii <= eps)[0], :]

    def filter_by_uncertainty(self, X: np.array, y: np.array):
        neigh = KNeighborsClassifier(n_neighbors=self.k)
        neigh.fit(X, y)
        confidence = neigh.predict_proba(X)
        cutoff = np.percentile(confidence, self.alpha * 100)
        unfiltered_idxs = np.where(confidence >= cutoff)[0]
        return X[unfiltered_idxs, :], y[unfiltered_idxs]

    def fit(self, X: np.array, y: np.array):
        self.n_labels = np.max(y) + 1
        self.kdtrees = [None] * self.n_labels
        if self.filtering == "uncertainty":
            X_filtered, y_filtered = self.filter_by_uncertainty(X, y)
        for label in range(self.n_labels):
            if self.filtering == "none":
                X_to_use = X[np.where(y == label)[0]]
                self.kdtrees[label] = KDTree(X_to_use)
            elif self.filtering == "density":
                X_to_use = self.filter_by_density(X[np.where(y == label)[0]])
                self.kdtrees[label] = KDTree(X_to_use)
            elif self.filtering == "uncertainty":
                X_to_use = X_filtered[np.where(y_filtered == label)[0]]
                self.kdtrees[label] = KDTree(X_to_use)

            if len(X_to_use) == 0:
                print(
                    "Filtered too much or missing examples from a label! Please lower "
                    "alpha or check data."
                )

    def get_score(self, X: np.array, y_pred: np.array):
        d = np.zeros((X.shape[0], self.n_labels), dtype=np.float64)
        for label_idx in range(self.n_labels):
            d[:, label_idx] = self.kdtrees[label_idx].query(X, k=2)[0][:, -1]

        sorted_d = np.sort(d, axis=1)
        d_to_pred = d[range(d.shape[0]), y_pred]
        d_to_closest_not_pred = np.where(
            sorted_d[:, 0] != d_to_pred, sorted_d[:, 0], sorted_d[:, 1]
        )
        return d_to_closest_not_pred / (d_to_pred + self.min_dist)


def method_scp(train_features, train_targets, test_features, test_probs, num_classes, k=5, T=0.1):
    """Softmin Class Probability (SCP) for Selective Classification."""
    import numpy as np
    from sklearn.neighbors import NearestNeighbors
    
    n_test = test_features.shape[0]
    distances = np.full((n_test, num_classes), np.inf)
    
    for c in range(num_classes):
        mask = (train_targets == c)
        c_feats = train_features[mask]
        if len(c_feats) == 0:
            continue
        k_c = min(k, len(c_feats))
        nn = NearestNeighbors(n_neighbors=k_c, metric="euclidean")
        nn.fit(c_feats)
        dist, _ = nn.kneighbors(test_features)
        distances[:, c] = np.mean(dist, axis=1)
        
    distances = np.where(np.isinf(distances), 1e6, distances)
    log_scores = -distances / T
    log_scores -= np.max(log_scores, axis=1, keepdims=True)
    exp_scores = np.exp(log_scores)
    scp_probs = exp_scores / (np.sum(exp_scores, axis=1, keepdims=True) + 1e-12)
    
    scp_conf = np.max(scp_probs, axis=1)
    
    return {
        "name": f"Softmin Class Probability (k={k}, T={T})",
        "params": {"k": k, "T": T},
        "test_probs": test_probs,
        "test_conf_override": scp_conf,
    }


def _to_numpy(x, dtype=None) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    arr = np.asarray(x)
    if dtype is not None:
        arr = arr.astype(dtype, copy=False)
    return arr


def _top2_from_logits(logits) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    logits_np = _to_numpy(logits)
    order = np.argsort(logits_np, axis=1)
    top1 = order[:, -1].astype(np.int64)
    top2 = order[:, -2].astype(np.int64)
    margin = logits_np[np.arange(logits_np.shape[0]), top1] - logits_np[np.arange(logits_np.shape[0]), top2]
    return top1, top2, margin


def _mclr_feature_normalizer(ref_features: np.ndarray, mode: str = "zscore_l2"):
    mean = np.mean(ref_features, axis=0, keepdims=True)
    std = np.std(ref_features, axis=0, keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)

    def normalize(features: np.ndarray) -> np.ndarray:
        if mode == "zscore":
            return (features - mean) / std
        if mode == "l2":
            z = features
        elif mode == "zscore_l2":
            z = (features - mean) / std
        else:
            raise ValueError(f"Unknown MCLR feature normalization mode: {mode}")
        norm = np.linalg.norm(z, axis=1, keepdims=True)
        return z / np.clip(norm, 1e-12, None)

    return normalize


def _build_class_neighbor_indices(ref_features: np.ndarray, ref_targets: np.ndarray, num_classes: int, n_anchor: int):
    class_index = {}
    for c in range(num_classes):
        idx = np.where(ref_targets == c)[0]
        if len(idx) == 0:
            continue
        nn = NearestNeighbors(n_neighbors=min(n_anchor, len(idx)), metric="euclidean")
        nn.fit(ref_features[idx])
        class_index[c] = (idx, nn)
    return class_index


def _compute_mixup_chord_statistics(
    query_features: np.ndarray,
    query_logits,
    ref_features: np.ndarray,
    ref_targets: np.ndarray,
    num_classes: int,
    k_chord: int = 5,
    n_anchor: int = 10,
    logit_temperature: float = 1.0,
    residual_temperature: float = None,
    feature_norm: str = "zscore_l2",
    eps: float = 1e-8,
) -> np.ndarray:
    """Return [R_ab, E_ab, -margin] MCLR statistic vectors."""
    query_features = _to_numpy(query_features, dtype=np.float32)
    ref_features = _to_numpy(ref_features, dtype=np.float32)
    ref_targets = _to_numpy(ref_targets, dtype=np.int64)
    query_logits = _to_numpy(query_logits, dtype=np.float32)

    normalize = _mclr_feature_normalizer(ref_features, mode=feature_norm)
    ref_z = normalize(ref_features).astype(np.float32)
    query_z = normalize(query_features).astype(np.float32)
    top1, top2, margin = _top2_from_logits(query_logits)
    class_index = _build_class_neighbor_indices(ref_z, ref_targets, num_classes, n_anchor)

    stats = np.zeros((query_z.shape[0], 3), dtype=np.float32)
    fallback_R = math.log(1.0 + eps)

    for n in range(query_z.shape[0]):
        a = int(top1[n])
        b = int(top2[n])
        z = query_z[n]

        if a not in class_index or b not in class_index:
            stats[n] = np.array([fallback_R, 1.0, -margin[n]], dtype=np.float32)
            continue

        idx_a, nn_a = class_index[a]
        idx_b, nn_b = class_index[b]
        _, loc_a = nn_a.kneighbors(z.reshape(1, -1), return_distance=True)
        _, loc_b = nn_b.kneighbors(z.reshape(1, -1), return_distance=True)
        za = ref_z[idx_a[loc_a[0]]]
        zb = ref_z[idx_b[loc_b[0]]]

        pair_a = za[:, None, :]
        pair_b = zb[None, :, :]
        chord = pair_a - pair_b
        denom = np.sum(chord * chord, axis=2)
        lam = np.sum((z - pair_b) * chord, axis=2) / np.clip(denom, eps, None)
        lam = np.clip(lam, 0.0, 1.0)
        proj = lam[:, :, None] * pair_a + (1.0 - lam[:, :, None]) * pair_b
        residual = np.sum((z - proj) ** 2, axis=2)

        flat_r = residual.reshape(-1)
        flat_lam = lam.reshape(-1)
        k_eff = min(k_chord, flat_r.shape[0])
        nearest = np.argpartition(flat_r, k_eff - 1)[:k_eff]
        nearest_r = flat_r[nearest]
        nearest_lam = flat_lam[nearest]

        R = float(np.mean(np.log(eps + nearest_r)))
        if residual_temperature is None:
            scale = float(np.median(nearest_r) + eps)
        else:
            scale = float(residual_temperature)
        weights = np.exp(-nearest_r / max(scale, eps))
        weights = weights / np.clip(np.sum(weights), eps, None)
        lambda_geo = float(np.sum(nearest_lam * weights))
        lambda_logit = 1.0 / (1.0 + np.exp(-float(margin[n]) / max(logit_temperature, eps)))
        E = (lambda_geo - lambda_logit) ** 2
        stats[n] = np.array([R, E, -margin[n]], dtype=np.float32)

    return stats


def _density_knn_score(query_m: np.ndarray, ref_m: np.ndarray, k: int, eps: float = 1e-8) -> np.ndarray:
    if ref_m.shape[0] == 0:
        return np.full(query_m.shape[0], -1e6, dtype=np.float32)
    k_eff = min(k, ref_m.shape[0])
    nn = NearestNeighbors(n_neighbors=k_eff, metric="euclidean")
    nn.fit(ref_m)
    dist, _ = nn.kneighbors(query_m)
    return -np.mean(np.log(eps + dist), axis=1).astype(np.float32)


def _standardize_mclr_stats(ref_m: np.ndarray, query_m: np.ndarray, clip_quantile: float = 0.01):
    ref_m = _to_numpy(ref_m, dtype=np.float32)
    query_m = _to_numpy(query_m, dtype=np.float32)
    if clip_quantile is not None and 0.0 < clip_quantile < 0.5:
        lo = np.quantile(ref_m, clip_quantile, axis=0, keepdims=True)
        hi = np.quantile(ref_m, 1.0 - clip_quantile, axis=0, keepdims=True)
        ref_m = np.clip(ref_m, lo, hi)
        query_m = np.clip(query_m, lo, hi)
    mean = np.mean(ref_m, axis=0, keepdims=True)
    std = np.std(ref_m, axis=0, keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)
    return (ref_m - mean) / std, (query_m - mean) / std


def method_delta_mclr(
    train_features,
    train_targets,
    val_features,
    val_logits,
    val_targets,
    test_features,
    test_logits,
    test_probs,
    num_classes,
):
    """Delta-MCLR: likelihood-ratio scoring in Mixup chord statistic space."""
    print("Evaluating Delta-MCLR...")

    train_features = _to_numpy(train_features, dtype=np.float32)
    train_targets = _to_numpy(train_targets, dtype=np.int64)
    val_features = _to_numpy(val_features, dtype=np.float32)
    val_logits = _to_numpy(val_logits, dtype=np.float32)
    val_targets = _to_numpy(val_targets, dtype=np.int64)
    test_features = _to_numpy(test_features, dtype=np.float32)
    test_logits = _to_numpy(test_logits, dtype=np.float32)
    test_probs = _to_numpy(test_probs, dtype=np.float32)

    val_preds, _, _ = _top2_from_logits(val_logits)
    val_correct = val_preds == val_targets

    val_m = _compute_mixup_chord_statistics(
        val_features,
        val_logits,
        train_features,
        train_targets,
        num_classes,
    )
    test_m = _compute_mixup_chord_statistics(
        test_features,
        test_logits,
        train_features,
        train_targets,
        num_classes,
    )

    split = val_m.shape[0] // 2
    ref_m_raw = val_m[:split]
    calib_m_raw = val_m[split:]
    ref_correct = val_correct[:split]
    calib_targets = val_targets[split:]
    calib_preds = val_preds[split:]

    if np.sum(ref_correct) == 0 or np.sum(~ref_correct) == 0:
        print("  Delta-MCLR fallback: validation split lacks both correct and wrong samples.")
        test_conf = _msp_np(test_logits)
        return {
            "name": "Delta-MCLR (fallback MSP)",
            "params": {},
            "test_probs": test_probs,
            "test_conf_override": test_conf,
        }

    ref_m, calib_m = _standardize_mclr_stats(ref_m_raw, calib_m_raw)
    m_correct = ref_m[ref_correct]
    m_wrong = ref_m[~ref_correct]

    candidate_k = [k for k in [3, 5, 10, 20, 30, 50] if k <= min(len(m_correct), len(m_wrong))]
    if not candidate_k:
        candidate_k = [1]

    best = None
    for k in candidate_k:
        calib_conf = _density_knn_score(calib_m, m_correct, k) - _density_knn_score(calib_m, m_wrong, k)
        metrics = evaluate_sc_metrics(calib_preds, calib_conf, calib_targets)
        if best is None or metrics["aurc"] < best["score"]:
            best = {"score": metrics["aurc"], "k_density": int(k)}

    val_m_std, test_m_std = _standardize_mclr_stats(val_m, test_m)
    m_correct = val_m_std[val_correct]
    m_wrong = val_m_std[~val_correct]
    k_final = min(best["k_density"], len(m_correct), len(m_wrong))
    test_conf = _density_knn_score(test_m_std, m_correct, k_final) - _density_knn_score(test_m_std, m_wrong, k_final)

    return {
        "name": "Delta-MCLR (Mixup Chord Likelihood Ratio)",
        "params": {
            "k_density": int(k_final),
            "k_chord": 5,
            "n_anchor": 10,
            "logit_temperature": 1.0,
            "residual_temperature": "local_median",
            "feature_norm": "zscore_l2",
            "stat_clip_quantile": 0.01,
        },
        "test_probs": np.copy(test_probs),
        "test_conf_override": test_conf,
    }


def _last_linear_params_np(model: torch.nn.Module):
    if hasattr(model, "module"):
        model = model.module

    for module in reversed(list(model.modules())):
        if isinstance(module, torch.nn.Linear):
            weight = module.weight.detach().cpu().numpy().astype(np.float32)
            bias = None
            if module.bias is not None:
                bias = module.bias.detach().cpu().numpy().astype(np.float32)
            return weight, bias

    return None, None


def _compute_centroid_geometry_stats(
    train_features: np.ndarray,
    train_targets: np.ndarray,
    query_features: np.ndarray,
    query_logits: np.ndarray,
    num_classes: int,
    eps: float = 1e-8,
):
    normalize = _mclr_feature_normalizer(train_features, mode="zscore_l2")
    train_z = normalize(train_features).astype(np.float32)
    query_z = normalize(query_features).astype(np.float32)
    preds, _, _ = _top2_from_logits(query_logits)

    dim = train_z.shape[1]
    centroids = np.zeros((num_classes, dim), dtype=np.float32)
    present = np.zeros(num_classes, dtype=bool)
    for c in range(num_classes):
        idx = np.where(train_targets == c)[0]
        if len(idx) == 0:
            continue
        centroids[c] = np.mean(train_z[idx], axis=0)
        present[c] = True

    q2 = np.sum(query_z * query_z, axis=1, keepdims=True)
    c2 = np.sum(centroids * centroids, axis=1, keepdims=True).T
    dist = q2 + c2 - 2.0 * (query_z @ centroids.T)
    dist[:, ~present] = np.inf

    rows = np.arange(query_z.shape[0])
    pred_dist = dist[rows, preds]
    dist_without_pred = dist.copy()
    dist_without_pred[rows, preds] = np.inf
    nearest_other = np.min(dist_without_pred, axis=1)

    finite_dist = dist[np.isfinite(dist)]
    soft_scale = float(np.median(finite_dist)) if finite_dist.size else 1.0
    soft_scale = max(soft_scale, eps)
    logits = -dist / soft_scale
    logits[~np.isfinite(logits)] = -1e6
    logits -= np.max(logits, axis=1, keepdims=True)
    soft = np.exp(logits)
    soft /= np.clip(np.sum(soft, axis=1, keepdims=True), eps, None)
    pred_soft = soft[rows, preds]

    stats = np.stack(
        [
            -pred_dist,
            nearest_other - pred_dist,
            pred_soft,
        ],
        axis=1,
    )
    names = ["centroid_neg_pred_dist", "centroid_margin", "centroid_soft_conf"]
    return np.nan_to_num(stats.astype(np.float32), nan=0.0, posinf=1e6, neginf=-1e6), names


def _compute_classifier_feature_stats(
    model: torch.nn.Module,
    features: np.ndarray,
    logits: np.ndarray,
    eps: float = 1e-8,
):
    weight, _ = _last_linear_params_np(model)
    if weight is None or weight.shape[1] != features.shape[1]:
        return np.zeros((features.shape[0], 0), dtype=np.float32), []

    features = features.astype(np.float32, copy=False)
    logits = logits.astype(np.float32, copy=False)
    top1, top2, margin = _top2_from_logits(logits)
    rows = np.arange(features.shape[0])

    boundary_norm = np.linalg.norm(weight[top1] - weight[top2], axis=1)
    geo_margin = margin / np.clip(boundary_norm, eps, None)

    feat_norm = np.linalg.norm(features, axis=1)
    weight_norm = np.linalg.norm(weight, axis=1)
    pred_cos = np.sum(features * weight[top1], axis=1) / np.clip(feat_norm * weight_norm[top1], eps, None)

    try:
        _, singular_values, vt = np.linalg.svd(weight.astype(np.float64), full_matrices=False)
        rank = int(np.sum(singular_values > singular_values[0] * 1e-6)) if singular_values.size else 0
        if rank > 0 and rank < features.shape[1]:
            basis = vt[:rank].astype(np.float32)
            row_proj = (features @ basis.T) @ basis
            row_norm = np.linalg.norm(row_proj, axis=1)
            residual_norm = np.linalg.norm(features - row_proj, axis=1)
        elif rank >= features.shape[1]:
            row_norm = feat_norm
            residual_norm = np.zeros_like(feat_norm)
        else:
            row_norm = np.zeros_like(feat_norm)
            residual_norm = feat_norm
    except np.linalg.LinAlgError:
        row_norm = feat_norm
        residual_norm = np.zeros_like(feat_norm)

    residual_ratio = residual_norm / np.clip(feat_norm, eps, None)
    row_ratio = row_norm / np.clip(feat_norm, eps, None)
    log_feat_norm = np.log(eps + feat_norm)

    stats = np.stack(
        [
            geo_margin,
            pred_cos,
            residual_ratio,
            row_ratio,
            log_feat_norm,
        ],
        axis=1,
    )
    names = ["geo_margin", "pred_weight_cos", "row_residual_ratio", "row_norm_ratio", "log_feature_norm"]
    return np.nan_to_num(stats.astype(np.float32), nan=0.0, posinf=1e6, neginf=-1e6), names


def _clip_fit_standardizer(ref_m: np.ndarray, clip_quantile: float = 0.01, eps: float = 1e-6):
    if clip_quantile is not None and 0.0 < clip_quantile < 0.5:
        lo = np.quantile(ref_m, clip_quantile, axis=0, keepdims=True)
        hi = np.quantile(ref_m, 1.0 - clip_quantile, axis=0, keepdims=True)
    else:
        lo = np.full((1, ref_m.shape[1]), -np.inf, dtype=np.float32)
        hi = np.full((1, ref_m.shape[1]), np.inf, dtype=np.float32)
    clipped = np.clip(ref_m, lo, hi)
    mean = np.mean(clipped, axis=0, keepdims=True)
    std = np.std(clipped, axis=0, keepdims=True)
    std = np.where(std < eps, 1.0, std)
    return lo.astype(np.float32), hi.astype(np.float32), mean.astype(np.float32), std.astype(np.float32)


def _clip_apply_standardizer(m: np.ndarray, scaler):
    lo, hi, mean, std = scaler
    return ((np.clip(m, lo, hi) - mean) / std).astype(np.float32)


def _compute_pfm_stats(
    train_features,
    train_targets,
    query_features,
    query_logits,
    num_classes: int,
    model: torch.nn.Module,
    p_choice,
):
    query_logits = _to_numpy(query_logits, dtype=np.float32)
    query_features = _to_numpy(query_features, dtype=np.float32)
    train_features = _to_numpy(train_features, dtype=np.float32)
    train_targets = _to_numpy(train_targets, dtype=np.int64)

    logit_stats = _compute_mllr_stats(query_logits, p_choice=p_choice)
    logit_names = ["pNorm", "margin", "tail_leakage", "endpointness"]

    centroid_stats, centroid_names = _compute_centroid_geometry_stats(
        train_features,
        train_targets,
        query_features,
        query_logits,
        num_classes,
    )
    classifier_stats, classifier_names = _compute_classifier_feature_stats(model, query_features, query_logits)

    stats = np.concatenate([logit_stats, centroid_stats, classifier_stats], axis=1)
    names = logit_names + centroid_names + classifier_names
    return np.nan_to_num(stats.astype(np.float32), nan=0.0, posinf=1e6, neginf=-1e6), names


def method_pnorm_feature_hybrid(
    train_features,
    train_targets,
    val_features,
    val_logits,
    val_targets,
    test_features,
    test_logits,
    test_probs,
    num_classes,
    model,
):
    """
    pNorm-Feature Hybrid: validation-trained correctness score.

    The score keeps pNorm as the main coordinate, then lets centroid and
    classifier-rowspace feature statistics make a small learned correction.
    """
    print("Evaluating pNorm-Feature Hybrid...")

    val_logits_np = _to_numpy(val_logits, dtype=np.float32)
    test_logits_np = _to_numpy(test_logits, dtype=np.float32)
    val_targets_np = _to_numpy(val_targets, dtype=np.int64)
    test_probs_np = _to_numpy(test_probs, dtype=np.float32)

    val_preds = np.argmax(val_logits_np, axis=1)
    val_correct = (val_preds == val_targets_np).astype(np.int64)
    split = val_logits_np.shape[0] // 2

    if len(np.unique(val_correct[:split])) < 2 or len(np.unique(val_correct[split:])) < 2:
        p_choice, _ = _select_best_pnorm_p(val_logits_np, val_targets_np, metric="aurc")
        test_conf = _pnorm_conf_from_logits(test_logits_np, p_choice)
        return {
            "name": "pNorm-Feature Hybrid (fallback pNorm)",
            "params": {"p": p_choice, "reason": "missing_correct_or_wrong_split"},
            "test_probs": test_probs_np,
            "test_conf_override": test_conf,
        }

    p_ref, _ = _select_best_pnorm_p(val_logits_np[:split], val_targets_np[:split], metric="aurc")
    val_stats_ref, stat_names = _compute_pfm_stats(
        train_features,
        train_targets,
        val_features,
        val_logits_np,
        num_classes,
        model,
        p_ref,
    )

    ref_raw = val_stats_ref[:split]
    calib_raw = val_stats_ref[split:]
    scaler = _clip_fit_standardizer(ref_raw)
    ref_stats = _clip_apply_standardizer(ref_raw, scaler)
    calib_stats = _clip_apply_standardizer(calib_raw, scaler)

    name_to_idx = {name: i for i, name in enumerate(stat_names)}
    subset_defs = {
        "logit": [name_to_idx[n] for n in ["pNorm", "margin", "tail_leakage", "endpointness"] if n in name_to_idx],
        "pnorm_centroid": [name_to_idx[n] for n in ["pNorm", "centroid_neg_pred_dist", "centroid_margin", "centroid_soft_conf"] if n in name_to_idx],
        "pnorm_classifier": [name_to_idx[n] for n in ["pNorm", "geo_margin", "pred_weight_cos", "row_residual_ratio", "row_norm_ratio", "log_feature_norm"] if n in name_to_idx],
        "all": list(range(len(stat_names))),
    }

    calib_targets = val_targets_np[split:]
    calib_preds = val_preds[split:]
    ref_correct = val_correct[:split]
    raw_pnorm_calib = val_stats_ref[split:, name_to_idx["pNorm"]]

    best = {
        "score": evaluate_sc_metrics(calib_preds, raw_pnorm_calib, calib_targets)["aurc"],
        "kind": "raw_pnorm",
        "subset": "pNorm",
        "C": None,
        "beta": 1.0,
    }

    c_grid = [0.03, 0.1, 0.3, 1.0, 3.0, 10.0]
    beta_grid = np.linspace(0.0, 1.0, 11)

    for subset_name, cols in subset_defs.items():
        if not cols:
            continue
        x_ref = ref_stats[:, cols]
        x_calib = calib_stats[:, cols]
        for c_value in c_grid:
            try:
                clf = LogisticRegression(
                    C=float(c_value),
                    class_weight="balanced",
                    max_iter=1000,
                    solver="lbfgs",
                )
                clf.fit(x_ref, ref_correct)
            except ValueError:
                continue

            calib_lr = clf.decision_function(x_calib).astype(np.float32)
            metrics = evaluate_sc_metrics(calib_preds, calib_lr, calib_targets)
            if metrics["aurc"] < best["score"]:
                best = {
                    "score": float(metrics["aurc"]),
                    "kind": "logreg",
                    "subset": subset_name,
                    "C": float(c_value),
                    "beta": 0.0,
                }

            pnorm_z = (raw_pnorm_calib - np.mean(raw_pnorm_calib)) / (np.std(raw_pnorm_calib) + 1e-6)
            lr_z = (calib_lr - np.mean(calib_lr)) / (np.std(calib_lr) + 1e-6)
            for beta in beta_grid:
                blend = float(beta) * pnorm_z + (1.0 - float(beta)) * lr_z
                metrics = evaluate_sc_metrics(calib_preds, blend, calib_targets)
                if metrics["aurc"] < best["score"]:
                    best = {
                        "score": float(metrics["aurc"]),
                        "kind": "blend",
                        "subset": subset_name,
                        "C": float(c_value),
                        "beta": float(beta),
                    }

    p_final, _ = _select_best_pnorm_p(val_logits_np, val_targets_np, metric="aurc")
    val_stats, stat_names = _compute_pfm_stats(
        train_features,
        train_targets,
        val_features,
        val_logits_np,
        num_classes,
        model,
        p_final,
    )
    test_stats, _ = _compute_pfm_stats(
        train_features,
        train_targets,
        test_features,
        test_logits_np,
        num_classes,
        model,
        p_final,
    )
    scaler = _clip_fit_standardizer(val_stats)
    val_std = _clip_apply_standardizer(val_stats, scaler)
    test_std = _clip_apply_standardizer(test_stats, scaler)
    name_to_idx = {name: i for i, name in enumerate(stat_names)}
    subset_defs = {
        "logit": [name_to_idx[n] for n in ["pNorm", "margin", "tail_leakage", "endpointness"] if n in name_to_idx],
        "pnorm_centroid": [name_to_idx[n] for n in ["pNorm", "centroid_neg_pred_dist", "centroid_margin", "centroid_soft_conf"] if n in name_to_idx],
        "pnorm_classifier": [name_to_idx[n] for n in ["pNorm", "geo_margin", "pred_weight_cos", "row_residual_ratio", "row_norm_ratio", "log_feature_norm"] if n in name_to_idx],
        "all": list(range(len(stat_names))),
    }

    if best["kind"] == "raw_pnorm":
        test_conf = test_stats[:, name_to_idx["pNorm"]]
    else:
        cols = subset_defs[best["subset"]]
        clf = LogisticRegression(
            C=float(best["C"]),
            class_weight="balanced",
            max_iter=1000,
            solver="lbfgs",
        )
        clf.fit(val_std[:, cols], val_correct)
        lr_score = clf.decision_function(test_std[:, cols]).astype(np.float32)
        if best["kind"] == "logreg":
            test_conf = lr_score
        else:
            raw_pnorm_val = val_stats[:, name_to_idx["pNorm"]]
            raw_pnorm_test = test_stats[:, name_to_idx["pNorm"]]
            pnorm_z_test = (raw_pnorm_test - np.mean(raw_pnorm_val)) / (np.std(raw_pnorm_val) + 1e-6)
            lr_val = clf.decision_function(val_std[:, cols]).astype(np.float32)
            lr_z_test = (lr_score - np.mean(lr_val)) / (np.std(lr_val) + 1e-6)
            test_conf = best["beta"] * pnorm_z_test + (1.0 - best["beta"]) * lr_z_test

    return {
        "name": "pNorm-Feature Hybrid",
        "params": {
            "p": p_final if isinstance(p_final, str) else int(p_final),
            "p_ref": p_ref if isinstance(p_ref, str) else int(p_ref),
            "kind": best["kind"],
            "subset": best["subset"],
            "C": best["C"],
            "beta": float(best["beta"]),
            "calib_aurc": float(best["score"]),
            "stat_names": stat_names,
        },
        "test_probs": test_probs_np,
        "test_conf_override": test_conf.astype(np.float32),
    }


def _classifier_row_basis(model: torch.nn.Module, feature_dim: int, eps: float = 1e-6):
    weight, bias = _last_linear_params_np(model)
    if weight is None or weight.shape[1] != feature_dim:
        return None, None, 0, None

    try:
        _, singular_values, vt = np.linalg.svd(weight.astype(np.float64), full_matrices=False)
    except np.linalg.LinAlgError:
        return weight, bias, 0, None

    if singular_values.size == 0:
        return weight, bias, 0, None
    rank = int(np.sum(singular_values > singular_values[0] * eps))
    basis = vt[:rank].astype(np.float32)
    return weight, bias, rank, basis


def _row_null_projection(features: np.ndarray, basis: np.ndarray):
    features = _to_numpy(features, dtype=np.float32)
    if basis is None or basis.size == 0:
        row = np.zeros_like(features)
        null = features
    else:
        row = (features @ basis.T) @ basis
        null = features - row
    return row.astype(np.float32), null.astype(np.float32)


def _fit_apply_feature_space(ref_x: np.ndarray, query_x: np.ndarray, l2: bool = True):
    mean = np.mean(ref_x, axis=0, keepdims=True)
    std = np.std(ref_x, axis=0, keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)
    ref = (ref_x - mean) / std
    query = (query_x - mean) / std
    if l2:
        ref = ref / np.clip(np.linalg.norm(ref, axis=1, keepdims=True), 1e-12, None)
        query = query / np.clip(np.linalg.norm(query, axis=1, keepdims=True), 1e-12, None)
    return ref.astype(np.float32), query.astype(np.float32)


def _tune_delta_knn_space(
    ref_raw: np.ndarray,
    ref_correct: np.ndarray,
    calib_raw: np.ndarray,
    calib_preds: np.ndarray,
    calib_targets: np.ndarray,
    candidate_k=(3, 5, 10, 20, 30, 50, 75, 100),
):
    ref_x, calib_x = _fit_apply_feature_space(ref_raw, calib_raw)
    correct_bank = ref_x[ref_correct]
    wrong_bank = ref_x[~ref_correct]
    max_k = min(len(correct_bank), len(wrong_bank))
    k_grid = [k for k in candidate_k if k <= max_k] or [1]

    best = None
    for k in k_grid:
        conf = _knn_log_density_score(calib_x, correct_bank, k) - _knn_log_density_score(calib_x, wrong_bank, k)
        metrics = evaluate_sc_metrics(calib_preds, conf, calib_targets)
        if best is None or metrics["aurc"] < best["score"]:
            best = {"score": float(metrics["aurc"]), "k": int(k), "auroc": float(metrics["auroc"])}
    return best


def _final_delta_knn_space(val_raw: np.ndarray, val_correct: np.ndarray, test_raw: np.ndarray, k: int):
    val_x, test_x = _fit_apply_feature_space(val_raw, test_raw)
    correct_bank = val_x[val_correct]
    wrong_bank = val_x[~val_correct]
    k_eff = min(int(k), len(correct_bank), len(wrong_bank))
    if k_eff <= 0:
        return np.zeros(test_x.shape[0], dtype=np.float32)
    return _knn_log_density_score(test_x, correct_bank, k_eff) - _knn_log_density_score(test_x, wrong_bank, k_eff)


def _method_row_or_null_delta_knn(
    val_features,
    val_logits,
    val_targets,
    test_features,
    test_probs,
    model,
    component: str,
):
    val_features = _to_numpy(val_features, dtype=np.float32)
    test_features = _to_numpy(test_features, dtype=np.float32)
    val_logits = _to_numpy(val_logits, dtype=np.float32)
    val_targets = _to_numpy(val_targets, dtype=np.int64)
    test_probs = _to_numpy(test_probs, dtype=np.float32)

    weight, bias, rank, basis = _classifier_row_basis(model, val_features.shape[1])
    if weight is None or basis is None:
        conf = _msp_np(val_logits)
        return None

    val_row, val_null = _row_null_projection(val_features, basis)
    test_row, test_null = _row_null_projection(test_features, basis)
    val_raw = val_null if component == "null" else val_row
    test_raw = test_null if component == "null" else test_row

    val_preds = np.argmax(val_logits, axis=1)
    val_correct = val_preds == val_targets
    split = val_features.shape[0] // 2
    ref_correct = val_correct[:split]
    if np.sum(ref_correct) == 0 or np.sum(~ref_correct) == 0:
        return None

    best = _tune_delta_knn_space(
        val_raw[:split],
        ref_correct,
        val_raw[split:],
        val_preds[split:],
        val_targets[split:],
    )
    test_conf = _final_delta_knn_space(val_raw, val_correct, test_raw, best["k"])
    null_dim = max(0, val_features.shape[1] - rank)

    label = "Null-space" if component == "null" else "Row-space"
    return {
        "name": f"Delta-{label} kNN",
        "params": {
            "k": int(best["k"]),
            "rank_W": int(rank),
            "null_dim": int(null_dim),
            "calib_aurc": float(best["score"]),
            "calib_auroc": float(best["auroc"]),
        },
        "test_probs": test_probs,
        "test_conf_override": test_conf,
    }


def method_delta_nullspace_knn(val_features, val_logits, val_targets, test_features, test_probs, model):
    """Hypothesis test: does z_perp alone contain correctness information?"""
    print("Evaluating Delta-Null-space kNN...")
    return _method_row_or_null_delta_knn(
        val_features,
        val_logits,
        val_targets,
        test_features,
        test_probs,
        model,
        component="null",
    )


def method_delta_rowspace_knn(val_features, val_logits, val_targets, test_features, test_probs, model):
    """Hypothesis test: does row(W) feature density explain correctness?"""
    print("Evaluating Delta-Row-space kNN...")
    return _method_row_or_null_delta_knn(
        val_features,
        val_logits,
        val_targets,
        test_features,
        test_probs,
        model,
        component="row",
    )


def _row_null_scalar_stats(features: np.ndarray, logits: np.ndarray, model: torch.nn.Module):
    weight, bias, rank, basis = _classifier_row_basis(model, features.shape[1])
    if weight is None or basis is None:
        return np.zeros((features.shape[0], 0), dtype=np.float32), [], 0, 0

    row, null = _row_null_projection(features, basis)
    feat_norm = np.linalg.norm(features, axis=1)
    row_norm = np.linalg.norm(row, axis=1)
    null_norm = np.linalg.norm(null, axis=1)
    null_ratio = null_norm / np.clip(feat_norm, 1e-8, None)
    row_ratio = row_norm / np.clip(feat_norm, 1e-8, None)

    top1, top2, margin = _top2_from_logits(logits)
    boundary_norm = np.linalg.norm(weight[top1] - weight[top2], axis=1)
    geo_margin = margin / np.clip(boundary_norm, 1e-8, None)

    stats = np.stack(
        [
            geo_margin,
            np.log1p(null_norm),
            null_ratio,
            row_ratio,
        ],
        axis=1,
    ).astype(np.float32)
    names = ["geo_margin", "log1p_null_norm", "null_ratio", "row_ratio"]
    null_dim = max(0, features.shape[1] - rank)
    return np.nan_to_num(stats, nan=0.0, posinf=1e6, neginf=-1e6), names, rank, null_dim


def _component_delta_for_stats(val_component, test_component, val_correct, split, k_component: int = 20):
    ref_x, calib_x = _fit_apply_feature_space(val_component[:split], val_component[split:])
    correct_bank = ref_x[val_correct[:split]]
    wrong_bank = ref_x[~val_correct[:split]]
    k_ref = min(k_component, len(correct_bank), len(wrong_bank))
    if k_ref <= 0:
        calib_delta = np.zeros(calib_x.shape[0], dtype=np.float32)
    else:
        calib_delta = _knn_log_density_score(calib_x, correct_bank, k_ref) - _knn_log_density_score(calib_x, wrong_bank, k_ref)

    val_x, test_x = _fit_apply_feature_space(val_component, test_component)
    correct_bank = val_x[val_correct]
    wrong_bank = val_x[~val_correct]
    k_all = min(k_component, len(correct_bank), len(wrong_bank))
    if k_all <= 0:
        test_delta = np.zeros(test_x.shape[0], dtype=np.float32)
        val_delta = np.zeros(val_x.shape[0], dtype=np.float32)
    else:
        val_delta = _knn_log_density_score(val_x, correct_bank, k_all) - _knn_log_density_score(val_x, wrong_bank, k_all)
        test_delta = _knn_log_density_score(test_x, correct_bank, k_all) - _knn_log_density_score(test_x, wrong_bank, k_all)
    return calib_delta, val_delta, test_delta


def method_delta_rnlr_statspace(
    val_features,
    val_logits,
    val_targets,
    test_features,
    test_logits,
    test_probs,
    model,
    candidate_k=(3, 5, 10, 20, 30, 50, 75, 100),
):
    """Row-Null Likelihood Ratio in low-dimensional statistic space."""
    print("Evaluating Delta-RNLR...")

    val_features = _to_numpy(val_features, dtype=np.float32)
    test_features = _to_numpy(test_features, dtype=np.float32)
    val_logits = _to_numpy(val_logits, dtype=np.float32)
    test_logits = _to_numpy(test_logits, dtype=np.float32)
    val_targets = _to_numpy(val_targets, dtype=np.int64)
    test_probs = _to_numpy(test_probs, dtype=np.float32)

    weight, bias, rank, basis = _classifier_row_basis(model, val_features.shape[1])
    if weight is None or basis is None:
        return None

    val_preds = np.argmax(val_logits, axis=1)
    val_correct = val_preds == val_targets
    split = val_features.shape[0] // 2
    if np.sum(val_correct[:split]) == 0 or np.sum(~val_correct[:split]) == 0:
        return None

    p_ref, _ = _select_best_pnorm_p(val_logits[:split], val_targets[:split], metric="aurc")
    val_logit_stats = _compute_mllr_stats(val_logits, p_choice=p_ref)
    test_logit_stats = _compute_mllr_stats(test_logits, p_choice=p_ref)
    val_rn_stats, rn_names, _, null_dim = _row_null_scalar_stats(val_features, val_logits, model)
    test_rn_stats, _, _, _ = _row_null_scalar_stats(test_features, test_logits, model)

    val_row, val_null = _row_null_projection(val_features, basis)
    test_row, test_null = _row_null_projection(test_features, basis)
    calib_null_delta, val_null_delta, test_null_delta = _component_delta_for_stats(val_null, test_null, val_correct, split)
    calib_row_delta, val_row_delta, test_row_delta = _component_delta_for_stats(val_row, test_row, val_correct, split)

    val_stats_ref = np.concatenate(
        [
            val_logit_stats,
            val_rn_stats,
            np.zeros((val_features.shape[0], 2), dtype=np.float32),
        ],
        axis=1,
    )
    val_stats_ref[split:, -2] = calib_null_delta
    val_stats_ref[split:, -1] = calib_row_delta
    # Fill the reference half with in-sample component deltas for standardization only.
    val_stats_ref[:split, -2] = np.mean(calib_null_delta) if calib_null_delta.size else 0.0
    val_stats_ref[:split, -1] = np.mean(calib_row_delta) if calib_row_delta.size else 0.0

    stat_names = ["pNorm", "margin", "tail_leakage", "endpointness"] + rn_names + ["null_delta", "row_delta"]
    ref_raw = val_stats_ref[:split]
    calib_raw = val_stats_ref[split:]
    scaler = _clip_fit_standardizer(ref_raw)
    ref_m = _clip_apply_standardizer(ref_raw, scaler)
    calib_m = _clip_apply_standardizer(calib_raw, scaler)
    correct_bank = ref_m[val_correct[:split]]
    wrong_bank = ref_m[~val_correct[:split]]
    max_k = min(len(correct_bank), len(wrong_bank))
    k_grid = [k for k in candidate_k if k <= max_k] or [1]

    best = None
    for k in k_grid:
        conf = _knn_log_density_score(calib_m, correct_bank, k) - _knn_log_density_score(calib_m, wrong_bank, k)
        metrics = evaluate_sc_metrics(val_preds[split:], conf, val_targets[split:])
        if best is None or metrics["aurc"] < best["score"]:
            best = {"score": float(metrics["aurc"]), "k": int(k), "auroc": float(metrics["auroc"])}

    p_final, _ = _select_best_pnorm_p(val_logits, val_targets, metric="aurc")
    val_logit_stats = _compute_mllr_stats(val_logits, p_choice=p_final)
    test_logit_stats = _compute_mllr_stats(test_logits, p_choice=p_final)
    val_stats = np.concatenate([val_logit_stats, val_rn_stats, val_null_delta[:, None], val_row_delta[:, None]], axis=1)
    test_stats = np.concatenate([test_logit_stats, test_rn_stats, test_null_delta[:, None], test_row_delta[:, None]], axis=1)

    scaler = _clip_fit_standardizer(val_stats)
    val_m = _clip_apply_standardizer(val_stats, scaler)
    test_m = _clip_apply_standardizer(test_stats, scaler)
    correct_bank = val_m[val_correct]
    wrong_bank = val_m[~val_correct]
    k_final = min(best["k"], len(correct_bank), len(wrong_bank))
    test_conf = _knn_log_density_score(test_m, correct_bank, k_final) - _knn_log_density_score(test_m, wrong_bank, k_final)

    return {
        "name": "Delta-RNLR (Row-Null LR)",
        "params": {
            "p": p_final if isinstance(p_final, str) else int(p_final),
            "p_ref": p_ref if isinstance(p_ref, str) else int(p_ref),
            "k_density": int(k_final),
            "rank_W": int(rank),
            "null_dim": int(null_dim),
            "stat_names": stat_names,
            "calib_aurc": float(best["score"]),
            "calib_auroc": float(best["auroc"]),
        },
        "test_probs": test_probs,
        "test_conf_override": test_conf.astype(np.float32),
    }


def _top2_channel_stats(features: np.ndarray, logits: np.ndarray, model: torch.nn.Module):
    weight, bias, rank, basis = _classifier_row_basis(model, features.shape[1])
    if weight is None:
        return np.zeros((features.shape[0], 0), dtype=np.float32), []

    top1, top2, margin = _top2_from_logits(logits)
    rows = np.arange(features.shape[0])
    boundary = weight[top1] - weight[top2]
    boundary_norm = np.linalg.norm(boundary, axis=1)
    geo_margin = margin / np.clip(boundary_norm, 1e-8, None)

    channel_off = np.zeros(features.shape[0], dtype=np.float32)
    channel_ratio = np.zeros(features.shape[0], dtype=np.float32)
    feat_norm = np.linalg.norm(features, axis=1)
    for i in range(features.shape[0]):
        pair = np.stack([weight[top1[i]], weight[top2[i]]], axis=0).astype(np.float64)
        try:
            q, _ = np.linalg.qr(pair.T, mode="reduced")
            proj = q @ (q.T @ features[i].astype(np.float64))
            residual = features[i].astype(np.float64) - proj
            channel_off[i] = float(np.linalg.norm(residual))
        except np.linalg.LinAlgError:
            channel_off[i] = 0.0
        channel_ratio[i] = channel_off[i] / max(float(feat_norm[i]), 1e-8)

    stats = np.stack(
        [
            geo_margin,
            np.log1p(channel_off),
            channel_ratio,
            np.log1p(boundary_norm),
        ],
        axis=1,
    ).astype(np.float32)
    names = ["top2_geo_margin", "log1p_top2_channel_off", "top2_channel_off_ratio", "log1p_top2_boundary_norm"]
    return np.nan_to_num(stats, nan=0.0, posinf=1e6, neginf=-1e6), names


def method_delta_mixw(
    val_features,
    val_logits,
    val_targets,
    test_features,
    test_logits,
    test_probs,
    model,
    candidate_k=(3, 5, 10, 20, 30, 50, 75, 100),
):
    """MixW: pNorm plus top-two classifier-head channel off-manifold stats."""
    print("Evaluating Delta-MixW...")

    val_features = _to_numpy(val_features, dtype=np.float32)
    test_features = _to_numpy(test_features, dtype=np.float32)
    val_logits = _to_numpy(val_logits, dtype=np.float32)
    test_logits = _to_numpy(test_logits, dtype=np.float32)
    val_targets = _to_numpy(val_targets, dtype=np.int64)
    test_probs = _to_numpy(test_probs, dtype=np.float32)

    val_preds = np.argmax(val_logits, axis=1)
    val_correct = val_preds == val_targets
    split = val_features.shape[0] // 2
    if np.sum(val_correct[:split]) == 0 or np.sum(~val_correct[:split]) == 0:
        return None

    p_ref, _ = _select_best_pnorm_p(val_logits[:split], val_targets[:split], metric="aurc")
    val_logit_stats = _compute_mllr_stats(val_logits, p_choice=p_ref)
    val_channel_stats, channel_names = _top2_channel_stats(val_features, val_logits, model)
    if val_channel_stats.shape[1] == 0:
        return None

    val_ref_stats = np.concatenate([val_logit_stats[:, :2], val_channel_stats], axis=1)
    stat_names = ["pNorm", "margin"] + channel_names
    ref_raw = val_ref_stats[:split]
    calib_raw = val_ref_stats[split:]
    scaler = _clip_fit_standardizer(ref_raw)
    ref_m = _clip_apply_standardizer(ref_raw, scaler)
    calib_m = _clip_apply_standardizer(calib_raw, scaler)
    correct_bank = ref_m[val_correct[:split]]
    wrong_bank = ref_m[~val_correct[:split]]
    max_k = min(len(correct_bank), len(wrong_bank))
    k_grid = [k for k in candidate_k if k <= max_k] or [1]

    best = None
    for k in k_grid:
        conf = _knn_log_density_score(calib_m, correct_bank, k) - _knn_log_density_score(calib_m, wrong_bank, k)
        metrics = evaluate_sc_metrics(val_preds[split:], conf, val_targets[split:])
        if best is None or metrics["aurc"] < best["score"]:
            best = {"score": float(metrics["aurc"]), "k": int(k), "auroc": float(metrics["auroc"])}

    p_final, _ = _select_best_pnorm_p(val_logits, val_targets, metric="aurc")
    val_logit_stats = _compute_mllr_stats(val_logits, p_choice=p_final)
    test_logit_stats = _compute_mllr_stats(test_logits, p_choice=p_final)
    val_channel_stats, _ = _top2_channel_stats(val_features, val_logits, model)
    test_channel_stats, _ = _top2_channel_stats(test_features, test_logits, model)
    val_stats = np.concatenate([val_logit_stats[:, :2], val_channel_stats], axis=1)
    test_stats = np.concatenate([test_logit_stats[:, :2], test_channel_stats], axis=1)

    scaler = _clip_fit_standardizer(val_stats)
    val_m = _clip_apply_standardizer(val_stats, scaler)
    test_m = _clip_apply_standardizer(test_stats, scaler)
    correct_bank = val_m[val_correct]
    wrong_bank = val_m[~val_correct]
    k_final = min(best["k"], len(correct_bank), len(wrong_bank))
    test_conf = _knn_log_density_score(test_m, correct_bank, k_final) - _knn_log_density_score(test_m, wrong_bank, k_final)

    return {
        "name": "Delta-MixW (Top2 Channel LR)",
        "params": {
            "p": p_final if isinstance(p_final, str) else int(p_final),
            "p_ref": p_ref if isinstance(p_ref, str) else int(p_ref),
            "k_density": int(k_final),
            "stat_names": stat_names,
            "calib_aurc": float(best["score"]),
            "calib_auroc": float(best["auroc"]),
        },
        "test_probs": test_probs,
        "test_conf_override": test_conf.astype(np.float32),
    }


def method_trust_score(train_features, train_targets, val_features, val_probs, val_targets, test_features, test_probs, num_classes):
    """Trust Score method aligned strictly with the paper."""
    print("Evaluating Trust Score...")
    
    val_preds = np.argmax(val_probs, axis=1)
    
    candidate_k = [5, 10, 20, 30, 50]
    best = None
    
    for k in candidate_k:
        ts = TrustScore(k=k, alpha=0.0, filtering="none")
        ts.fit(train_features, train_targets)
        val_trust = ts.get_score(val_features, val_preds)
        
        metrics = evaluate_sc_metrics(val_preds, val_trust, val_targets)
        score = metrics["auroc"]
        if best is None or score > best["score"]:
            best = {"score": score, "k": k}
            
    print(f"Best k for Trust Score: {best['k']} (AUROC: {best['score']:.4f})")
    
    ts = TrustScore(k=best["k"], alpha=0.0, filtering="none")
    ts.fit(train_features, train_targets)
    
    test_preds = np.argmax(test_probs, axis=1)
    test_conf = ts.get_score(test_features, test_preds)
    
    return {
        "name": "Trust Score",
        "params": {"k": int(best["k"])},
        "test_probs": np.copy(test_probs),
        "test_conf_override": test_conf,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate post-hoc baseline methods for Mixup SC")
    
    # Build dataset choices (include Tiny-ImageNet)
    dataset_choices = ["cifar10", "cifar100"] + list(MEDMNIST_DATASETS.keys()) + [
        "tinyimagenet", "skin_cancer_isic", "chest_xray", "mri_tumor", "alzheimer",
        "tuberculosis", "sars_cov_2_ct_scan", "chest_ct_scan"
    ]
    
    parser.add_argument("--dataset", type=str, default="cifar10", choices=dataset_choices)
    parser.add_argument("--arch", type=str, default="resnet18", choices=["resnet18", "resnet34", "resnet50", "resnet101", "resnet110", "vgg16_bn", "vit_b_16", "vit_b_4", "wrn28_10", "dense", "cmixer"])
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
    elif dataset in ["skin_cancer_isic", "chest_xray", "tuberculosis", "sars_cov_2_ct_scan"]:
        num_classes = 2
    elif dataset in ["mri_tumor", "alzheimer", "chest_ct_scan"]:
        num_classes = 4
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
    if "model_state" in ckpt:
        state_dict = ckpt["model_state"]
    else:
        state_dict = ckpt
        
    # Handle DataParallel prefix if present
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if k.startswith('module.'):
            name = k[7:] # remove `module.`
        else:
            name = k
        new_state_dict[name] = v
        
    model.load_state_dict(new_state_dict)
    model.eval()

    print("Extracting training features for Train-ref methods...")
    train_data = collect_outputs(model, train_loader, device)
    val_data = collect_outputs(model, val_loader, device)
    test_data = collect_outputs(model, test_loader, device)

    # Baseline from pre-trained Mixup checkpoint.
    baseline = evaluate_sc_metrics(test_data["preds"], test_data["confs"], test_data["targets"], probs=test_data["probs"])

    # print("\n===== Phase 1: Hypotheses =====")
    # print("RQ1 (Logit calibration): Can class-wise temperatures reduce overconfidence and improve RC ranking?")
    # print("RQ2 (Latent distance): Can local neighborhood evidence in feature space refine uncertain predictions?")
    # print("RQ3 (Conformal features): Can prototype-distance conformal p-values improve coverage-risk ordering?")
    # print("RQ4 (Orthogonal projection): Can isolating Mixup channel artifacts via SOCP improve coverage-risk?")

    print("\n===== Phase 2: Execute Baseline Post-hoc Methods =====")

    val_logits_t = val_data["logits"].to(device)
    val_targets_t = torch.tensor(val_data["targets"], dtype=torch.long, device=device)
    train_logits_t = train_data["logits"].to(device)
    test_logits_t = test_data["logits"].to(device)
    val_inputs_t = val_data["inputs"] if "inputs" in val_data else None
    test_inputs_t = test_data["inputs"] if "inputs" in test_data else None

    methods = [
        # Baseline simple methods
        # method_energy(val_logits_t, val_targets_t, test_logits_t),
        # method_mahalanobis(val_data["features"], val_data["targets"], test_data["features"], test_data["probs"], num_classes),
        
        # DOCTOR methods
        method_doctor_alpha(val_data["probs"], val_data["targets"], test_data["probs"], num_classes),
        method_doctor_beta(val_data["probs"], val_data["targets"], test_data["probs"], num_classes),
        # method_trust_score(train_data["features"], train_data["targets"], val_data["features"], val_data["probs"], val_data["targets"], test_data["features"], test_data["probs"], num_classes),
        # pnorm and pnorm+temperature
        method_5_maxlogit_pnorm(val_logits_t, val_targets_t, test_logits_t),
        method_6_maxlogit_pnorm_temperature(val_logits_t, val_targets_t, test_logits_t),

        # method_delta_mllr(
        #     val_logits_t,
        #     val_data["targets"],
        #     test_logits_t,
        #     test_data["probs"],
        # ),
        # method_pnorm_feature_hybrid(
        #     train_data["features"],
        #     train_data["targets"],
        #     val_data["features"],
        #     val_logits_t,
        #     val_data["targets"],
        #     test_data["features"],
        #     test_logits_t,
        #     test_data["probs"],
        #     num_classes,
        #     model,
        # ),
        # method_delta_rowspace_knn(
        #     val_data["features"],
        #     val_logits_t,
        #     val_data["targets"],
        #     test_data["features"],
        #     test_data["probs"],
        #     model,
        # ),
        # method_delta_nullspace_knn(
        #     val_data["features"],
        #     val_logits_t,
        #     val_data["targets"],
        #     test_data["features"],
        #     test_data["probs"],
        #     model,
        # ),
        # method_delta_rnlr_statspace(
        #     val_data["features"],
        #     val_logits_t,
        #     val_data["targets"],
        #     test_data["features"],
        #     test_logits_t,
        #     test_data["probs"],
        #     model,
        # ),
        # method_delta_mixw(
        #     val_data["features"],
        #     val_logits_t,
        #     val_data["targets"],
        #     test_data["features"],
        #     test_logits_t,
        #     test_data["probs"],
        #     model,
        # ),
        # method_scp(train_data["features"], train_data["targets"], test_data["features"], test_data["probs"], num_classes),
        method_delta_mclr(
            train_data["features"],
            train_data["targets"],
            val_data["features"],
            val_logits_t,
            val_data["targets"],
            test_data["features"],
            test_logits_t,
            test_data["probs"],
            num_classes,
        ),

        # method_residual_mllr_logistic(
        #     val_logits_t,
        #     val_data["targets"],
        #     test_logits_t,
        #     test_data["probs"],
        # ),

        # pNorm calibration fixes
        # method_pnorm_softmax(val_logits_t, val_targets_t, test_logits_t),
        # method_pnorm_platt(val_logits_t, val_targets_t, test_logits_t),
        # method_pnorm_isotonic(val_logits_t, val_targets_t, test_logits_t),
        # method_pnorm_blend(val_logits_t, val_targets_t, test_logits_t),

        # ODIN
        # method_odin(model, test_inputs_t, device) if test_inputs_t is not None else None,

        # kNN-OOD (Sun et al. ICML 2022)
        # method_knn_ood(val_data["features"], test_data["features"], test_data["probs"], k=50),

        # Selective Classification Under Distribution Shifts (Liang et al. 2024)
        method_rl_conf_m(test_logits_t, test_data["probs"]),
        method_rl_geo_m(test_logits_t, test_data["probs"], model),
        
        # SR_ent and SIRC
        method_sr_ent(test_data["probs"]),
        *method_vim_and_sirc(
            torch.from_numpy(train_data["features"]).float().to(device),
            train_logits_t,
            torch.from_numpy(test_data["features"]).float().to(device),
            test_logits_t,
            test_data["probs"],
            model,
        ),

        # Feature-space kNN blending (Grid Search)
        method_2_feature_knn_logit_blend(
            val_data["features"],
            val_data["probs"],
            val_data["targets"],
            test_data["features"],
            test_data["probs"],
            num_classes,
        ),


        # method_spherical_feature_knn(
        #     val_data["features"],
        #     val_data["probs"],
        #     val_data["targets"],
        #     test_data["features"],
        #     test_data["probs"],
        #     num_classes,
        #     fixed_alpha=1.0,
        # ),

        # Other Classifiers
        # method_feature_classifier_blend(
        #     val_data["features"],
        #     val_data["probs"],
        #     val_data["targets"],
        #     test_data["features"],
        #     test_data["probs"],
        #     num_classes,
        #     classifier_type="logreg",
        # ),
        # method_feature_classifier_blend(
        #     val_data["features"],
        #     val_data["probs"],
        #     val_data["targets"],
        #     test_data["features"],
        #     test_data["probs"],
        #     num_classes,
        #     classifier_type="svm_linear",
        # ),
        # method_feature_classifier_blend(
        #     val_data["features"],
        #     val_data["probs"],
        #     val_data["targets"],
        #     test_data["features"],
        #     test_data["probs"],
        #     num_classes,
        #     classifier_type="svm_rbf",
        # ),
        
        # Combinations
        # method_spherical_logit_knn_blend(
        #     val_logits_t,
        #     val_data["targets"],
        #     test_logits_t,
        #     val_data["probs"],
        #     test_data["probs"],
        #     num_classes,
        # ),
        # method_combo_pnorm_sknn(
        #     val_logits_t,
        #     val_data["targets"],
        #     test_logits_t,
        #     val_data["features"],
        #     val_data["probs"],
        #     test_data["features"],
        #     test_data["probs"],
        #     num_classes,
        # ),
        
        # Advanced Engineering of Feature Space
        # method_affine_shift_knn(
        #     val_data["features"],
        #     val_data["probs"],
        #     val_data["targets"],
        #     test_data["features"],
        #     test_data["probs"],
        #     num_classes,
        # ),
        # method_virtual_flat_minima(
        #     val_data["features"],
        #     val_data["probs"],
        #     val_data["targets"],
        #     test_data["features"],
        #     test_data["probs"],
        #     num_classes,
        # ),
        # method_simplex_orthogonal_knn(
        #     val_data["features"],
        #     val_data["probs"],
        #     val_data["targets"],
        #     test_data["features"],
        #     test_data["probs"],
        #     num_classes,
        #     model,
        # ),

    ]


    
    # Remove None entries (if ODIN is not available)
    methods = [m for m in methods if m is not None]

    print("\n===== Phase 3: Evaluate (Acc, AUROC, AURC, E-AURC, ECE, NLL, Brier) =====")
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
        metrics = evaluate_sc_metrics(preds, confs, test_data["targets"], probs=probs)
        rows.append({"method": method["name"], **metrics, "params": method["params"]})

    # Comparative summary sorted by AURC (lower better).
    baseline_naurc = rows[0]["naurc"]
    improved = [r for r in rows[1:] if r["naurc"] < baseline_naurc]

    print("\nComparative Summary (Test):")
    print(f"{'Method':50s} {'Acc':>8s} {'AUROC':>8s} {'AURC':>8s} {'E-AURC':>8s} {'NAURC':>8s} {'ECE':>8s} {'NLL':>8s} {'Brier':>8s}")
    print("-" * 123)
    for r in rows:
        print(f"{r['method'][:50]:50s} {r['accuracy']:8.4f} {r['auroc']:8.4f} {r['aurc']:8.4f} {r['eaurc']:8.4f} {r['naurc']:8.4f} {r['ece']:8.4f} {r['nll']:8.4f} {r['brier']:8.4f}")

    if improved:
        best = sorted(improved, key=lambda x: x["naurc"])[0]
        conclusion = (
            f"Primary candidate: {best['method']} (NAURC {best['naurc']:.4f} vs baseline {baseline_naurc:.4f})."
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
