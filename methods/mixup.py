"""Mixup training method."""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .base import BaseMethod


class MixupMethod(BaseMethod):
    """Standard Mixup for data augmentation."""

    def __init__(self, model: torch.nn.Module, alpha: float = 1.0, p: float = 1.0):
        """
        Initialize Mixup.

        Args:
            model: Neural network model
            alpha: Beta distribution parameter (higher = more diverse mixing)
            p: Probability of applying mixup
        """
        super().__init__(model)
        self.alpha = alpha
        self.p = p

    def _infer_num_classes(self, targets: torch.Tensor) -> int:
        """Infer model output classes for one-hot target construction."""
        if hasattr(self.model, "num_classes"):
            return int(self.model.num_classes)

        # Common classifier heads.
        if hasattr(self.model, "fc") and hasattr(self.model.fc, "out_features"):
            return int(self.model.fc.out_features)
            
        if hasattr(self.model, "head") and hasattr(self.model.head, "out_features"):
            return int(self.model.head.out_features)

        if hasattr(self.model, "classifier"):
            classifier = self.model.classifier
            if hasattr(classifier, "out_features"):
                return int(classifier.out_features)
            if isinstance(classifier, nn.Sequential):
                for layer in reversed(classifier):
                    if hasattr(layer, "out_features"):
                        return int(layer.out_features)

        # Fallback for unknown architectures.
        return int(targets.max().item() + 1)

    def mixup_batch(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
    ) -> tuple:
        """
        Apply mixup to a batch.

        Args:
            inputs: Input batch [N, C, H, W]
            targets: Target labels [N] or [N, 1] or [N, ...] (will be squeezed)

        Returns:
            tuple: (mixed_inputs, mixed_targets, lam)
        """
        batch_size = inputs.size(0)

        # Ensure targets are 1D [N] by squeezing all dimensions after the first
        # This handles [N], [N, 1], and other edge cases
        original_targets = targets
        targets = targets.view(batch_size, -1)[:, 0]  # Take first column and ensure [N]
        targets = targets.long()

        # Randomly decide whether to apply mixup
        if np.random.rand() > self.p:
            return inputs, original_targets, None

        # Sample lambda from Beta distribution
        lam = np.random.beta(self.alpha, self.alpha)

        # Random permutation for mixing
        index = torch.randperm(batch_size).to(inputs.device)

        # Mix inputs
        mixed_inputs = lam * inputs + (1 - lam) * inputs[index, :]

        # Convert targets to one-hot for soft label mixing
        num_classes = self._infer_num_classes(targets)
        targets_onehot = F.one_hot(targets, num_classes=num_classes).float()
        targets_index_onehot = F.one_hot(targets[index], num_classes=num_classes).float()

        # Mix targets
        mixed_targets = lam * targets_onehot + (1 - lam) * targets_index_onehot

        return mixed_inputs, mixed_targets, lam

    def forward_and_loss(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        use_mixup: bool = True,
    ) -> tuple:
        """
        Forward pass with Mixup.

        Args:
            inputs: Input batch [N, C, H, W]
            targets: Target labels [N]
            use_mixup: Whether to apply mixup in this batch

        Returns:
            tuple: (logits, loss)
        """
        if use_mixup:
            mixed_inputs, mixed_targets, lam = self.mixup_batch(inputs, targets)
        else:
            mixed_inputs = inputs
            mixed_targets = targets
            lam = None

        logits = self.model(mixed_inputs)

        # Compute loss with mixed targets
        if lam is not None:
            # Soft label loss (targets are probabilities)
            loss = -(mixed_targets * F.log_softmax(logits, dim=1)).sum(dim=1).mean()
        else:
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def compute_accuracy(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> float:
        """Compute accuracy (targets should be hard labels for accuracy)."""
        _, predictions = torch.max(logits, 1)
        
        # Ensure targets are 1D by taking first column
        targets_1d = targets.view(logits.size(0), -1)[:, 0].long()
        
        correct = (predictions == targets_1d).sum().item()
        accuracy = correct / logits.size(0)
        return accuracy


class MixupVariant1(MixupMethod):
    """
    Mixup variant: Temperature scaling for confidence distribution.

    Applies temperature scaling after mixup to control confidence.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        alpha: float = 1.0,
        p: float = 1.0,
        temperature: float = 1.0,
    ):
        super().__init__(model, alpha=alpha, p=p)
        self.temperature = temperature

    def forward_and_loss(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        use_mixup: bool = True,
    ) -> tuple:
        """Forward pass with Mixup and temperature scaling."""
        if use_mixup:
            mixed_inputs, mixed_targets, lam = self.mixup_batch(inputs, targets)
        else:
            mixed_inputs = inputs
            mixed_targets = targets
            lam = None

        logits = self.model(mixed_inputs)

        # Apply temperature scaling
        logits = logits / self.temperature

        # Compute loss
        if lam is not None:
            loss = -(mixed_targets * F.log_softmax(logits, dim=1)).sum(dim=1).mean()
        else:
            loss = F.cross_entropy(logits, targets)

        return logits, loss


class MixupVariant2(MixupMethod):
    """
    Mixup variant: Adaptive alpha based on confidence.

    Alpha is adjusted based on model's confidence distribution.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        base_alpha: float = 1.0,
        alpha: float = None,
        p: float = 1.0,
    ):
        # Keep compatibility with generic factory passing `alpha`.
        if alpha is not None:
            base_alpha = alpha
        super().__init__(model, alpha=base_alpha, p=p)
        self.base_alpha = base_alpha
        self.confidence_history = []

    def forward_and_loss(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        use_mixup: bool = True,
    ) -> tuple:
        """Forward pass with adaptive mixup alpha."""
        if not use_mixup:
            logits = self.model(inputs)
            loss = F.cross_entropy(logits, targets)
            return logits, loss

        # Compute current confidence distribution
        with torch.no_grad():
            logits_temp = self.model(inputs)
            probs = F.softmax(logits_temp, dim=1)
            confidences = probs.max(dim=1)[0]
            mean_conf = confidences.mean().item()
            self.confidence_history.append(mean_conf)

        # Adapt alpha based on confidence (lower confidence -> higher mixup strength)
        confidence_ratio = min(1.0, mean_conf / 0.8)  # Normalize to [0, 1]
        adaptive_alpha = self.base_alpha * (2.0 - confidence_ratio)  # Range: [alpha, 2*alpha]
        self.alpha = adaptive_alpha

        # Apply mixup with adaptive alpha
        mixed_inputs, mixed_targets, lam = self.mixup_batch(inputs, targets)
        logits = self.model(mixed_inputs)

        if lam is not None:
            loss = -(mixed_targets * F.log_softmax(logits, dim=1)).sum(dim=1).mean()
        else:
            loss = F.cross_entropy(logits, targets)

        return logits, loss
