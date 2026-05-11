"""Base training method."""
import torch
import torch.nn.functional as F


class BaseMethod:
    """Base class for training methods."""

    def __init__(self, model: torch.nn.Module):
        self.model = model

    def forward_and_loss(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
    ) -> tuple:
        """
        Forward pass and compute loss.

        Args:
            inputs: Input batch [N, C, H, W]
            targets: Target labels [N]

        Returns:
            tuple: (logits, loss)
        """
        logits = self.model(inputs)
        loss = F.cross_entropy(logits, targets)
        return logits, loss

    def compute_accuracy(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> float:
        """Compute classification accuracy."""
        _, predictions = torch.max(logits, 1)
        correct = (predictions == targets).sum().item()
        accuracy = correct / targets.size(0)
        return accuracy


class StandardMethod(BaseMethod):
    """Standard training without data augmentation/regularization."""

    def forward_and_loss(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
    ) -> tuple:
        """Standard forward pass."""
        return super().forward_and_loss(inputs, targets)
