"""Main training loop and trainer class."""
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader
import numpy as np
from typing import Optional, Dict, Tuple
import os

from evaluation import (
    SelectionMetrics,
    CalibrationMetrics,
    compute_metrics_from_logits,
    compute_probs_from_logits,
)
from utils import ExperimentLogger


class Trainer:
    """Main trainer class for training and evaluation."""

    def __init__(
        self,
        model: torch.nn.Module,
        method,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: DataLoader,
        device: torch.device,
        config,
        logger: Optional[ExperimentLogger] = None,
    ):
        """
        Initialize trainer.

        Args:
            model: Neural network model
            method: Training method (standard, mixup, etc.)
            train_loader: Training dataloader
            val_loader: Validation dataloader
            test_loader: Test dataloader
            device: GPU/CPU device
            config: Configuration object
            logger: Experiment logger
        """
        self.model = model.to(device)
        self.method = method
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = device
        self.config = config
        self.logger = logger
        self.coverage_levels = config.evaluation.coverage_levels

        # Setup optimizer
        if config.training.optimizer.lower() == "adam":
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=config.training.learning_rate,
                weight_decay=config.training.weight_decay,
            )
        elif config.training.optimizer.lower() == "adamw":
            self.optimizer = optim.AdamW(
                self.model.parameters(),
                lr=config.training.learning_rate,
                weight_decay=config.training.weight_decay,
            )
        else:
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=config.training.learning_rate,
                momentum=config.training.momentum,
                weight_decay=config.training.weight_decay,
            )

        # Setup learning rate scheduler
        if config.training.lr_schedule == "cosine":
            if getattr(config.training, "warmup_epochs", 0) > 0:
                warmup = lr_scheduler.LinearLR(
                    self.optimizer, start_factor=0.01, end_factor=1.0, total_iters=config.training.warmup_epochs
                )
                cosine = lr_scheduler.CosineAnnealingLR(
                    self.optimizer, T_max=config.training.epochs - config.training.warmup_epochs
                )
                self.scheduler = lr_scheduler.SequentialLR(
                    self.optimizer, schedulers=[warmup, cosine], milestones=[config.training.warmup_epochs]
                )
            else:
                self.scheduler = lr_scheduler.CosineAnnealingLR(
                    self.optimizer, T_max=config.training.epochs
                )
        elif config.training.lr_schedule == "step":
            self.scheduler = lr_scheduler.StepLR(
                self.optimizer,
                step_size=config.training.epochs // 3,
                gamma=0.1,
            )
        elif config.training.lr_schedule == "exponential":
            self.scheduler = lr_scheduler.ExponentialLR(self.optimizer, gamma=0.95)
        else:
            self.scheduler = None

        # Training state
        self.current_epoch = 0
        self.best_val_auroc = -np.inf
        self.best_val_loss = np.inf

    def train_epoch(self) -> Dict[str, float]:
        """
        Execute one training epoch.

        Returns:
            Dictionary with training metrics
        """
        self.model.train()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        all_confidences = []
        all_correctness = []

        for batch_idx, (inputs, targets) in enumerate(self.train_loader):
            inputs, targets = inputs.to(self.device), targets.to(self.device)

            # Forward pass and loss
            logits, loss = self.method.forward_and_loss(inputs, targets, use_mixup=True)

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Accumulate metrics
            total_loss += loss.item()
            acc = self.method.compute_accuracy(logits, targets)
            total_correct += int(acc * targets.size(0))
            total_samples += targets.size(0)

            # Collect confidence and correctness for SC metrics
            confidences, correctness = compute_metrics_from_logits(logits, targets)
            # Ensure they are 1D arrays before extending
            all_confidences.extend(np.atleast_1d(confidences).flatten())
            all_correctness.extend(np.atleast_1d(correctness).flatten())

            # Log batch metrics
            if (batch_idx + 1) % self.config.logging.log_frequency == 0:
                batch_loss = total_loss / (batch_idx + 1)
                batch_acc = total_correct / total_samples
                if self.logger:
                    metrics = {
                        "train/batch_loss": batch_loss,
                        "train/batch_accuracy": batch_acc,
                        "train/learning_rate": self.optimizer.param_groups[0]["lr"],
                    }
                    self.logger.log_metrics(metrics, step=self.current_epoch)

        # Epoch statistics
        epoch_loss = total_loss / len(self.train_loader)
        epoch_acc = total_correct / total_samples
        all_confidences = np.array(all_confidences, dtype=np.float32)
        all_correctness = np.array(all_correctness, dtype=np.int32)

        # Compute SC metrics
        sc_metrics = SelectionMetrics.compute_all_metrics(
            all_confidences, all_correctness, self.coverage_levels
        )

        return {
            "loss": epoch_loss,
            "accuracy": epoch_acc,
            **sc_metrics,
        }

    def validate_epoch(self) -> Dict[str, float]:
        """
        Execute validation.

        Returns:
            Dictionary with validation metrics
        """
        self.model.eval()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        all_logits = []
        all_targets = []

        with torch.no_grad():
            for inputs, targets in self.val_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                logits, loss = self.method.forward_and_loss(inputs, targets, use_mixup=False)
                total_loss += loss.item()

                acc = self.method.compute_accuracy(logits, targets)
                total_correct += int(acc * targets.size(0))
                total_samples += targets.size(0)

                all_logits.append(logits.cpu())
                all_targets.append(targets.cpu())

        # Compute metrics
        epoch_loss = total_loss / len(self.val_loader)
        epoch_acc = total_correct / total_samples

        all_logits = torch.cat(all_logits, dim=0)
        all_targets = torch.cat(all_targets, dim=0)

        confidences, correctness = compute_metrics_from_logits(all_logits, all_targets)
        probs = compute_probs_from_logits(all_logits)

        sc_metrics = SelectionMetrics.compute_all_metrics(
            confidences, correctness, self.coverage_levels
        )

        cal_metrics = CalibrationMetrics.compute_all_calibration_metrics(
            confidences, correctness, probs, all_targets.numpy()
        )

        conf_stats = SelectionMetrics.compute_confidence_distribution(confidences)

        return {
            "loss": epoch_loss,
            "accuracy": epoch_acc,
            **sc_metrics,
            **cal_metrics,
            **conf_stats,
        }

    def test_epoch(self) -> Dict[str, float]:
        """
        Execute testing.

        Returns:
            Dictionary with test metrics
        """
        self.model.eval()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        all_logits = []
        all_targets = []

        with torch.no_grad():
            for inputs, targets in self.test_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                logits, loss = self.method.forward_and_loss(inputs, targets, use_mixup=False)
                total_loss += loss.item()

                acc = self.method.compute_accuracy(logits, targets)
                total_correct += int(acc * targets.size(0))
                total_samples += targets.size(0)

                all_logits.append(logits.cpu())
                all_targets.append(targets.cpu())

        # Compute metrics
        epoch_loss = total_loss / len(self.test_loader)
        epoch_acc = total_correct / total_samples

        all_logits = torch.cat(all_logits, dim=0)
        all_targets = torch.cat(all_targets, dim=0)

        confidences, correctness = compute_metrics_from_logits(all_logits, all_targets)
        probs = compute_probs_from_logits(all_logits)

        sc_metrics = SelectionMetrics.compute_all_metrics(
            confidences, correctness, self.coverage_levels
        )

        cal_metrics = CalibrationMetrics.compute_all_calibration_metrics(
            confidences, correctness, probs, all_targets.numpy()
        )

        conf_stats = SelectionMetrics.compute_confidence_distribution(confidences)

        return {
            "loss": epoch_loss,
            "accuracy": epoch_acc,
            **sc_metrics,
            **cal_metrics,
            **conf_stats,
        }

    def train(self):
        """Execute complete training loop."""
        print(f"\n=== Starting training on {self.device} ===")
        print(f"Model: {self.config.model.arch}, Dataset: {self.config.data.dataset}")
        print(f"Method: {self.config.method.name}\n")

        for epoch in range(self.config.training.epochs):
            self.current_epoch = epoch

            # Training
            train_metrics = self.train_epoch()
            print(
                f"Epoch {epoch + 1}/{self.config.training.epochs} | "
                f"Train Loss: {train_metrics['loss']:.4f} | "
                f"Train Acc: {train_metrics['accuracy']:.4f} | "
                f"Train AUROC: {train_metrics['auroc']:.4f}"
            )

            # Validation
            if (epoch + 1) % self.config.evaluation.val_frequency == 0:
                val_metrics = self.validate_epoch()
                print(
                    f"  Val Loss: {val_metrics['loss']:.4f} | "
                    f"Val Acc: {val_metrics['accuracy']:.4f} | "
                    f"Val AUROC: {val_metrics['auroc']:.4f} | "
                    f"Val AURC: {val_metrics['aurc']:.4f} | "
                    f"Val ECE: {val_metrics['ece']:.4f} | "
                    f"Val Brier: {val_metrics['brier']:.4f}"
                )

                # Log to wandb/tensorboard
                if self.logger:
                    metrics = {}
                    for key, val in train_metrics.items():
                        metrics[f"train/{key}"] = val
                    for key, val in val_metrics.items():
                        metrics[f"val/{key}"] = val

                    self.logger.log_metrics(metrics, step=epoch)

                    # Log confidence histogram
                    if self.config.logging.use_wandb:
                        # Extract confidences again for histogram (slightly inefficient but ok)
                        all_logits = []
                        all_targets = []
                        with torch.no_grad():
                            for inputs, targets in self.val_loader:
                                inputs, targets = inputs.to(self.device), targets.to(self.device)
                                logits, _ = self.method.forward_and_loss(
                                    inputs, targets, use_mixup=False
                                )
                                all_logits.append(logits.cpu())
                                all_targets.append(targets.cpu())

                        all_logits = torch.cat(all_logits, dim=0)
                        all_targets = torch.cat(all_targets, dim=0)
                        self.logger.log_histogram("val/logits_distribution", all_logits, epoch)

                # Early stopping and model saving
                if val_metrics["auroc"] > self.best_val_auroc:
                    self.best_val_auroc = val_metrics["auroc"]
                    self.save_checkpoint("best_auroc", epoch)

            # Learning rate schedule
            if self.scheduler:
                self.scheduler.step()

        # Final test evaluation
        print(f"\n=== Final Test Evaluation ===")
        test_metrics = self.test_epoch()
        print(f"Test Loss: {test_metrics['loss']:.4f}")
        print(f"Test Accuracy: {test_metrics['accuracy']:.4f}")
        print(f"Test AUROC: {test_metrics['auroc']:.4f}")
        print(f"Test AURC: {test_metrics['aurc']:.4f}")
        print(f"Test E-AURC: {test_metrics['eaurc']:.4f}")
        print(f"Test ECE: {test_metrics['ece']:.4f}")
        print(f"Test MCE: {test_metrics['mce']:.4f}")
        print(f"Test Brier: {test_metrics['brier']:.4f}")
        print(f"Test NLL: {test_metrics['nll']:.4f}")

        if self.logger:
            metrics = {f"test/{key}": val for key, val in test_metrics.items()}
            self.logger.log_metrics(metrics, step=self.config.training.epochs)
            self.logger.finish()

        return test_metrics

    def save_checkpoint(self, name: str = "latest", epoch: int = 0):
        """Save model checkpoint."""
        os.makedirs("./checkpoints", exist_ok=True)
        path = f"./checkpoints/{self.config.logging.experiment_name}_{name}_epoch_{epoch}.pt"
        torch.save(
            {
                "epoch": epoch,
                "model_state": self.model.state_dict(),
                "optimizer_state": self.optimizer.state_dict(),
                "config": self.config.to_dict(),
            },
            path,
        )
        print(f"Checkpoint saved: {path}")

    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state"])
        self.current_epoch = checkpoint["epoch"]
        print(f"Checkpoint loaded: {path}")
