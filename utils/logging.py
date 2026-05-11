"""Logging and experiment tracking utilities."""
import os
from typing import Optional
import torch
import wandb


class ExperimentLogger:
    """Unified logging interface for W&B and/or TensorBoard."""

    def __init__(
        self,
        use_wandb: bool = True,
        use_tensorboard: bool = False,
        project_name: str = "mixup-sc",
        experiment_name: str = "default",
        config_dict: Optional[dict] = None,
    ):
        self.use_wandb = use_wandb
        self.use_tensorboard = use_tensorboard
        self.config_dict = config_dict or {}

        if self.use_wandb:
            wandb.init(
                project=project_name,
                name=experiment_name,
                config=self.config_dict,
                save_code=True,
            )

        if self.use_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter
                self.writer = SummaryWriter(log_dir=f"./runs/{experiment_name}")
            except ImportError:
                print("TensorBoard not installed. Skipping TensorBoard logging.")
                self.use_tensorboard = False

    def log_metrics(self, metrics: dict, step: int):
        """Log metrics to all enabled logging systems."""
        if self.use_wandb:
            wandb.log(metrics, step=step)

        if self.use_tensorboard:
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    self.writer.add_scalar(key, value, step)

    def log_histogram(self, name: str, values: torch.Tensor, step: int):
        """Log histogram."""
        if self.use_wandb:
            wandb.log({name: wandb.Histogram(values.detach().cpu().numpy())}, step=step)

        if self.use_tensorboard:
            self.writer.add_histogram(name, values, global_step=step)

    def log_model_checkpoint(self, model: torch.nn.Module, name: str = "model"):
        """Save and log model checkpoint."""
        if not os.path.exists("./checkpoints"):
            os.makedirs("./checkpoints")

        path = f"./checkpoints/{name}.pt"
        torch.save(model.state_dict(), path)

        if self.use_wandb:
            wandb.save(path)

    def log_artifact(self, path: str, artifact_type: str = "model"):
        """Log artifact to W&B."""
        if self.use_wandb:
            artifact = wandb.Artifact(os.path.basename(path), type=artifact_type)
            artifact.add_file(path)
            wandb.log_artifact(artifact)

    def finish(self):
        """Finalize logging."""
        if self.use_wandb:
            wandb.finish()

        if self.use_tensorboard:
            self.writer.close()


def setup_logging(config):
    """Initialize logger from config."""
    return ExperimentLogger(
        use_wandb=config.logging.use_wandb,
        use_tensorboard=config.logging.use_tensorboard,
        project_name=config.logging.project_name,
        experiment_name=config.logging.experiment_name,
        config_dict=config.to_dict(),
    )
