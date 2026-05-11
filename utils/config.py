"""Configuration management."""
from dataclasses import dataclass, asdict
from typing import Optional
import json
import yaml


@dataclass
class DataConfig:
    """Data configuration."""
    dataset: str = "cifar10"  # cifar10 or cifar100
    data_dir: str = "./data"
    batch_size: int = 128
    val_batch_size: int = 256
    num_workers: int = 4
    augmentation: bool = True


@dataclass
class ModelConfig:
    """Model configuration."""
    arch: str = "resnet50"  # resnet18, resnet34, resnet50
    num_classes: Optional[int] = None  # Auto-set based on dataset
    pretrained: bool = False


@dataclass
class MethodConfig:
    """Training method configuration."""
    name: str = "standard"  # standard, mixup, mixup_variant1, etc.
    mixup_alpha: float = 1.0  # Beta distribution parameter for mixup
    cutmix_alpha: float = 1.0  # For cutmix if used
    prob: float = 1.0  # Probability of applying mixup


@dataclass
class TrainingConfig:
    """Training configuration."""
    epochs: int = 200
    learning_rate: float = 0.1
    momentum: float = 0.9
    weight_decay: float = 5e-4
    optimizer: str = "sgd"  # sgd or adam
    lr_schedule: str = "cosine"  # cosine, step, exponential
    warmup_epochs: int = 5
    seed: int = 42


@dataclass
class LoggingConfig:
    """Logging configuration."""
    use_wandb: bool = True
    use_tensorboard: bool = False
    project_name: str = "mixup-sc"
    experiment_name: str = "default"
    log_frequency: int = 100  # Log every N batches
    save_frequency: int = 10  # Save checkpoint every N epochs


@dataclass
class EvaluationConfig:
    """Evaluation configuration."""
    compute_auroc: bool = True
    compute_aurc: bool = True
    compute_eaurc: bool = True
    val_frequency: int = 1  # Validate every N epochs
    coverage_levels: list = None  # Coverage levels for AURC (0.0 to 1.0)

    def __post_init__(self):
        if self.coverage_levels is None:
            self.coverage_levels = list(range(0, 101, 5))  # 0-100%


@dataclass
class Config:
    """Master configuration."""
    data: DataConfig
    model: ModelConfig
    method: MethodConfig
    training: TrainingConfig
    logging: LoggingConfig
    evaluation: EvaluationConfig

    @staticmethod
    def from_yaml(path: str) -> "Config":
        """Load config from YAML file."""
        with open(path, "r") as f:
            cfg_dict = yaml.safe_load(f)
        
        return Config(
            data=DataConfig(**cfg_dict.get("data", {})),
            model=ModelConfig(**cfg_dict.get("model", {})),
            method=MethodConfig(**cfg_dict.get("method", {})),
            training=TrainingConfig(**cfg_dict.get("training", {})),
            logging=LoggingConfig(**cfg_dict.get("logging", {})),
            evaluation=EvaluationConfig(**cfg_dict.get("evaluation", {})),
        )

    def to_dict(self) -> dict:
        """Convert config to dictionary."""
        return {
            "data": asdict(self.data),
            "model": asdict(self.model),
            "method": asdict(self.method),
            "training": asdict(self.training),
            "logging": asdict(self.logging),
            "evaluation": asdict(self.evaluation),
        }

    def save_yaml(self, path: str):
        """Save config to YAML file."""
        with open(path, "w") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)
