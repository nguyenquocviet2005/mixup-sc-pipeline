"""Main entry point for training experiments."""
import argparse
import torch
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils import Config, get_device, setup_logging
from data import get_dataloaders, MEDMNIST_DATASETS
from models import get_model
from methods import get_method
from training import Trainer


def main(args):
    """Main training function."""
    # Load configuration
    if args.config:
        config = Config.from_yaml(args.config)
    else:
        config = Config(
            data=__import__("utils", fromlist=["DataConfig"]).DataConfig(),
            model=__import__("utils", fromlist=["ModelConfig"]).ModelConfig(),
            method=__import__("utils", fromlist=["MethodConfig"]).MethodConfig(),
            training=__import__("utils", fromlist=["TrainingConfig"]).TrainingConfig(),
            logging=__import__("utils", fromlist=["LoggingConfig"]).LoggingConfig(),
            evaluation=__import__("utils", fromlist=["EvaluationConfig"]).EvaluationConfig(),
        )

    # Override with command line arguments
    if args.dataset:
        config.data.dataset = args.dataset
    if args.method:
        config.method.name = args.method
    if args.epochs:
        config.training.epochs = args.epochs
    if args.batch_size:
        config.data.batch_size = args.batch_size
    if args.learning_rate:
        config.training.learning_rate = args.learning_rate
    if args.arch:
        config.model.arch = args.arch
    if args.no_wandb:
        config.logging.use_wandb = False

    # Auto-set num_classes based on dataset
    if config.data.dataset == "cifar10":
        config.model.num_classes = 10
    elif config.data.dataset == "cifar100":
        config.model.num_classes = 100
    elif config.data.dataset.lower() == "tinyimagenet":
        config.model.num_classes = 200
    elif config.data.dataset.lower() in MEDMNIST_DATASETS:
        config.model.num_classes = MEDMNIST_DATASETS[config.data.dataset.lower()]["num_classes"]

    # Create experiment name
    if not args.exp_name:
        config.logging.experiment_name = (
            f"{config.method.name}_{config.data.dataset}_{config.model.arch}"
        )
    else:
        config.logging.experiment_name = args.exp_name

    print(f"Configuration:")
    print(config.to_dict())

    # Setup device
    device = get_device()

    # Load data
    print(f"\nLoading {config.data.dataset.upper()} dataset...")
    train_loader, val_loader, test_loader, num_classes = get_dataloaders(
        dataset=config.data.dataset,
        data_dir=config.data.data_dir,
        batch_size=config.data.batch_size,
        val_batch_size=config.data.val_batch_size,
        num_workers=config.data.num_workers,
        augmentation=config.data.augmentation,
        seed=config.training.seed,
    )
    config.model.num_classes = num_classes
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Val samples: {len(val_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")

    # Build model
    print(f"\nBuilding model: {config.model.arch}...")
    input_size = 64 if config.data.dataset.lower() == "tinyimagenet" else 32
    model = get_model(config.model.arch, num_classes=config.model.num_classes, input_size=input_size)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

    # Setup training method
    print(f"\nSetup training method: {config.method.name}...")
    method = get_method(
        config.method.name,
        model,
        alpha=config.method.mixup_alpha,
        p=config.method.prob,
    )

    # Setup logger
    logger = setup_logging(config) if config.logging.use_wandb or config.logging.use_tensorboard else None

    # Create trainer
    trainer = Trainer(
        model=model,
        method=method,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        device=device,
        config=config,
        logger=logger,
    )

    # Save config
    config_dir = Path("./experiments/configs")
    config_dir.mkdir(parents=True, exist_ok=True)
    config.save_yaml(config_dir / f"{config.logging.experiment_name}.yaml")

    # Train
    test_metrics = trainer.train()

    return test_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train Mixup variants for Selective Classification"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML config file",
    )
    
    # Build dataset choices
    dataset_choices = ["cifar10", "cifar100", "tinyimagenet"] + list(MEDMNIST_DATASETS.keys())
    
    parser.add_argument(
        "--dataset",
        type=str,
        choices=dataset_choices,
        help="Dataset to use (CIFAR: cifar10, cifar100; Tiny-ImageNet: tinyimagenet; MedMNIST: pathmnist, chestmnist, dermamnist, etc.)",
    )
    parser.add_argument(
        "--method",
        type=str,
        choices=["standard", "mixup", "mixup_variant1", "mixup_variant2"],
        help="Training method",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        help="Training batch size",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        help="Learning rate",
    )
    parser.add_argument(
        "--arch",
        type=str,
        choices=["resnet18", "resnet34", "resnet50", "resnet101", "vit_b_16", "vit_b_4", "vgg16_bn"],
        help="Backbone architecture",
    )
    parser.add_argument(
        "--exp-name",
        type=str,
        default=None,
        help="Experiment name",
    )
    parser.add_argument(
        "--no-wandb",
        action="store_true",
        help="Disable W&B logging",
    )

    args = parser.parse_args()

    main(args)
