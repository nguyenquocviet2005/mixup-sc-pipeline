"""Data loading utilities for CIFAR-10/100 and MedMNIST datasets."""
import os
import torch
from torch.utils.data import DataLoader, Subset, Dataset
from torchvision import datasets, transforms
import numpy as np
from pathlib import Path


class SingleOrMultiLabelConverter(Dataset):
    """Wraps a dataset to convert targets to proper format (scalar for single-label or class index for multi-label)."""
    def __init__(self, dataset):
        """
        Args:
            dataset: The underlying dataset
        """
        self.dataset = dataset
        
        # Determine if multi-label by checking first sample
        sample_target = dataset[0][1]
        if isinstance(sample_target, np.ndarray):
            self.is_multilabel = sample_target.shape[0] > 1
        else:  # torch.Tensor
            self.is_multilabel = sample_target.shape[0] > 1
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        img, target = self.dataset[idx]
        
        if self.is_multilabel:
            # Multi-label: pick first positive label
            if isinstance(target, np.ndarray):
                label_indices = np.where(target > 0)[0]
                label = int(label_indices[0]) if len(label_indices) > 0 else 0
            else:  # torch.Tensor
                label_indices = torch.where(target > 0)[0]
                label = int(label_indices[0].item()) if len(label_indices) > 0 else 0
        else:
            # Single-label: extract from shape (1,) or return as is
            if isinstance(target, np.ndarray):
                label = int(target.item()) if target.shape[0] == 1 else int(target)
            else:  # torch.Tensor
                label = int(target.item() if target.shape[0] == 1 else target)
        
        return img, label


def custom_collate_fn(batch):
    """Custom collate function that ensures targets are 1D tensors."""
    imgs, targets = zip(*batch)
    imgs = torch.stack(imgs)
    targets = torch.tensor(targets, dtype=torch.long)  # Ensure 1D
    return imgs, targets


class GrayscaleTo3Channel:
    """Convert single-channel grayscale images to 3-channel RGB by replication."""
    def __call__(self, img):
        if isinstance(img, torch.Tensor):
            # If already tensor with 1 channel, replicate
            if img.shape[0] == 1:
                return img.expand(3, -1, -1)
            elif img.shape[0] == 3:
                return img
            else:
                raise ValueError(f"Unexpected number of channels: {img.shape[0]}")
        else:
            # PIL Image - convert to RGB if grayscale
            if hasattr(img, 'mode'):
                if img.mode == 'L':  # Grayscale
                    return img.convert('RGB')
            return img


class TinyImageNetValDataset(Dataset):
    """Tiny-ImageNet official validation split reader.

    Tiny-ImageNet ships validation images in `val/images/` with labels stored in
    `val/val_annotations.txt`. `torchvision.datasets.ImageFolder` cannot read this
    layout directly, so we map filenames to labels here.
    """

    def __init__(self, root: str, transform=None):
        from pathlib import Path
        from PIL import Image

        self.root = Path(root)
        self.transform = transform
        self.images_dir = self.root / "val" / "images"
        self.ann_file = self.root / "val" / "val_annotations.txt"

        if not self.images_dir.exists():
            raise FileNotFoundError(f"Tiny-ImageNet validation images directory not found: {self.images_dir}")
        if not self.ann_file.exists():
            raise FileNotFoundError(f"Tiny-ImageNet validation annotations not found: {self.ann_file}")

        train_dir = self.root / "train"
        if not train_dir.exists():
            raise FileNotFoundError(f"Tiny-ImageNet train directory not found: {train_dir}")

        self.class_names = sorted([p.name for p in train_dir.iterdir() if p.is_dir()])
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.class_names)}

        # Prefer class-folder layout if the images were already reorganized.
        val_dir = self.root / "val"
        class_dirs = [p for p in val_dir.iterdir() if p.is_dir() and p.name != "images"]

        self.samples = []
        if class_dirs:
            for class_dir in sorted(class_dirs, key=lambda p: p.name):
                if class_dir.name not in self.class_to_idx:
                    continue
                target = self.class_to_idx[class_dir.name]
                for img_path in sorted(class_dir.iterdir()):
                    if img_path.is_file() and img_path.suffix.lower() in {".jpeg", ".jpg", ".png", ".bmp", ".webp"}:
                        self.samples.append((img_path, target))
        else:
            with open(self.ann_file, "r") as f:
                for line in f:
                    parts = line.strip().split("\t")
                    if len(parts) >= 2:
                        img_name, cls_name = parts[0], parts[1]
                        img_path = self.images_dir / img_name
                        if cls_name in self.class_to_idx:
                            self.samples.append((img_path, self.class_to_idx[cls_name]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        from PIL import Image

        img_path, target = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, target


class ChestCTScanDataset(Dataset):
    """Chest CT scan dataset with canonical labels across inconsistent split folder names."""

    classes = ['adenocarcinoma', 'large.cell.carcinoma', 'normal', 'squamous.cell.carcinoma']
    class_to_idx = {name: idx for idx, name in enumerate(classes)}
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tif', '.tiff'}

    def __init__(self, root: str, transform=None):
        from pathlib import Path
        from PIL import Image

        self.root = Path(root)
        self.transform = transform
        self._image_cls = Image
        self.samples = []
        self.targets = []

        for dirpath, _, filenames in os.walk(self.root):
            label = self._label_from_dir(Path(dirpath).name)
            if label is None:
                continue
            for filename in sorted(filenames):
                path = Path(dirpath) / filename
                if path.suffix.lower() in self.image_extensions:
                    self.samples.append((path, label))
                    self.targets.append(label)

        if len(self.samples) == 0:
            raise RuntimeError(f"No chest CT images found under {self.root}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, target = self.samples[idx]
        img = self._image_cls.open(img_path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, target

    @classmethod
    def _label_from_dir(cls, dirname):
        name = dirname.lower()
        if name.startswith('adenocarcinoma'):
            return cls.class_to_idx['adenocarcinoma']
        if name.startswith('large.cell.carcinoma'):
            return cls.class_to_idx['large.cell.carcinoma']
        if name == 'normal':
            return cls.class_to_idx['normal']
        if name.startswith('squamous.cell.carcinoma'):
            return cls.class_to_idx['squamous.cell.carcinoma']
        return None


# Try to import medmnist
try:
    from medmnist import INFO
    MEDMNIST_AVAILABLE = True
except ImportError:
    MEDMNIST_AVAILABLE = False
    INFO = {}

# MedMNIST dataset metadata
# num_classes: Number of output classes for the classifier
# num_labels: Number of labels in multi-label format (if multi-label dataset)
MEDMNIST_DATASETS = {
    "pathmnist": {"num_classes": 9, "num_labels": 9, "info": "PathMNIST (9 tissue types from histopathological images)", "class_name": "PathMNIST"},
    "chestmnist": {"num_classes": 14, "num_labels": 14, "info": "ChestMNIST (14 anatomical regions, multi-label)", "class_name": "ChestMNIST"},
    "dermamnist": {"num_classes": 7, "num_labels": 7, "info": "DermaMNIST (7 skin disease classes)", "class_name": "DermaMNIST"},
    "bloodmnist": {"num_classes": 8, "num_labels": 8, "info": "BloodMNIST (8 blood cell types)", "class_name": "BloodMNIST"},
    "tissuesmnist": {"num_classes": 8, "num_labels": 8, "info": "TissueMNIST (8 tissue types)", "class_name": "TissueMNIST"},
    "organamnist": {"num_classes": 11, "num_labels": 11, "info": "OrganAMNIST (11 organ classes, axial view)", "class_name": "OrganAMNIST"},
    "organcmnist": {"num_classes": 11, "num_labels": 11, "info": "OrganCMNIST (11 organ classes, coronal view)", "class_name": "OrganCMNIST"},
    "organsmnist": {"num_classes": 11, "num_labels": 11, "info": "OrganSMNIST (11 organ classes, sagittal view)", "class_name": "OrganSMNIST"},
    "octmnist": {"num_classes": 4, "num_labels": 4, "info": "OCTMNIST (4 OCT ophthalmic disease classes)", "class_name": "OCTMNIST"},
    "pneumoniamnist": {"num_classes": 2, "num_labels": 2, "info": "PneumoniaMNIST (2 pneumonia classes)", "class_name": "PneumoniaMNIST"},
    "retinamnist": {"num_classes": 5, "num_labels": 5, "info": "RetinaMNIST (5 diabetic retinopathy severity levels)", "class_name": "RetinaMNIST"},
    "breastmnist": {"num_classes": 2, "num_labels": 2, "info": "BreastMNIST (2 breast cancer classes)", "class_name": "BreastMNIST"},
}

# Tiny-ImageNet metadata
TINYIMAGENET_META = {
    "tinyimagenet": {"num_classes": 200, "info": "Tiny-ImageNet (200 classes, 64x64 images)", "root_dir_name": "tiny-imagenet-200"}
}


def _resolve_pathmnist_224_root(data_dir: str) -> tuple[Path, bool]:
    """Find an existing MedMNIST+ PathMNIST-224 npz root when available."""
    data_root = Path(data_dir)
    repo_root = Path(__file__).resolve().parents[1]
    filename = "pathmnist_224.npz"
    candidates = [
        data_root / "pathmnist_224",
        data_root,
        repo_root / "data" / "pathmnist_224",
        repo_root / "data",
        repo_root.parent / "FMFP" / "data" / "pathmnist_224",
        Path.cwd() / "FMFP" / "data" / "pathmnist_224",
        Path.cwd().parent / "FMFP" / "data" / "pathmnist_224",
    ]

    for root in candidates:
        if (root / filename).exists():
            return root, False

    return data_root / "pathmnist_224", True


def get_transforms(dataset: str = "cifar10", augmentation: bool = True):
    """Get train and validation transforms for CIFAR."""
    if dataset.lower() == "cifar10":
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2470, 0.2435, 0.2616)
        num_classes = 10
    elif dataset.lower() == "cifar100":
        mean = (0.5071, 0.4867, 0.4408)
        std = (0.2675, 0.2565, 0.2761)
        num_classes = 100
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    if augmentation:
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
    else:
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])

    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    return train_transform, val_transform, num_classes


def get_medmnist_transforms(dataset: str = "pathmnist", augmentation: bool = True):
    """
    Get train and validation transforms for MedMNIST.
    
    MedMNIST images are 28x28, so we use simpler augmentation than CIFAR.
    Automatically converts grayscale (1-channel) images to 3-channel RGB
    so they work with standard ResNet architectures.
    """
    if dataset.lower() not in MEDMNIST_DATASETS:
        raise ValueError(f"Unknown MedMNIST dataset: {dataset}. Available: {list(MEDMNIST_DATASETS.keys())}")
    
    num_classes = MEDMNIST_DATASETS[dataset.lower()]["num_classes"]
    
    # Standard ImageNet normalization (for 3-channel images)
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    
    if augmentation:
        train_transform = transforms.Compose([
            GrayscaleTo3Channel(),
            transforms.RandomRotation(degrees=10),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
    else:
        train_transform = transforms.Compose([
            GrayscaleTo3Channel(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
    
    val_transform = transforms.Compose([
        GrayscaleTo3Channel(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    
    return train_transform, val_transform, num_classes


def get_tinyimagenet_transforms(augmentation: bool = True):
    """Get transforms for Tiny-ImageNet (64x64 RGB)."""
    # Use ImageNet mean/std
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    if augmentation:
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(64, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
    else:
        train_transform = transforms.Compose([
            transforms.Resize(64),
            transforms.CenterCrop(64),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])

    val_transform = transforms.Compose([
        transforms.Resize(64),
        transforms.CenterCrop(64),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    return train_transform, val_transform, TINYIMAGENET_META["tinyimagenet"]["num_classes"]


def get_cifar_dataloaders(
    dataset: str = "cifar10",
    data_dir: str = "./data",
    batch_size: int = 128,
    val_batch_size: int = 256,
    num_workers: int = 4,
    augmentation: bool = True,
    val_split: float = 0.1,
    seed: int = 42,
):
    """
    Get CIFAR-10/100 dataloaders with train/val/test split.

    Args:
        dataset: 'cifar10' or 'cifar100'
        data_dir: Directory to store/load data
        batch_size: Training batch size
        val_batch_size: Validation/test batch size
        num_workers: Number of data loading workers
        augmentation: Whether to use data augmentation on training set
        val_split: Fraction of training data to use for validation (0.0-1.0)
        seed: Random seed for reproducibility

    Returns:
        tuple: (train_loader, val_loader, test_loader, num_classes)
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Get transforms
    train_transform, val_transform, num_classes = get_transforms(dataset, augmentation)

    # Load dataset
    if dataset.lower() == "cifar10":
        dataset_class = datasets.CIFAR10
    elif dataset.lower() == "cifar100":
        dataset_class = datasets.CIFAR100
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    # Full training set (for train/val split)
    full_train = dataset_class(
        root=data_dir,
        train=True,
        download=True,
        transform=train_transform,
    )

    # Test set
    test_set = dataset_class(
        root=data_dir,
        train=False,
        download=True,
        transform=val_transform,
    )

    num_train = len(full_train)
    indices = list(range(num_train))
    split = int(np.floor(val_split * num_train))
    np.random.seed(seed)
    np.random.shuffle(indices)
    val_indices = indices[:split]
    train_indices = indices[split:]

    train_set = Subset(full_train, train_indices)
    val_set = Subset(full_train, val_indices)

    # Create dataloaders
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_set,
        batch_size=val_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_set,
        batch_size=val_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader, num_classes


def get_medmnist_dataloaders(
    dataset: str = "pathmnist",
    data_dir: str = "./data",
    batch_size: int = 128,
    val_batch_size: int = 256,
    num_workers: int = 4,
    augmentation: bool = True,
    val_split: float = 0.1,
    seed: int = 42,
):
    """
    Get MedMNIST dataloaders with train/val/test split.
    
    Args:
        dataset: MedMNIST dataset name (e.g., 'pathmnist', 'chestmnist', etc.)
        data_dir: Directory to store/load data
        batch_size: Training batch size
        val_batch_size: Validation/test batch size
        num_workers: Number of data loading workers
        augmentation: Whether to use data augmentation on training set
        val_split: Fraction of training data to use for validation (0.0-1.0)
        seed: Random seed for reproducibility
    
    Returns:
        tuple: (train_loader, val_loader, test_loader, num_classes)
    """
    if not MEDMNIST_AVAILABLE:
        raise ImportError("MedMNIST is not installed. Install with: pip install medmnist")
    
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    dataset_lower = dataset.lower()
    if dataset_lower not in MEDMNIST_DATASETS:
        raise ValueError(f"Unknown MedMNIST dataset: {dataset}. Available: {list(MEDMNIST_DATASETS.keys())}")
    
    # PathMNIST was trained from MedMNIST+ at 224x224 RGB in FMFP.
    if dataset_lower == "pathmnist":
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        train_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip() if augmentation else transforms.Lambda(lambda x: x),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        val_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        num_classes = MEDMNIST_DATASETS[dataset_lower]["num_classes"]
        size = 224
        as_rgb = True
        mmap_mode = "r"
    else:
        train_transform, val_transform, num_classes = get_medmnist_transforms(dataset, augmentation)
        size = None
        as_rgb = False
        mmap_mode = None
    
    download = True
    data_root = Path(data_dir)
    if dataset_lower == "pathmnist":
        data_root, download = _resolve_pathmnist_224_root(data_dir)
    data_root.mkdir(parents=True, exist_ok=True)
    
    # Dynamically import the correct dataset class using medmnist module
    try:
        import medmnist
        
        # Get class name from mapping
        class_name = MEDMNIST_DATASETS[dataset_lower]["class_name"]
        
        # Get the class from medmnist module
        if not hasattr(medmnist, class_name):
            raise AttributeError(f"MedMNIST module does not have class {class_name}")
        
        DataClass = getattr(medmnist, class_name)
        
        # Load MedMNIST datasets (split by official train/val/test)
        full_train_raw = DataClass(
            split="train",
            download=download,
            root=str(data_root),
            transform=train_transform,
            as_rgb=as_rgb,
            size=size,
            mmap_mode=mmap_mode,
        )
        
        val_set_raw = DataClass(
            split="val",
            download=download,
            root=str(data_root),
            transform=val_transform,
            as_rgb=as_rgb,
            size=size,
            mmap_mode=mmap_mode,
        )
        
        test_set_raw = DataClass(
            split="test",
            download=download,
            root=str(data_root),
            transform=val_transform,
            as_rgb=as_rgb,
            size=size,
            mmap_mode=mmap_mode,
        )
        
        # Check if the targets need conversion (either multi-label or single-label with wrong shape)
        # All MedMNIST datasets should go through the converter to ensure scalar targets
        sample_target = val_set_raw[0][1]
        if isinstance(sample_target, np.ndarray) and len(sample_target.shape) > 0:
            full_train = SingleOrMultiLabelConverter(full_train_raw)
            val_set = SingleOrMultiLabelConverter(val_set_raw)
            test_set = SingleOrMultiLabelConverter(test_set_raw)
        else:
            full_train = full_train_raw
            val_set = val_set_raw
            test_set = test_set_raw
    except Exception as e:
        raise RuntimeError(f"Error loading MedMNIST dataset '{dataset}': {e}. Make sure medmnist is installed and dataset name is correct.")
    
    # For some datasets, validation set might be small or empty
    # If user wants to further split training data for validation, we do that
    if val_split > 0 and (len(val_set) == 0 or len(val_set) < 100):
        # If official val set is empty/small, split from training
        n_train = len(full_train)
        n_val = int(n_train * val_split)
        n_train = n_train - n_val
        
        indices = np.random.permutation(len(full_train))
        train_indices = indices[:n_train]
        val_indices = indices[n_train:]
        
        train_set = Subset(full_train, train_indices)
        val_set = Subset(full_train, val_indices)
    else:
        train_set = full_train
    
    # Create dataloaders
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=custom_collate_fn,
    )
    
    val_loader = DataLoader(
        val_set,
        batch_size=val_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=custom_collate_fn,
    )
    
    test_loader = DataLoader(
        test_set,
        batch_size=val_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=custom_collate_fn,
    )
    
    return train_loader, val_loader, test_loader, num_classes


def _prepare_tiny_val_folder(root: str):
    """Reorganize Tiny-ImageNet validation images into subfolders using val_annotations.txt.

    This is a best-effort helper that will create `val/<class>` folders and move images accordingly
    if the user left the original Tiny-ImageNet `val/images` layout. It will skip if folders already
    exist. This helper does not delete the original `images` folder.
    """
    import shutil
    from pathlib import Path

    root_p = Path(root)
    val_dir = root_p / "val"
    images_dir = val_dir / "images"
    ann_file = val_dir / "val_annotations.txt"

    if not val_dir.exists() or not images_dir.exists() or not ann_file.exists():
        return

    # Read annotations
    with open(ann_file, "r") as f:
        lines = [l.strip().split() for l in f.readlines() if l.strip()]

    for parts in lines:
        img_name, cls = parts[0], parts[1]
        target_dir = val_dir / cls
        target_dir.mkdir(parents=True, exist_ok=True)
        src = images_dir / img_name
        dst = target_dir / img_name
        if src.exists() and not dst.exists():
            try:
                shutil.move(str(src), str(dst))
            except Exception:
                # best-effort move; ignore failures
                pass


def get_tinyimagenet_dataloaders(
    data_dir: str = "./data",
    batch_size: int = 128,
    val_batch_size: int = 256,
    num_workers: int = 4,
    augmentation: bool = True,
    val_split: float = 0.0,
    seed: int = 42,
):
    """Get Tiny-ImageNet dataloaders.

    Notes:
    - This helper expects the Tiny-ImageNet data to be placed under `data_dir/tiny-imagenet-200`.
    - If the `val` folder uses the original `images` + `val_annotations.txt` layout, it will try to
      reorganize validation images into class subfolders.
    - Tiny-ImageNet is not distributed via torchvision; download from the official source and
      extract into `data_dir/tiny-imagenet-200` before using this loader.
    """
    import random
    from pathlib import Path

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    train_transform, val_transform, num_classes = get_tinyimagenet_transforms(augmentation)

    root = Path(data_dir) / TINYIMAGENET_META["tinyimagenet"]["root_dir_name"]
    if not root.exists():
        raise RuntimeError(f"Tiny-ImageNet folder not found at {root}. Download and extract Tiny-ImageNet into this location.")

    # Prepare val folder if necessary (best-effort; not required for the custom val dataset)
    _prepare_tiny_val_folder(str(root))

    train_dir = root / "train"
    val_dir = root / "val"

    if not train_dir.exists():
        raise RuntimeError(f"Tiny-ImageNet train directory not found at {train_dir}")

    # Use ImageFolder for train and official val (after potential reorg)
    train_set_full = datasets.ImageFolder(root=str(train_dir), transform=train_transform)

    # Split official train into train/validation, keep official val as test.
    if val_split > 0:
        n_train = len(train_set_full)
        n_val = int(n_train * val_split)
        n_train = n_train - n_val

        indices = np.random.permutation(len(train_set_full))
        train_indices = indices[:n_train]
        val_indices = indices[n_train:]

        train_set = Subset(train_set_full, train_indices)
        val_set = Subset(train_set_full, val_indices)
    else:
        train_set = train_set_full
        val_set = train_set_full

    # Official Tiny-ImageNet val set is labeled, but not in ImageFolder layout.
    test_set = TinyImageNetValDataset(root=str(root), transform=val_transform)

    # Create dataloaders
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_set,
        batch_size=val_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_set,
        batch_size=val_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader, num_classes


def get_chest_xray_dataloaders(
    data_dir: str = "./data",
    batch_size: int = 128,
    val_batch_size: int = 256,
    num_workers: int = 4,
    augmentation: bool = True,
    val_split: float = 0.1,
    seed: int = 42,
):
    """Get Chest X-Ray dataloaders."""
    import os
    torch.manual_seed(seed)
    np.random.seed(seed)

    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    if augmentation:
        train_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
    else:
        train_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    train_set = datasets.ImageFolder(root=os.path.join(data_dir, 'chest_xray', 'train'), transform=train_transform)
    val_set = datasets.ImageFolder(root=os.path.join(data_dir, 'chest_xray', 'val'), transform=val_transform)
    test_set = datasets.ImageFolder(root=os.path.join(data_dir, 'chest_xray', 'test'), transform=val_transform)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_set, batch_size=val_batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_set, batch_size=val_batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader, 2

def get_mri_tumor_dataloaders(
    data_dir: str = "./data",
    batch_size: int = 128,
    val_batch_size: int = 256,
    num_workers: int = 4,
    augmentation: bool = True,
    val_split: float = 0.1,
    seed: int = 42,
):
    """Get MRI Tumor dataloaders."""
    import os
    from PIL import Image
    torch.manual_seed(seed)
    np.random.seed(seed)

    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    if augmentation:
        train_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
    else:
        train_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    train_dir = os.path.join(data_dir, 'mri_tumor', 'Training')
    test_dir = os.path.join(data_dir, 'mri_tumor', 'Testing')

    full_train_set = datasets.ImageFolder(root=train_dir)
    num_total = len(full_train_set)
    indices = list(range(num_total))
    np.random.shuffle(indices)

    split_val = int(np.floor(0.1 * num_total))
    val_idx = indices[:split_val]
    train_idx = indices[split_val:]

    class ImageFolderSubset(Dataset):
        def __init__(self, full_set, indices, transform=None):
            self.full_set = full_set
            self.indices = indices
            self.transform = transform

        def __len__(self):
            return len(self.indices)

        def __getitem__(self, idx):
            real_idx = self.indices[idx]
            path, target = self.full_set.samples[real_idx]
            img = Image.open(path).convert('RGB')
            if self.transform:
                img = self.transform(img)
            return img, target

    train_set = ImageFolderSubset(full_train_set, train_idx, train_transform)
    val_set = ImageFolderSubset(full_train_set, val_idx, val_transform)

    class ImageFolderWithIdx(Dataset):
        def __init__(self, root, transform=None):
            self.dataset = datasets.ImageFolder(root=root, transform=transform)

        def __len__(self):
            return len(self.dataset)

        def __getitem__(self, idx):
            img, target = self.dataset[idx]
            return img, target

    test_set = ImageFolderWithIdx(root=test_dir, transform=val_transform)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_set, batch_size=val_batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_set, batch_size=val_batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader, 4

def get_skin_cancer_dataloaders(
    data_dir: str = "./data",
    batch_size: int = 128,
    val_batch_size: int = 256,
    num_workers: int = 4,
    augmentation: bool = True,
    val_split: float = 0.1,
    seed: int = 42,
):
    """Get Skin Cancer ISIC dataloaders."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    if augmentation:
        train_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
    else:
        train_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    data_path = "/home/viet2005/workspace/Research/mixup/FMFP/data/skin_cancer_isic"
    
    full_set = datasets.ImageFolder(root=data_path)
    num_total = len(full_set)
    indices = list(range(num_total))
    np.random.seed(42)  # Match split in FMFP
    np.random.shuffle(indices)

    split_val = int(np.floor(0.1 * num_total))
    split_test = int(np.floor(0.1 * num_total))

    test_indices = indices[:split_test]
    val_indices = indices[split_test:split_test+split_val]
    train_indices = indices[split_test+split_val:]

    train_set_full = datasets.ImageFolder(root=data_path, transform=train_transform)
    val_set_full = datasets.ImageFolder(root=data_path, transform=val_transform)

    train_set = Subset(train_set_full, train_indices)
    val_set = Subset(val_set_full, val_indices)
    test_set = Subset(val_set_full, test_indices)

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_set,
        batch_size=val_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_set,
        batch_size=val_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader, 2


def get_alzheimer_dataloaders(
    data_dir: str = "./data",
    batch_size: int = 128,
    val_batch_size: int = 256,
    num_workers: int = 4,
    augmentation: bool = True,
    val_split: float = 0.1,
    seed: int = 42,
):
    """Get Alzheimer dataloaders using the same split as FMFP training."""
    import os
    from PIL import Image

    torch.manual_seed(seed)
    np.random.seed(seed)

    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    if augmentation:
        train_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
    else:
        train_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    candidates = [
        Path(data_dir) / "Alzheimer_s Dataset",
        Path("../FMFP/data/Alzheimer_s Dataset"),
    ]
    data_path = next((p for p in candidates if p.exists()), candidates[0])
    train_dir = data_path / "train"
    test_dir = data_path / "test"

    full_train_set = datasets.ImageFolder(root=str(train_dir))
    num_total = len(full_train_set)
    indices = list(range(num_total))
    np.random.seed(42)
    np.random.shuffle(indices)

    split_val = int(np.floor(0.1 * num_total))
    val_indices = indices[:split_val]
    train_indices = indices[split_val:]

    class ImageFolderSubset(Dataset):
        def __init__(self, full_set, indices, transform=None):
            self.full_set = full_set
            self.indices = indices
            self.transform = transform

        def __len__(self):
            return len(self.indices)

        def __getitem__(self, idx):
            real_idx = self.indices[idx]
            path, target = self.full_set.samples[real_idx]
            img = Image.open(path).convert("RGB")
            if self.transform:
                img = self.transform(img)
            return img, target

    train_set = ImageFolderSubset(full_train_set, train_indices, train_transform)
    val_set = ImageFolderSubset(full_train_set, val_indices, val_transform)
    test_set = datasets.ImageFolder(root=str(test_dir), transform=val_transform)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=val_batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=val_batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, test_loader, 4


def get_tuberculosis_dataloaders(
    data_dir: str = "./data",
    batch_size: int = 128,
    val_batch_size: int = 256,
    num_workers: int = 4,
    augmentation: bool = True,
    val_split: float = 0.1,
    seed: int = 42,
):
    """Get tuberculosis dataloaders using the same deterministic 80/10/10 split as FMFP training."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    if augmentation:
        train_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
    else:
        train_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    candidates = [
        Path(data_dir) / "TB_Chest_Radiography_Database",
        Path("../FMFP/data/TB_Chest_Radiography_Database"),
    ]
    data_path = next((p for p in candidates if p.exists()), candidates[0])
    full_set = datasets.ImageFolder(root=str(data_path))

    num_total = len(full_set)
    indices = list(range(num_total))
    np.random.seed(42)
    np.random.shuffle(indices)

    split_val = int(np.floor(0.1 * num_total))
    split_test = int(np.floor(0.1 * num_total))

    test_indices = indices[:split_test]
    val_indices = indices[split_test:split_test + split_val]
    train_indices = indices[split_test + split_val:]

    train_set_full = datasets.ImageFolder(root=str(data_path), transform=train_transform)
    val_set_full = datasets.ImageFolder(root=str(data_path), transform=val_transform)

    train_set = Subset(train_set_full, train_indices)
    val_set = Subset(val_set_full, val_indices)
    test_set = Subset(val_set_full, test_indices)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=val_batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=val_batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, test_loader, 2


def get_sars_cov_2_ct_scan_dataloaders(
    data_dir: str = "./data",
    batch_size: int = 128,
    val_batch_size: int = 256,
    num_workers: int = 4,
    augmentation: bool = True,
    val_split: float = 0.1,
    seed: int = 42,
):
    """Get SARS-CoV-2 CT dataloaders using the same deterministic 80/10/10 split as FMFP training."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    if augmentation:
        train_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
    else:
        train_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    candidates = [
        Path(data_dir) / "sars_cov_2_ct_scan",
        Path("../FMFP/data/sars_cov_2_ct_scan"),
    ]
    data_path = next((p for p in candidates if p.exists()), candidates[0])
    full_set = datasets.ImageFolder(root=str(data_path))

    num_total = len(full_set)
    indices = list(range(num_total))
    np.random.seed(42)
    np.random.shuffle(indices)

    split_val = int(np.floor(0.1 * num_total))
    split_test = int(np.floor(0.1 * num_total))

    test_indices = indices[:split_test]
    val_indices = indices[split_test:split_test + split_val]
    train_indices = indices[split_test + split_val:]

    train_set_full = datasets.ImageFolder(root=str(data_path), transform=train_transform)
    val_set_full = datasets.ImageFolder(root=str(data_path), transform=val_transform)

    train_set = Subset(train_set_full, train_indices)
    val_set = Subset(val_set_full, val_indices)
    test_set = Subset(val_set_full, test_indices)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=val_batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=val_batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, test_loader, 2


def get_chest_ct_scan_dataloaders(
    data_dir: str = "./data",
    batch_size: int = 128,
    val_batch_size: int = 256,
    num_workers: int = 4,
    augmentation: bool = True,
    val_split: float = 0.1,
    seed: int = 42,
):
    """Get chest CT scan dataloaders using the official train/valid/test folders."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    if augmentation:
        train_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
    else:
        train_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    candidates = [
        Path(data_dir) / "chest_ct_scan",
        Path("../FMFP/data/chest_ct_scan"),
    ]
    data_path = next((p for p in candidates if p.exists()), candidates[0])

    train_set = ChestCTScanDataset(root=str(data_path / "train"), transform=train_transform)
    val_set = ChestCTScanDataset(root=str(data_path / "valid"), transform=val_transform)
    test_set = ChestCTScanDataset(root=str(data_path / "test"), transform=val_transform)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=val_batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=val_batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, test_loader, 4


def get_dataloaders(
    dataset: str = "cifar10",
    data_dir: str = "./data",
    batch_size: int = 128,
    val_batch_size: int = 256,
    num_workers: int = 4,
    augmentation: bool = True,
    val_split: float = 0.1,
    seed: int = 42,
):
    """
    Unified interface to get dataloaders for any supported dataset.
    
    Automatically routes to appropriate loader based on dataset name.
    
    Args:
        dataset: Dataset name ('cifar10', 'cifar100', or any MedMNIST dataset)
        data_dir: Directory to store/load data
        batch_size: Training batch size
        val_batch_size: Validation/test batch size
        num_workers: Number of data loading workers
        augmentation: Whether to use data augmentation on training set
        val_split: Fraction of training data to use for validation (0.0-1.0)
        seed: Random seed for reproducibility
    
    Returns:
        tuple: (train_loader, val_loader, test_loader, num_classes)
    """
    dataset_lower = dataset.lower()
    
    if dataset_lower in ["cifar10", "cifar100"]:
        return get_cifar_dataloaders(
            dataset=dataset,
            data_dir=data_dir,
            batch_size=batch_size,
            val_batch_size=val_batch_size,
            num_workers=num_workers,
            augmentation=augmentation,
            val_split=val_split,
            seed=seed,
        )
    elif dataset_lower in MEDMNIST_DATASETS:
        return get_medmnist_dataloaders(
            dataset=dataset,
            data_dir=data_dir,
            batch_size=batch_size,
            val_batch_size=val_batch_size,
            num_workers=num_workers,
            augmentation=augmentation,
            val_split=val_split,
            seed=seed,
        )
    elif dataset_lower == "skin_cancer_isic":
        return get_skin_cancer_dataloaders(
            data_dir=data_dir,
            batch_size=batch_size,
            val_batch_size=val_batch_size,
            num_workers=num_workers,
            augmentation=augmentation,
            val_split=val_split,
            seed=seed,
        )
    elif dataset_lower == "chest_xray":
        return get_chest_xray_dataloaders(
            data_dir=data_dir,
            batch_size=batch_size,
            val_batch_size=val_batch_size,
            num_workers=num_workers,
            augmentation=augmentation,
            val_split=val_split,
            seed=seed,
        )
    elif dataset_lower == "mri_tumor":
        return get_mri_tumor_dataloaders(
            data_dir=data_dir,
            batch_size=batch_size,
            val_batch_size=val_batch_size,
            num_workers=num_workers,
            augmentation=augmentation,
            val_split=val_split,
            seed=seed,
        )
    elif dataset_lower == "alzheimer":
        return get_alzheimer_dataloaders(
            data_dir=data_dir,
            batch_size=batch_size,
            val_batch_size=val_batch_size,
            num_workers=num_workers,
            augmentation=augmentation,
            val_split=val_split,
            seed=seed,
        )
    elif dataset_lower == "tuberculosis":
        return get_tuberculosis_dataloaders(
            data_dir=data_dir,
            batch_size=batch_size,
            val_batch_size=val_batch_size,
            num_workers=num_workers,
            augmentation=augmentation,
            val_split=val_split,
            seed=seed,
        )
    elif dataset_lower == "sars_cov_2_ct_scan":
        return get_sars_cov_2_ct_scan_dataloaders(
            data_dir=data_dir,
            batch_size=batch_size,
            val_batch_size=val_batch_size,
            num_workers=num_workers,
            augmentation=augmentation,
            val_split=val_split,
            seed=seed,
        )
    elif dataset_lower == "chest_ct_scan":
        return get_chest_ct_scan_dataloaders(
            data_dir=data_dir,
            batch_size=batch_size,
            val_batch_size=val_batch_size,
            num_workers=num_workers,
            augmentation=augmentation,
            val_split=val_split,
            seed=seed,
        )
    else:
        available = ["cifar10", "cifar100"] + list(MEDMNIST_DATASETS.keys())
        # Extend available options with tinyimagenet
        available = available + [
            "tinyimagenet", "skin_cancer_isic", "chest_xray", "mri_tumor",
            "alzheimer", "tuberculosis", "sars_cov_2_ct_scan", "chest_ct_scan"
        ]
        if dataset_lower == "tinyimagenet":
            return get_tinyimagenet_dataloaders(
                data_dir=data_dir,
                batch_size=batch_size,
                val_batch_size=val_batch_size,
                num_workers=num_workers,
                augmentation=augmentation,
                val_split=val_split,
                seed=seed,
            )
        raise ValueError(f"Unknown dataset: {dataset}. Available datasets: {', '.join(available)}")
