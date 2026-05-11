"""Device utilities for GPU/CPU management."""
import torch


def get_device():
    """Get the appropriate device (GPU/CPU)."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    return device


def setup_device_kwargs():
    """Return kwargs for DataLoader based on device."""
    if torch.cuda.is_available():
        return {"pin_memory": True, "num_workers": 4}
    else:
        return {"num_workers": 0}
