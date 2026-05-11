"""Models package."""
from .resnet import resnet18, resnet34, resnet50, resnet101
from .vit import vit_b_16, vit_b_4
from .vgg import vgg16_bn

def get_model(arch: str, num_classes: int = 10, input_size: int = 32):
    """Get model by architecture name."""
    models = {
        "resnet18": resnet18,
        "resnet34": resnet34,
        "resnet50": resnet50,
        "resnet101": resnet101,
        "vit_b_16": vit_b_16,
        "vit_b_4": vit_b_4,
        "vgg16_bn": vgg16_bn,
    }

    if arch not in models:
        raise ValueError(f"Unknown architecture: {arch}")

    if arch.startswith("vgg"):
        return models[arch](num_classes=num_classes, input_size=input_size)
    return models[arch](num_classes=num_classes)

__all__ = ["get_model", "resnet18", "resnet34", "resnet50", "resnet101", "vit_b_16", "vit_b_4", "vgg16_bn"]
