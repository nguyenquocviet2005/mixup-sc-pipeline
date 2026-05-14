"""Models package."""
from .resnet import resnet18, resnet34, resnet50, resnet101
from .vit import vit_b_16, vit_b_4
from .vgg import vgg16_bn

import sys
from pathlib import Path

# Add FMFP to path to import resnet110
fmfp_path = Path("/home/viet2005/workspace/Research/mixup/FMFP")
if str(fmfp_path) not in sys.path:
    sys.path.append(str(fmfp_path))

try:
    from model.resnet import resnet110
except ImportError:
    print("Warning: Could not import resnet110 from FMFP")
    resnet110 = None

try:
    from model.wrn import WideResNet
except ImportError:
    print("Warning: Could not import WideResNet from FMFP")
    WideResNet = None

try:
    from model.densenet_BC import DenseNet3
except ImportError:
    print("Warning: Could not import DenseNet3 from FMFP")
    DenseNet3 = None

try:
    from model.convmixer import ConvMixer
except ImportError:
    print("Warning: Could not import ConvMixer from FMFP")
    ConvMixer = None

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
        "resnet110": resnet110,
        "wrn28_10": WideResNet,
        "dense": DenseNet3,
        "cmixer": ConvMixer,
    }

    if arch not in models:
        raise ValueError(f"Unknown architecture: {arch}")

    if models[arch] is None:
        raise ValueError(f"Architecture {arch} is not available (failed to import)")

    if arch.startswith("vgg"):
        return models[arch](num_classes=num_classes, input_size=input_size)
    if arch == "wrn28_10":
        return WideResNet(depth=28, num_classes=num_classes, widen_factor=10)
    if arch == "dense":
        return DenseNet3(depth=100, num_classes=num_classes, growth_rate=12, reduction=0.5, bottleneck=True, dropRate=0.0)
    if arch == "cmixer":
        return ConvMixer(256, 16, kernel_size=8, patch_size=1, n_classes=num_classes)
    if arch == "resnet50":
        import torchvision.models as tv_models
        return tv_models.resnet50(num_classes=num_classes)
    if arch == "resnet101":
        import torchvision.models as tv_models
        return tv_models.resnet101(num_classes=num_classes)
    return models[arch](num_classes=num_classes)

__all__ = ["get_model", "resnet18", "resnet34", "resnet50", "resnet101", "vit_b_16", "vit_b_4", "vgg16_bn", "resnet110"]

