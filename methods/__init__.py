"""Methods package."""
from .base import BaseMethod, StandardMethod
from .mixup import MixupMethod, MixupVariant1, MixupVariant2

__all__ = [
    "BaseMethod",
    "StandardMethod",
    "MixupMethod",
    "MixupVariant1",
    "MixupVariant2",
]


def get_method(name: str, model, **kwargs):
    """Get training method by name."""
    methods = {
        "standard": StandardMethod,
        "mixup": MixupMethod,
        "mixup_variant1": MixupVariant1,
        "mixup_variant2": MixupVariant2,
    }

    if name not in methods:
        raise ValueError(f"Unknown method: {name}")

    return methods[name](model, **kwargs)
