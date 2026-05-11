import torch
import torch.nn as nn
import timm
from timm.models.vision_transformer import VisionTransformer

def vit_b_16(num_classes=10, img_size=32, pretrained=False):
    """
    ViT-Base with patch size 16.
    By default configured for CIFAR/MedMNIST resolutions (e.g., 32x32).
    """
    if pretrained:
        # Note: Pretrained weights are typically for 224x224. 
        # timm handles resizing position embeddings if img_size is different.
        model = timm.create_model(
            'vit_base_patch16_224', 
            pretrained=True, 
            num_classes=num_classes, 
            img_size=img_size
        )
    else:
        model = VisionTransformer(
            img_size=img_size,
            patch_size=16,
            embed_dim=768,
            depth=12,
            num_heads=12,
            num_classes=num_classes
        )
    return model

def vit_b_4(num_classes=10, img_size=32, pretrained=False):
    """
    ViT-Base with patch size 4.
    Highly effective for small resolution images like CIFAR 32x32 (yields 8x8=64 patches).
    Pretrained weights are not officially available for patch_size=4 in timm's default registry,
    so we initialize it from scratch if pretrained is requested, or use patch interpolation if possible.
    """
    if pretrained:
        print("Warning: No standard pretrained weights for ViT-B/4. Initializing from scratch.")
        
    model = VisionTransformer(
        img_size=img_size,
        patch_size=4,
        embed_dim=768,
        depth=12,
        num_heads=12,
        num_classes=num_classes
    )
    return model
