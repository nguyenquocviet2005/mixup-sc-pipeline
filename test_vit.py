import torch
import timm

model = timm.create_model('vit_base_patch16_224', img_size=32, num_classes=10)
x = torch.randn(2, 3, 32, 32)

# Check timm interface for penultimate features
try:
    features = model.forward_features(x)
    print("forward_features output shape:", features.shape)
    
    penultimate = model.forward_head(features, pre_logits=True)
    print("penultimate features shape:", penultimate.shape)
    
    logits = model.head(penultimate)
    print("logits shape:", logits.shape)
except Exception as e:
    print(e)
