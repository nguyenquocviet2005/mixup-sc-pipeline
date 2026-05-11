"""Convert checkpoints from external format to evaluation script format."""
import torch
from pathlib import Path

def convert_checkpoint(ckpt_path: Path, output_path: Path):
    """
    Convert checkpoint to evaluation script format.
    Expected format: {'model_state': state_dict, 'config': {...}}
    Input formats:
      - {'model_state_dict': state_dict, 'arch': str, 'args': dict, ...}
      - {'model_state_dict': state_dict, ...}
    """
    ckpt = torch.load(ckpt_path, map_location='cpu')
    
    # Extract model state
    if 'model_state' in ckpt:
        model_state = ckpt['model_state']
    elif 'model_state_dict' in ckpt:
        model_state = ckpt['model_state_dict']
    else:
        raise ValueError(f"No model state found in {ckpt_path}")
    
    # Extract or infer config
    config = ckpt.get('config', {})
    if not config:
        # Build config from available metadata
        args = ckpt.get('args', {})
        config = {
            'model': {'arch': ckpt.get('arch', args.get('arch', 'resnet18'))},
            'data': {'dataset': args.get('dataset', 'unknown')},
        }
    
    # Create new checkpoint in expected format
    new_ckpt = {
        'model_state': model_state,
        'config': config,
    }
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(new_ckpt, output_path)
    print(f"✓ Converted: {ckpt_path.name} → {output_path.name}")
    print(f"  Config: {config}")


if __name__ == "__main__":
    checkpoint_dir = Path("/home/viet2005/workspace/Research/mixup/mixup-sc-pipeline/checkpoints_from_quan")
    output_dir = Path("/home/viet2005/workspace/Research/mixup/mixup-sc-pipeline/checkpoints_converted")
    
    for ckpt_file in sorted(checkpoint_dir.glob("*.pth")):
        output_file = output_dir / ckpt_file.name
        try:
            convert_checkpoint(ckpt_file, output_file)
        except Exception as e:
            print(f"✗ Error converting {ckpt_file.name}: {e}")
    
    print(f"\nConversion complete. Files saved to: {output_dir}")
