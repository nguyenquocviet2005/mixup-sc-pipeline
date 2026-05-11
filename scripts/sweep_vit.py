import argparse
import subprocess
import itertools
import numpy as np

def run_sweep():
    parser = argparse.ArgumentParser(description="Sweep hyperparameters for ViT-B/4")
    parser.add_argument("--dataset", type=str, choices=["cifar10", "cifar100", "tinyimagenet"], required=True)
    args = parser.parse_args()

    # 10 logarithmically spaced learning rates from 1e-4 to 3e-3
    learning_rates = np.logspace(np.log10(1e-4), np.log10(3e-3), 10)
    
    # Weight decay sweep
    weight_decays = [0.0, 0.01, 0.02, 0.03, 0.04, 0.05]
    
    # For α, they tested 1.0 (default) and 0.4
    alphas = [1.0, 0.4]
    
    base_config = f"experiments/configs/vit_b_4_{args.dataset}.yaml"
    
    print(f"Starting sweep for {args.dataset}")
    print(f"Total configurations to test: {len(learning_rates) * len(weight_decays) * len(alphas)}")
    
    for alpha, wd, lr in itertools.product(alphas, weight_decays, learning_rates):
        exp_name = f"vit4_mixup_{args.dataset}_a{alpha}_wd{wd}_lr{lr:.5f}"
        print(f"\n======================================")
        print(f"Running: {exp_name}")
        print(f"======================================")
        
        # We can pass most args via command line, but method alpha and weight decay 
        # need to be updated in config. However, your script currently doesn't expose 
        # weight decay as a direct argparse flag, so we'll pass it if we update main.py, 
        # or we just edit the config on the fly. Let's do it via python by reading/writing YAML.
        
        import yaml
        with open(base_config, 'r') as f:
            cfg = yaml.safe_load(f)
            
        cfg['training']['learning_rate'] = float(lr)
        cfg['training']['weight_decay'] = float(wd)
        cfg['method']['mixup_alpha'] = float(alpha)
        
        temp_config = f"experiments/configs/temp_sweep.yaml"
        with open(temp_config, 'w') as f:
            yaml.dump(cfg, f)
            
        cmd = [
            ".venv/bin/python", "scripts/main.py",
            "--config", temp_config,
            "--exp-name", exp_name
        ]
        
        # Uncomment the next line to actually run the training
        # subprocess.run(cmd)
        print("Command:", " ".join(cmd))

if __name__ == "__main__":
    run_sweep()
