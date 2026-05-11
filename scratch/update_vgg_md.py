import json
from pathlib import Path

def load_data(ds):
    path = Path(f"./results/vgg16_bn_robustness_{ds}.json")
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    res = {}
    for row in data['summary']:
        res[row['method']] = {
            'acc': (row['accuracy_mean'], row['accuracy_std']),
            'aurc': (row['aurc_mean'], row['aurc_std']),
            'eaurc': (row['eaurc_mean'], row['eaurc_std']),
        }
    return res

methods_order = [
    "Baseline Mixup",
    "Method 2: Feature-kNN / Logit Probability Blending",
    "MaxLogit pNorm",
    "DOCTOR-Alpha",
    "MaxLogit pNorm+",
    "ODIN",
    "Energy Score",
    "Mahalanobis Distance",
    "DOCTOR-Beta"
]

def format_cell(val_tuple, is_best):
    if not val_tuple: return "-"
    mean, std = val_tuple
    s = f"{mean:.4f}±{std:.4f}"
    if is_best:
        return f"**{s}**"
    return s

def generate_table(metric_key, is_lower_better=True):
    lines = []
    lines.append("| Method | CIFAR-10 | CIFAR-100 | Tiny-ImageNet | Average |")
    lines.append("|---|---|---|---|---|")
    
    d_list = [load_data('cifar10'), load_data('cifar100'), load_data('tinyimagenet')]
    
    # find best per column
    best_vals = [None, None, None, None]
    
    for i, d in enumerate(d_list):
        if not d: continue
        means = [d[m][metric_key][0] for m in methods_order if m in d]
        if means:
            best_vals[i] = min(means) if is_lower_better else max(means)
            
    # find best for average
    avg_means = []
    for m in methods_order:
        if all(d and m in d for d in d_list):
            avg = sum(d[m][metric_key][0] for d in d_list) / 3.0
            avg_means.append(avg)
    if avg_means:
        best_vals[3] = min(avg_means) if is_lower_better else max(avg_means)
        
    for m in methods_order:
        row_cells = []
        avg_mean = None
        if all(d and m in d for d in d_list):
            avg_mean = sum(d[m][metric_key][0] for d in d_list) / 3.0
            
        for i, d in enumerate(d_list):
            if d and m in d:
                mean, std = d[m][metric_key]
                is_best = False
                if best_vals[i] is not None:
                    # check if it matches the best value (within floating point)
                    is_best = abs(mean - best_vals[i]) < 1e-7
                row_cells.append(format_cell((mean, std), is_best))
            else:
                row_cells.append("-")
                
        if avg_mean is not None:
            is_best = False
            if best_vals[3] is not None:
                is_best = abs(avg_mean - best_vals[3]) < 1e-7
            avg_str = f"{avg_mean:.4f}"
            if is_best:
                avg_str = f"**{avg_str}**"
            row_cells.append(avg_str)
        else:
            row_cells.append("-")
            
        method_name = m.replace("Method 2: Feature-kNN / Logit Probability Blending", "Method 2: Feature-kNN / Blending")
        lines.append(f"| {method_name} | {' | '.join(row_cells)} |")
        
    return "\n".join(lines)

def main():
    md_path = Path("./results/ROBUSTNESS_EVALUATION_SUMMARY.md")
    content = md_path.read_text()
    
    vgg_section = "\n\n## VGG-16 Natural Image Datasets Results\n\n"
    vgg_section += "### AURC Results (Lower is Better)\n"
    vgg_section += generate_table('aurc', True) + "\n\n"
    
    vgg_section += "### E-AURC Results (Lower is Better)\n"
    vgg_section += generate_table('eaurc', True) + "\n\n"
    
    vgg_section += "### Accuracy Results (Higher is Better)\n"
    vgg_section += generate_table('acc', False) + "\n"
    
    if "VGG-16 Natural Image Datasets Results" not in content:
        if "## Summary & Key Insights" in content:
            content = content.replace("## Summary & Key Insights", vgg_section + "\n## Summary & Key Insights")
        else:
            content += vgg_section
        
        md_path.write_text(content)
        print("Updated MD file.")
    else:
        print("VGG section already exists.")

if __name__ == "__main__":
    main()
