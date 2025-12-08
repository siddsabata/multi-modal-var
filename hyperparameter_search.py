"""
Hyperparameter search for feature selection and weighted fusion.

Systematically tests different values of:
- k_features: Number of DNA features to select
- dna_weight: Weight for DNA features in weighted fusion

Compares all configurations and reports the best.
"""

import subprocess
import json
import re
from pathlib import Path


def parse_roc_auc(output):
    """Extract ROC-AUC from training output."""
    # Look for the cross-validation ROC-AUC line
    pattern = r'ROC_AUC\s+:\s+([\d.]+)\s+±\s+([\d.]+)'
    matches = re.findall(pattern, output)
    if matches:
        # Return the last match (from CV results)
        mean, std = matches[-1]
        return float(mean), float(std)
    return None, None


def run_experiment(k_features=None, dna_weight=None, data_dir="analysis", model="xgboost"):
    """Run a single experiment and extract results."""
    cmd = [
        ".venv/bin/python",
        "train_logistic_regression.py",
        "--data_dir", data_dir,
        "--model", model,
        "--run_feature_selection"
    ]

    if k_features is not None:
        cmd.extend(["--dna_k_features", str(k_features)])
    if dna_weight is not None:
        cmd.extend(["--dna_weight", str(dna_weight)])

    print(f"\n{'='*80}")
    if k_features:
        print(f"Running Feature Selection with k={k_features}")
    if dna_weight:
        print(f"Running Weighted Fusion with weight={dna_weight}")
    print(f"{'='*80}")

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"Error running experiment: {result.stderr}")
        return None

    # Parse results - look for the feature selection or weighted fusion results
    output = result.stdout

    # Extract ROC-AUC for feature selection and weighted fusion
    results = {}

    # Split into sections by configuration
    sections = output.split("######################################################################")

    for section in sections:
        if k_features and f"Feature Selection (k={k_features}" in section:
            # Extract CV results for feature selection
            cv_section = section.split("Cross-Validation Performance")[1] if "Cross-Validation Performance" in section else ""
            match = re.search(r'ROC_AUC\s+:\s+([\d.]+)\s+±\s+([\d.]+)', cv_section)
            if match:
                results['feature_selection'] = {
                    'k': k_features,
                    'roc_auc_mean': float(match.group(1)),
                    'roc_auc_std': float(match.group(2))
                }

        if dna_weight and f"Weighted Fusion (DNA weight={dna_weight}" in section:
            # Extract CV results for weighted fusion
            cv_section = section.split("Cross-Validation Performance")[1] if "Cross-Validation Performance" in section else ""
            match = re.search(r'ROC_AUC\s+:\s+([\d.]+)\s+±\s+([\d.]+)', cv_section)
            if match:
                results['weighted_fusion'] = {
                    'weight': dna_weight,
                    'roc_auc_mean': float(match.group(1)),
                    'roc_auc_std': float(match.group(2))
                }

    return results


def main():
    print("\n" + "="*80)
    print(" HYPERPARAMETER SEARCH FOR MULTIMODAL FUSION")
    print("="*80)

    # Hyperparameter grids
    k_values = [100, 250, 500, 1000, 2000]
    weight_values = [0.1, 0.2, 0.3, 0.5, 0.7]

    # Store all results
    all_results = {
        'feature_selection': [],
        'weighted_fusion': []
    }

    # Phase 1: Feature Selection
    print("\n" + "="*80)
    print(" PHASE 1: FEATURE SELECTION HYPERPARAMETER SEARCH")
    print("="*80)

    for k in k_values:
        results = run_experiment(k_features=k, dna_weight=0.3)  # Use default weight
        if results and 'feature_selection' in results:
            all_results['feature_selection'].append(results['feature_selection'])
            print(f"  k={k:4d}: ROC-AUC = {results['feature_selection']['roc_auc_mean']:.4f} ± {results['feature_selection']['roc_auc_std']:.4f}")

    # Phase 2: Weighted Fusion
    print("\n" + "="*80)
    print(" PHASE 2: WEIGHTED FUSION HYPERPARAMETER SEARCH")
    print("="*80)

    for weight in weight_values:
        results = run_experiment(k_features=500, dna_weight=weight)  # Use default k
        if results and 'weighted_fusion' in results:
            all_results['weighted_fusion'].append(results['weighted_fusion'])
            print(f"  weight={weight:.1f}: ROC-AUC = {results['weighted_fusion']['roc_auc_mean']:.4f} ± {results['weighted_fusion']['roc_auc_std']:.4f}")

    # Summary
    print("\n" + "="*80)
    print(" HYPERPARAMETER SEARCH SUMMARY")
    print("="*80)

    # Best feature selection
    if all_results['feature_selection']:
        best_fs = max(all_results['feature_selection'], key=lambda x: x['roc_auc_mean'])
        print(f"\nBest Feature Selection:")
        print(f"  k = {best_fs['k']}")
        print(f"  ROC-AUC = {best_fs['roc_auc_mean']:.4f} ± {best_fs['roc_auc_std']:.4f}")

        print(f"\nAll Feature Selection Results:")
        for res in sorted(all_results['feature_selection'], key=lambda x: x['roc_auc_mean'], reverse=True):
            print(f"  k={res['k']:4d}: {res['roc_auc_mean']:.4f} ± {res['roc_auc_std']:.4f}")

    # Best weighted fusion
    if all_results['weighted_fusion']:
        best_wf = max(all_results['weighted_fusion'], key=lambda x: x['roc_auc_mean'])
        print(f"\nBest Weighted Fusion:")
        print(f"  weight = {best_wf['weight']}")
        print(f"  ROC-AUC = {best_wf['roc_auc_mean']:.4f} ± {best_wf['roc_auc_std']:.4f}")

        print(f"\nAll Weighted Fusion Results:")
        for res in sorted(all_results['weighted_fusion'], key=lambda x: x['roc_auc_mean'], reverse=True):
            print(f"  weight={res['weight']:.1f}: {res['roc_auc_mean']:.4f} ± {res['roc_auc_std']:.4f}")

    # Baseline for comparison
    print(f"\nBaseline (Protein Only): 0.7679 ROC-AUC")

    # Save results to file
    output_file = "hyperparameter_search_results.json"
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {output_file}")

    print("\n" + "="*80)
    print(" SEARCH COMPLETE")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
