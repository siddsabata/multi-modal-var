"""
Efficient hyperparameter search for feature selection and weighted fusion.

Runs all hyperparameter configurations in a single execution (loads data once).
Tests multiple values of k_features and dna_weight to find optimal settings.
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

# Import from training script
import sys
sys.path.insert(0, str(Path(__file__).parent))
from train_logistic_regression import (
    load_data, filter_failed_samples, create_model, evaluate_model
)


def evaluate_feature_selection(X_dna, X_protein, y, k_features, n_folds, random_seed, model_type='xgboost'):
    """
    Evaluate feature selection with k DNA features using cross-validation.
    Returns mean ROC-AUC across folds.
    """
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_seed)
    roc_aucs = []

    for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X_dna, y)):
        X_dna_train, X_dna_val = X_dna[train_idx], X_dna[val_idx]
        X_protein_train, X_protein_val = X_protein[train_idx], X_protein[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # Select top K DNA features
        selector = SelectKBest(mutual_info_classif, k=k_features)
        X_dna_train_selected = selector.fit_transform(X_dna_train, y_train)
        X_dna_val_selected = selector.transform(X_dna_val)

        # Scale each modality
        scaler_dna = StandardScaler()
        scaler_protein = StandardScaler()

        X_dna_train_scaled = scaler_dna.fit_transform(X_dna_train_selected)
        X_dna_val_scaled = scaler_dna.transform(X_dna_val_selected)

        X_protein_train_scaled = scaler_protein.fit_transform(X_protein_train)
        X_protein_val_scaled = scaler_protein.transform(X_protein_val)

        # Concatenate
        X_train = np.concatenate([X_dna_train_scaled, X_protein_train_scaled], axis=1)
        X_val = np.concatenate([X_dna_val_scaled, X_protein_val_scaled], axis=1)

        # PCA
        pca = PCA(n_components=0.95, random_state=random_seed)
        X_train_pca = pca.fit_transform(X_train)
        X_val_pca = pca.transform(X_val)

        # Train model
        imbalance_ratio = np.sum(y_train == 1) / np.sum(y_train == 0)
        model = create_model(model_type, imbalance_ratio, random_seed)
        model.fit(X_train_pca, y_train)

        # Evaluate
        y_prob = model.predict_proba(X_val_pca)[:, 1]
        roc_auc = roc_auc_score(y_val, y_prob)
        roc_aucs.append(roc_auc)

    return np.mean(roc_aucs), np.std(roc_aucs)


def evaluate_weighted_fusion(X_dna, X_protein, y, dna_weight, n_folds, random_seed, model_type='xgboost'):
    """
    Evaluate weighted fusion with given DNA weight using cross-validation.
    Returns mean ROC-AUC across folds.
    """
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_seed)
    roc_aucs = []

    for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X_dna, y)):
        X_dna_train, X_dna_val = X_dna[train_idx], X_dna[val_idx]
        X_protein_train, X_protein_val = X_protein[train_idx], X_protein[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # Scale each modality
        scaler_dna = StandardScaler()
        scaler_protein = StandardScaler()

        X_dna_train_scaled = scaler_dna.fit_transform(X_dna_train)
        X_dna_val_scaled = scaler_dna.transform(X_dna_val)

        X_protein_train_scaled = scaler_protein.fit_transform(X_protein_train)
        X_protein_val_scaled = scaler_protein.transform(X_protein_val)

        # Apply weights
        X_dna_train_weighted = X_dna_train_scaled * dna_weight
        X_dna_val_weighted = X_dna_val_scaled * dna_weight

        X_protein_train_weighted = X_protein_train_scaled * 1.0
        X_protein_val_weighted = X_protein_val_scaled * 1.0

        # Concatenate
        X_train = np.concatenate([X_dna_train_weighted, X_protein_train_weighted], axis=1)
        X_val = np.concatenate([X_dna_val_weighted, X_protein_val_weighted], axis=1)

        # PCA
        pca = PCA(n_components=0.95, random_state=random_seed)
        X_train_pca = pca.fit_transform(X_train)
        X_val_pca = pca.transform(X_val)

        # Train model
        imbalance_ratio = np.sum(y_train == 1) / np.sum(y_train == 0)
        model = create_model(model_type, imbalance_ratio, random_seed)
        model.fit(X_train_pca, y_train)

        # Evaluate
        y_prob = model.predict_proba(X_val_pca)[:, 1]
        roc_auc = roc_auc_score(y_val, y_prob)
        roc_aucs.append(roc_auc)

    return np.mean(roc_aucs), np.std(roc_aucs)


def main():
    parser = argparse.ArgumentParser(
        description="Efficient hyperparameter search for multimodal fusion"
    )
    parser.add_argument("--data_dir", type=str, default="analysis")
    parser.add_argument("--model", type=str, default="xgboost")
    parser.add_argument("--n_folds", type=int, default=5)
    parser.add_argument("--random_seed", type=int, default=42)
    parser.add_argument("--search_type", type=str, default="both",
                       choices=["feature_selection", "weighted_fusion", "both"])

    args = parser.parse_args()

    print("="*80)
    print(" EFFICIENT HYPERPARAMETER SEARCH")
    print("="*80)

    # Load data once
    print("\nLoading data...")
    X_dna, X_protein, y, variant_ids, failed_indices = load_data(args.data_dir)
    X_dna, X_protein, y, variant_ids = filter_failed_samples(
        X_dna, X_protein, y, variant_ids, failed_indices
    )

    print(f"Dataset: {len(y)} samples")
    print(f"DNA dims: {X_dna.shape[1]}, Protein dims: {X_protein.shape[1]}")

    # Hyperparameter grids
    k_values = [100, 250, 500, 1000, 2000]
    weight_values = [0.1, 0.2, 0.3, 0.5, 0.7]

    results = {}

    # Phase 1: Feature Selection
    if args.search_type in ["feature_selection", "both"]:
        print("\n" + "="*80)
        print(" PHASE 1: FEATURE SELECTION (k-values)")
        print("="*80)
        print(f"\nTesting k = {k_values}")

        results['feature_selection'] = []

        for k in k_values:
            print(f"\n  Testing k={k}...", end=" ", flush=True)
            mean_auc, std_auc = evaluate_feature_selection(
                X_dna, X_protein, y, k, args.n_folds, args.random_seed, args.model
            )
            results['feature_selection'].append({
                'k': k,
                'roc_auc_mean': mean_auc,
                'roc_auc_std': std_auc
            })
            print(f"ROC-AUC = {mean_auc:.4f} ± {std_auc:.4f}")

    # Phase 2: Weighted Fusion
    if args.search_type in ["weighted_fusion", "both"]:
        print("\n" + "="*80)
        print(" PHASE 2: WEIGHTED FUSION (DNA weights)")
        print("="*80)
        print(f"\nTesting weights = {weight_values}")

        results['weighted_fusion'] = []

        for weight in weight_values:
            print(f"\n  Testing weight={weight}...", end=" ", flush=True)
            mean_auc, std_auc = evaluate_weighted_fusion(
                X_dna, X_protein, y, weight, args.n_folds, args.random_seed, args.model
            )
            results['weighted_fusion'].append({
                'weight': weight,
                'roc_auc_mean': mean_auc,
                'roc_auc_std': std_auc
            })
            print(f"ROC-AUC = {mean_auc:.4f} ± {std_auc:.4f}")

    # Summary
    print("\n" + "="*80)
    print(" HYPERPARAMETER SEARCH RESULTS")
    print("="*80)

    baseline_auc = 0.7679  # Protein-only baseline

    # Feature Selection Results
    if 'feature_selection' in results and results['feature_selection']:
        print("\n" + "-"*80)
        print(" Feature Selection Results (sorted by ROC-AUC)")
        print("-"*80)
        print(f"{'k':>6}  {'ROC-AUC':>10}  {'Std':>8}  {'vs Baseline':>12}")
        print("-"*80)

        sorted_fs = sorted(results['feature_selection'], key=lambda x: x['roc_auc_mean'], reverse=True)
        for res in sorted_fs:
            diff = res['roc_auc_mean'] - baseline_auc
            marker = "★" if diff > 0 else ""
            print(f"{res['k']:6d}  {res['roc_auc_mean']:10.4f}  {res['roc_auc_std']:8.4f}  {diff:+12.4f} {marker}")

        best_fs = sorted_fs[0]
        print(f"\nBest: k={best_fs['k']} → ROC-AUC = {best_fs['roc_auc_mean']:.4f} ± {best_fs['roc_auc_std']:.4f}")

    # Weighted Fusion Results
    if 'weighted_fusion' in results and results['weighted_fusion']:
        print("\n" + "-"*80)
        print(" Weighted Fusion Results (sorted by ROC-AUC)")
        print("-"*80)
        print(f"{'Weight':>8}  {'ROC-AUC':>10}  {'Std':>8}  {'vs Baseline':>12}")
        print("-"*80)

        sorted_wf = sorted(results['weighted_fusion'], key=lambda x: x['roc_auc_mean'], reverse=True)
        for res in sorted_wf:
            diff = res['roc_auc_mean'] - baseline_auc
            marker = "★" if diff > 0 else ""
            print(f"{res['weight']:8.1f}  {res['roc_auc_mean']:10.4f}  {res['roc_auc_std']:8.4f}  {diff:+12.4f} {marker}")

        best_wf = sorted_wf[0]
        print(f"\nBest: weight={best_wf['weight']} → ROC-AUC = {best_wf['roc_auc_mean']:.4f} ± {best_wf['roc_auc_std']:.4f}")

    # Overall comparison
    print("\n" + "-"*80)
    print(" Baseline Comparison")
    print("-"*80)
    print(f"Protein-only baseline: {baseline_auc:.4f} ROC-AUC")

    if 'feature_selection' in results and results['feature_selection']:
        best_fs = max(results['feature_selection'], key=lambda x: x['roc_auc_mean'])
        diff = best_fs['roc_auc_mean'] - baseline_auc
        status = "BEATS" if diff > 0 else "CLOSE TO" if abs(diff) < 0.01 else "BELOW"
        print(f"Best Feature Selection: {best_fs['roc_auc_mean']:.4f} ({diff:+.4f}) - {status} baseline")

    if 'weighted_fusion' in results and results['weighted_fusion']:
        best_wf = max(results['weighted_fusion'], key=lambda x: x['roc_auc_mean'])
        diff = best_wf['roc_auc_mean'] - baseline_auc
        status = "BEATS" if diff > 0 else "CLOSE TO" if abs(diff) < 0.01 else "BELOW"
        print(f"Best Weighted Fusion:  {best_wf['roc_auc_mean']:.4f} ({diff:+.4f}) - {status} baseline")

    # Save results
    import json
    output_file = "hyperparameter_search_results.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_file}")

    print("\n" + "="*80)
    print(" SEARCH COMPLETE")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
