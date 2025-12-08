"""
Comprehensive analysis of the full SNV dataset.

This script analyzes the embeddings, labels, and provides statistics
to understand what will be used for training.
"""

import numpy as np
import pandas as pd
from pathlib import Path


def analyze_dataset(data_dir="analysis"):
    """Comprehensive dataset analysis."""
    data_dir = Path(data_dir)

    print("="*80)
    print(" FULL DATASET ANALYSIS")
    print("="*80)

    # Load all data
    print("\nLoading data...")
    dna_data = np.load(data_dir / 'dna_delta_embeddings.npz')
    protein_data = np.load(data_dir / 'protein_delta_embeddings.npz')
    df = pd.read_parquet(data_dir / 'snvs.parquet')

    X_dna = dna_data['delta_embeddings']
    X_protein = protein_data['delta_embeddings']
    failed_indices = protein_data.get('failed_indices', np.array([]))
    y = df['label'].values

    print(f"\n{'-'*80}")
    print(" RAW DATA OVERVIEW")
    print(f"{'-'*80}")
    print(f"Total samples in dataset: {len(y):,}")
    print(f"DNA embeddings shape: {X_dna.shape}")
    print(f"Protein embeddings shape: {X_protein.shape}")

    # Embedding statistics
    print(f"\n{'-'*80}")
    print(" EMBEDDING STATISTICS")
    print(f"{'-'*80}")

    print("\nDNA Embeddings:")
    print(f"  Dimension: {X_dna.shape[1]}")
    print(f"  Mean abs value: {np.abs(X_dna).mean():.6f}")
    print(f"  Std dev: {X_dna.std():.6f}")
    print(f"  Range: [{X_dna.min():.2f}, {X_dna.max():.2f}]")
    dna_norms = np.linalg.norm(X_dna, axis=1)
    print(f"  L2 norm range: [{dna_norms.min():.2f}, {dna_norms.max():.2f}]")

    print("\nProtein Embeddings:")
    print(f"  Dimension: {X_protein.shape[1]}")
    print(f"  Mean abs value: {np.abs(X_protein).mean():.6f}")
    print(f"  Std dev: {X_protein.std():.6f}")
    print(f"  Range: [{X_protein.min():.2f}, {X_protein.max():.2f}]")
    protein_norms = np.linalg.norm(X_protein, axis=1)
    print(f"  L2 norm range: [{protein_norms.min():.2f}, {protein_norms.max():.2f}]")

    # Calculate scale ratio
    dna_mean_abs = np.abs(X_dna).mean()
    protein_mean_abs = np.abs(X_protein[protein_norms > 0]).mean()  # Exclude failed
    scale_ratio = dna_mean_abs / protein_mean_abs if protein_mean_abs > 0 else float('inf')
    print(f"\n  Scale ratio (DNA/Protein): {scale_ratio:.1f}×")

    # Label distribution
    print(f"\n{'-'*80}")
    print(" LABEL DISTRIBUTION (RAW)")
    print(f"{'-'*80}")
    unique_labels, counts = np.unique(y, return_counts=True)
    for label, count in zip(unique_labels, counts):
        if label == -1:
            label_name = "Uncertain/Conflicting"
        elif label == 0:
            label_name = "Benign"
        elif label == 1:
            label_name = "Pathogenic"
        else:
            label_name = f"Unknown-{label}"
        pct = 100 * count / len(y)
        print(f"  {label_name:25s} ({label:2d}): {count:7,} ({pct:5.2f}%)")

    # Failed embeddings
    print(f"\n{'-'*80}")
    print(" FAILED PROTEIN EMBEDDINGS")
    print(f"{'-'*80}")
    print(f"Failed count: {len(failed_indices):,} ({100*len(failed_indices)/len(y):.1f}%)")

    if len(failed_indices) > 0:
        # Check label distribution of failed samples
        failed_labels = y[failed_indices]
        unique_failed, counts_failed = np.unique(failed_labels, return_counts=True)
        print(f"\nLabel distribution of failed samples:")
        for label, count in zip(unique_failed, counts_failed):
            if label == -1:
                label_name = "Uncertain"
            elif label == 0:
                label_name = "Benign"
            elif label == 1:
                label_name = "Pathogenic"
            else:
                label_name = f"Unknown"
            pct = 100 * count / len(failed_indices)
            print(f"  {label_name:15s}: {count:6,} ({pct:5.2f}%)")

    # Filter for training
    print(f"\n{'-'*80}")
    print(" FILTERING FOR TRAINING")
    print(f"{'-'*80}")

    valid_mask = np.ones(len(y), dtype=bool)

    # Remove failed embeddings
    n_failed = len(failed_indices)
    if n_failed > 0:
        valid_mask[failed_indices] = False
        print(f"Removing {n_failed:,} failed protein embeddings")

    # Remove uncertain labels
    uncertain_mask = (y == -1)
    n_uncertain = uncertain_mask.sum()
    if n_uncertain > 0:
        valid_mask &= ~uncertain_mask
        print(f"Removing {n_uncertain:,} uncertain labels (label=-1)")

    # Compute overlap
    overlap = (failed_indices.reshape(-1, 1) == np.where(uncertain_mask)[0]).any(axis=1).sum()
    if overlap > 0:
        print(f"  (Including {overlap:,} samples that are both failed and uncertain)")

    # Final dataset
    X_dna_clean = X_dna[valid_mask]
    X_protein_clean = X_protein[valid_mask]
    y_clean = y[valid_mask]

    print(f"\n{'-'*80}")
    print(" FINAL TRAINING DATASET")
    print(f"{'-'*80}")
    print(f"Total samples for training: {len(y_clean):,}")
    print(f"Removed: {len(y) - len(y_clean):,} samples ({100*(len(y) - len(y_clean))/len(y):.1f}%)")

    print(f"\nFinal label distribution:")
    unique_clean, counts_clean = np.unique(y_clean, return_counts=True)
    for label, count in zip(unique_clean, counts_clean):
        label_name = "Benign" if label == 0 else "Pathogenic"
        pct = 100 * count / len(y_clean)
        print(f"  {label_name:15s} ({label}): {count:7,} ({pct:5.2f}%)")

    # Class imbalance ratio
    if len(unique_clean) == 2:
        imbalance_ratio = max(counts_clean) / min(counts_clean)
        print(f"\nClass imbalance ratio: {imbalance_ratio:.2f}:1")

    # Sample size considerations
    print(f"\n{'-'*80}")
    print(" TRAINING CONSIDERATIONS")
    print(f"{'-'*80}")
    print(f"Train/test split (80/20):")
    n_train = int(0.8 * len(y_clean))
    n_test = len(y_clean) - n_train
    print(f"  Training samples: ~{n_train:,}")
    print(f"  Test samples: ~{n_test:,}")

    print(f"\n5-fold cross-validation:")
    n_per_fold = len(y_clean) // 5
    print(f"  Samples per fold: ~{n_per_fold:,}")

    # Dimensionality warning
    total_dims = X_dna.shape[1] + X_protein.shape[1]
    print(f"\nDimensionality:")
    print(f"  Combined features: {total_dims:,}")
    print(f"  Training samples: {n_train:,}")
    if total_dims > n_train:
        print(f"  ⚠ WARNING: p ({total_dims:,}) >> n ({n_train:,}) - PCA reduction essential!")
    else:
        print(f"  ✓ n > p (no curse of dimensionality)")

    # Memory estimate
    memory_mb = (X_dna_clean.nbytes + X_protein_clean.nbytes) / (1024**2)
    print(f"\nMemory usage:")
    print(f"  Embeddings: {memory_mb:.1f} MB")

    print(f"\n{'='*80}")
    print(" ANALYSIS COMPLETE")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    analyze_dataset()
