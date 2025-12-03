"""
Utility script to verify and analyze generated embeddings.

Usage:
    python check_embeddings.py --protein_path protein_delta_embeddings.npz --dna_path dna_delta_embeddings.npz
"""

import argparse
import numpy as np
import pandas as pd


def load_and_check_npz(filepath, name="Embeddings"):
    """Load and display info about npz file."""
    print(f"\n{'='*60}")
    print(f"{name}: {filepath}")
    print('='*60)

    try:
        data = np.load(filepath)

        print(f"\nAvailable keys: {list(data.keys())}")

        if 'delta_embeddings' in data:
            embeddings = data['delta_embeddings']
            print(f"\nDelta Embeddings Shape: {embeddings.shape}")
            print(f"  - Number of variants: {embeddings.shape[0]}")
            print(f"  - Embedding dimension: {embeddings.shape[1]}")
            print(f"  - Data type: {embeddings.dtype}")

            # Check for NaNs or Infs
            nan_count = np.isnan(embeddings).sum()
            inf_count = np.isinf(embeddings).sum()
            zero_rows = np.all(embeddings == 0, axis=1).sum()

            print(f"\nData Quality:")
            print(f"  - NaN values: {nan_count}")
            print(f"  - Inf values: {inf_count}")
            print(f"  - Zero rows (failed embeddings): {zero_rows}")

            # Basic statistics
            print(f"\nStatistics:")
            print(f"  - Mean: {embeddings.mean():.6f}")
            print(f"  - Std: {embeddings.std():.6f}")
            print(f"  - Min: {embeddings.min():.6f}")
            print(f"  - Max: {embeddings.max():.6f}")

            # L2 norm distribution
            norms = np.linalg.norm(embeddings, axis=1)
            print(f"\nL2 Norm Statistics:")
            print(f"  - Mean: {norms.mean():.6f}")
            print(f"  - Std: {norms.std():.6f}")
            print(f"  - Min: {norms.min():.6f}")
            print(f"  - Max: {norms.max():.6f}")

        if 'variant_ids' in data:
            variant_ids = data['variant_ids']
            print(f"\nVariant IDs:")
            print(f"  - Count: {len(variant_ids)}")
            print(f"  - First 5: {variant_ids[:5]}")
            print(f"  - Last 5: {variant_ids[-5:]}")

        if 'embedding_dim' in data:
            print(f"\nMetadata:")
            print(f"  - Embedding dim: {data['embedding_dim']}")

        if 'model_name' in data:
            print(f"  - Model name: {data['model_name']}")

        if 'layer_name' in data:
            print(f"  - Layer name: {data['layer_name']}")

        return data

    except FileNotFoundError:
        print(f"ERROR: File not found: {filepath}")
        return None
    except Exception as e:
        print(f"ERROR: Failed to load file: {e}")
        return None


def compare_embeddings(protein_data, dna_data, parquet_path=None):
    """Compare protein and DNA embeddings for consistency."""
    print(f"\n{'='*60}")
    print("Comparison & Consistency Checks")
    print('='*60)

    if protein_data is None or dna_data is None:
        print("ERROR: Cannot compare - one or both embedding files failed to load")
        return

    protein_ids = protein_data['variant_ids']
    dna_ids = dna_data['variant_ids']

    protein_embeddings = protein_data['delta_embeddings']
    dna_embeddings = dna_data['delta_embeddings']

    # Check ID alignment
    print(f"\nVariant ID Alignment:")
    if np.array_equal(protein_ids, dna_ids):
        print("  ✓ Protein and DNA variant IDs match perfectly")
    else:
        print("  ✗ WARNING: Protein and DNA variant IDs DO NOT match!")
        print(f"    - Protein IDs: {len(protein_ids)}")
        print(f"    - DNA IDs: {len(dna_ids)}")
        return

    # Check counts
    print(f"\nCount Alignment:")
    if protein_embeddings.shape[0] == dna_embeddings.shape[0]:
        print(f"  ✓ Both have {protein_embeddings.shape[0]} variants")
    else:
        print(f"  ✗ WARNING: Different number of variants!")
        print(f"    - Protein: {protein_embeddings.shape[0]}")
        print(f"    - DNA: {dna_embeddings.shape[0]}")

    # Check against parquet if provided
    if parquet_path:
        try:
            df = pd.read_parquet(parquet_path)
            print(f"\nParquet File Alignment:")
            print(f"  - Parquet has {len(df)} variants")

            if np.array_equal(protein_ids, df['variant_id'].values):
                print("  ✓ Embedding variant IDs match parquet file")
            else:
                print("  ✗ WARNING: Variant IDs don't match parquet file!")

        except Exception as e:
            print(f"\nCouldn't check parquet file: {e}")

    # Multimodal embedding info
    print(f"\nMultimodal Embedding:")
    total_dim = protein_embeddings.shape[1] + dna_embeddings.shape[1]
    print(f"  - Protein dimension: {protein_embeddings.shape[1]}")
    print(f"  - DNA dimension: {dna_embeddings.shape[1]}")
    print(f"  - Combined dimension: {total_dim}")

    # Example of combining
    print(f"\nExample: Combining embeddings")
    print(f"  multimodal = np.concatenate([protein_deltas, dna_deltas], axis=1)")
    print(f"  Result shape: ({protein_embeddings.shape[0]}, {total_dim})")


def main():
    parser = argparse.ArgumentParser(
        description="Check and verify generated embeddings"
    )
    parser.add_argument(
        "--protein_path",
        type=str,
        default="protein_delta_embeddings.npz",
        help="Path to protein embeddings npz file"
    )
    parser.add_argument(
        "--dna_path",
        type=str,
        default="dna_delta_embeddings.npz",
        help="Path to DNA embeddings npz file"
    )
    parser.add_argument(
        "--parquet_path",
        type=str,
        default="snvs.parquet",
        help="Path to original SNV parquet file (optional)"
    )

    args = parser.parse_args()

    # Load and check each file
    protein_data = load_and_check_npz(args.protein_path, "Protein Embeddings")
    dna_data = load_and_check_npz(args.dna_path, "DNA Embeddings")

    # Compare
    compare_embeddings(protein_data, dna_data, args.parquet_path)

    print(f"\n{'='*60}")
    print("Check complete!")
    print('='*60)


if __name__ == "__main__":
    main()
