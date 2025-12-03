"""
Generate DNA delta embeddings using base pretrained Evo2 model.

For each SNV in the parquet file:
    dna_delta_embedding = evo2(wt_dna) - evo2(alt_dna)

Saves embeddings to npz file with indices matching the parquet file.
"""

import argparse
import os
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

try:
    from evo2 import Evo2
except ImportError:
    print("Error: evo2 package not found. Please install it with: pip install evo2")
    print("Or ensure you're in the correct conda environment (evo2 environment)")
    raise


def get_dna_embedding(model, dna_seq, layer_name, device='cuda'):
    """
    Extract embedding from Evo2 for a single DNA sequence.

    Args:
        model: Evo2 model
        dna_seq: DNA sequence string (ACGT format)
        layer_name: Name of layer to extract embeddings from
        device: Device to run on

    Returns:
        embedding: Mean-pooled embedding vector [embedding_dim]
    """
    # Convert sequence to uppercase to ensure consistency
    dna_seq = dna_seq.upper()

    # Tokenize the DNA sequence
    input_ids = torch.tensor(
        model.tokenizer.tokenize(dna_seq),
        dtype=torch.int,
    ).unsqueeze(0).to(device)  # [1, L]

    # Get embeddings from the specified layer
    with torch.no_grad():
        outputs, embeddings = model(
            input_ids,
            return_embeddings=True,
            layer_names=[layer_name]
        )
        hidden_states = embeddings[layer_name]  # [1, L, dim]

        # Mean pool across sequence length
        embedding = hidden_states.mean(dim=1).squeeze(0)  # [dim]

    return embedding.cpu().numpy()


def generate_dna_embeddings(
    parquet_path,
    output_path,
    model_name="evo2_7b",
    layer_name=None,
    batch_size=8,
    device='cuda'
):
    """
    Generate delta embeddings for all SNVs in parquet file.

    Args:
        parquet_path: Path to SNV parquet file
        output_path: Path to save npz file
        model_name: Evo2 model name (e.g., 'evo2_7b', 'evo2_40b', 'evo2_1b_base')
        layer_name: Name of layer to extract embeddings from (auto-detected if None)
        batch_size: Number of SNVs to process before updating progress
        device: Device to run on
    """
    print("Loading SNV data from parquet...")
    df = pd.read_parquet(parquet_path)
    print(f"Loaded {len(df)} SNVs")

    print(f"\nLoading Evo2 base pretrained model: {model_name}...")
    print("This may take a few minutes and requires significant GPU memory...")
    model = Evo2(model_name)
    print(f"Model loaded: {model_name}")

    # Auto-detect layer name based on model if not specified
    if layer_name is None:
        # Different models have different architectures
        # For evo2_7b, we use a later layer
        if '7b' in model_name:
            layer_name = 'blocks.28.mlp.l3'
        elif '40b' in model_name:
            layer_name = 'blocks.44.mlp.l3'  # 40B model has more layers
        elif '1b' in model_name:
            layer_name = 'blocks.20.mlp.l3'  # 1B model has fewer layers
        else:
            layer_name = 'blocks.28.mlp.l3'  # default
        print(f"Auto-detected layer name: {layer_name}")
    else:
        print(f"Using specified layer name: {layer_name}")

    # Get embedding dimension by running a test sequence
    print("\nDetermining embedding dimension...")
    test_seq = "ACGT" * 10
    test_embedding = get_dna_embedding(model, test_seq, layer_name, device)
    embedding_dim = test_embedding.shape[0]
    print(f"Embedding dimension: {embedding_dim}")

    # Prepare output arrays
    delta_embeddings = np.zeros((len(df), embedding_dim), dtype=np.float32)
    variant_ids = df['variant_id'].values

    print("\nGenerating DNA delta embeddings...")

    # Process in batches to show progress
    num_batches = (len(df) + batch_size - 1) // batch_size

    for batch_idx in tqdm(range(num_batches), desc="Processing batches"):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(df))

        for i in range(start_idx, end_idx):
            row = df.iloc[i]

            try:
                # Get embeddings for wild-type and alternate DNA sequences
                wt_embedding = get_dna_embedding(
                    model, row['wt_dna'], layer_name, device
                )
                alt_embedding = get_dna_embedding(
                    model, row['alt_dna'], layer_name, device
                )

                # Compute delta embedding
                delta_embeddings[i] = wt_embedding - alt_embedding

            except Exception as e:
                print(f"\nWarning: Failed to process SNV at index {i} (variant_id={row['variant_id']}): {e}")
                # Leave as zeros for failed embeddings
                continue

    # Save to npz file
    print(f"\nSaving embeddings to {output_path}...")
    np.savez_compressed(
        output_path,
        delta_embeddings=delta_embeddings,
        variant_ids=variant_ids,
        embedding_dim=embedding_dim,
        model_name=model_name,
        layer_name=layer_name
    )
    print("Done!")
    print(f"Saved {len(delta_embeddings)} DNA delta embeddings")
    print(f"Shape: {delta_embeddings.shape}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate DNA delta embeddings using Evo2"
    )
    parser.add_argument(
        "--parquet_path",
        type=str,
        default="snvs.parquet",
        help="Path to SNV parquet file"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="dna_delta_embeddings.npz",
        help="Path to save output npz file"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        choices=['evo2_7b', 'evo2_40b', 'evo2_7b_base', 'evo2_40b_base', 'evo2_1b_base'],
        default='evo2_7b',
        help="Evo2 model name to use"
    )
    parser.add_argument(
        "--layer_name",
        type=str,
        default=None,
        help="Layer name to extract embeddings from (auto-detected if not specified)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Number of SNVs to process before updating progress"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run on (cuda or cpu)"
    )

    args = parser.parse_args()

    # Check if CUDA is available
    if args.device == "cuda" and not torch.cuda.is_available():
        print("Warning: CUDA not available, falling back to CPU (will be very slow)")
        args.device = "cpu"

    generate_dna_embeddings(
        parquet_path=args.parquet_path,
        output_path=args.output_path,
        model_name=args.model_name,
        layer_name=args.layer_name,
        batch_size=args.batch_size,
        device=args.device
    )


if __name__ == "__main__":
    main()
