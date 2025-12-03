"""
Generate protein delta embeddings using base pretrained ProGen2 model.

For each SNV in the parquet file:
    protein_delta_embedding = progen(prot_wt) - progen(prot_mut)

Saves embeddings to npz file with indices matching the parquet file.
"""

import argparse
import os
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from progen.model import ProGenForCausalLM
from progen.progen_vocab import Alphabet


def get_protein_embedding(model, alphabet, protein_seq, device='cuda', max_length=2048):
    """
    Extract embedding from ProGen2 for a single protein sequence.

    Args:
        model: ProGenForCausalLM model
        alphabet: ProGen alphabet for tokenization
        protein_seq: Protein sequence string
        device: Device to run on
        max_length: Maximum total tokens INCLUDING BOS/EOS (model supports 2048)

    Returns:
        embedding: Mean-pooled embedding vector [embedding_dim]

    Raises:
        ValueError: If sequence is longer than max_length after tokenization
    """
    # Encode with BOS and EOS tokens first
    tokens = alphabet.encode_line(protein_seq, prepend_bos=True, append_eos=True).long()

    # Check if sequence is too long (tokens already include BOS and EOS)
    if tokens.size(0) > max_length:
        raise ValueError(f"Sequence too long: {tokens.size(0)} tokens (max: {max_length})")

    tokens = tokens.unsqueeze(0).to(device)  # [1, L]

    # Get embeddings from the transformer
    with torch.no_grad():
        # Get word embeddings and pass through transformer
        x = model.transformer.wte(tokens)
        transformer_outputs = model.transformer(
            input_ids=None,
            inputs_embeds=x,
            output_hidden_states=False,
            return_dict=True
        )
        hidden_states = transformer_outputs.last_hidden_state  # [1, L, dim]

        # Mean pool across sequence length (excluding padding if any)
        # For simplicity, we mean pool all tokens including BOS/EOS
        embedding = hidden_states.mean(dim=1).squeeze(0)  # [dim]

    return embedding.cpu().numpy()


def generate_protein_embeddings(
    parquet_path,
    output_path,
    pretrained_model_path,
    pretrained_model_name="progen2-small",
    batch_size=8,
    max_length=1024,
    device='cuda'
):
    """
    Generate delta embeddings for all SNVs in parquet file.

    Args:
        parquet_path: Path to SNV parquet file
        output_path: Path to save npz file
        pretrained_model_path: Directory containing pretrained model
        pretrained_model_name: Name of pretrained model folder
        batch_size: Number of SNVs to process at once
        max_length: Maximum protein sequence length
        device: Device to run on
    """
    print("Loading SNV data from parquet...")
    df = pd.read_parquet(parquet_path)
    print(f"Loaded {len(df)} SNVs")

    print("\nLoading ProGen2 base pretrained model...")
    alphabet = Alphabet()
    model_path = os.path.join(pretrained_model_path, pretrained_model_name)
    model = ProGenForCausalLM.from_pretrained(model_path, local_files_only=True).to(device)
    model.eval()
    print(f"Model loaded from {model_path}")
    print(f"Embedding dimension: {model.config.n_embd}")

    # Prepare output arrays
    embedding_dim = model.config.n_embd
    delta_embeddings = np.zeros((len(df), embedding_dim), dtype=np.float32)
    variant_ids = df['variant_id'].values

    print("\nGenerating protein delta embeddings...")

    # Process in batches to show progress
    num_batches = (len(df) + batch_size - 1) // batch_size

    failed_indices = []
    for batch_idx in tqdm(range(num_batches), desc="Processing batches"):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(df))

        for i in range(start_idx, end_idx):
            row = df.iloc[i]

            try:
                # Get embeddings for wild-type and mutant proteins
                wt_embedding = get_protein_embedding(
                    model, alphabet, row['prot_wt'], device, max_length
                )
                mut_embedding = get_protein_embedding(
                    model, alphabet, row['prot_mut'], device, max_length
                )

                # Compute delta embedding
                delta_embeddings[i] = wt_embedding - mut_embedding

            except Exception as e:
                # Keep as zeros for failed embeddings
                failed_indices.append(i)
                print(f"\nWarning: Failed to process SNV at index {i} (variant_id={row['variant_id']}): {e}")
                continue

    # Save to npz file
    print(f"\nSaving embeddings to {output_path}...")
    np.savez_compressed(
        output_path,
        delta_embeddings=delta_embeddings,
        variant_ids=variant_ids,
        embedding_dim=embedding_dim,
        model_name=pretrained_model_name,
        failed_indices=np.array(failed_indices) if failed_indices else np.array([])
    )
    print("Done!")
    print(f"Saved {len(delta_embeddings)} protein delta embeddings")
    print(f"Shape: {delta_embeddings.shape}")
    if failed_indices:
        print(f"\nWarning: {len(failed_indices)} SNVs failed (kept as zeros)")
        print(f"Failed indices: {failed_indices}")
    else:
        print(f"\nAll {len(delta_embeddings)} embeddings generated successfully!")


def main():
    parser = argparse.ArgumentParser(
        description="Generate protein delta embeddings using ProGen2"
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
        default="protein_delta_embeddings.npz",
        help="Path to save output npz file"
    )
    parser.add_argument(
        "--pretrained_model_path",
        type=str,
        default="pretrained_model",
        help="Directory containing pretrained ProGen2 model"
    )
    parser.add_argument(
        "--pretrained_model_name",
        type=str,
        default="progen2-small",
        help="Name of pretrained model folder"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Number of SNVs to process before updating progress"
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=2048,
        help="Maximum total tokens INCLUDING BOS/EOS (ProGen2 supports 2048)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run on (cuda or cpu)"
    )

    args = parser.parse_args()

    generate_protein_embeddings(
        parquet_path=args.parquet_path,
        output_path=args.output_path,
        pretrained_model_path=args.pretrained_model_path,
        pretrained_model_name=args.pretrained_model_name,
        batch_size=args.batch_size,
        max_length=args.max_length,
        device=args.device
    )


if __name__ == "__main__":
    main()
