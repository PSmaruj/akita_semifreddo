"""
generate_files_for_ledidi.py

Generates one-hot encoded input sequences and precomputed model targets
for use with the Ledidi sequence optimisation framework.

For each genomic window in a given fold of the Akita v2 BED file:
  - Extracts the DNA sequence from the genome FASTA
  - One-hot encodes it  ->  saved as (1, 4, 1310720)   float32 tensor
  - Runs the frozen model  ->  saved as (1, 1, 130305)  float32 tensor

Both shapes are directly compatible with Ledidi:
  X    (1, n_channels, length)  passed as the sequence to optimise
  y_bar (1, *)                  passed as the desired model output

Usage:
    python generate_optimization_files.py \
        --fold 0 \
        --bed_file /project2/fudenber_735/tensorflow_models/akita/v2/data/mm10/sequences.bed \
        --fasta_file /project2/fudenber_735/genomes/mm10/mm10.fa \
        --model_weights /home1/smaruj/akita_pytorch/models/finetuned/mouse/Hsieh2019_mESC/checkpoints/Akita_v2_mouse_Hsieh2019_mESC_model0_finetuned.pth \
        --out_dir /project2/fudenber_735/smaruj/sequence_design/ledidi_semifreddo_akita/optimizations/genome_rewiring
"""

import argparse
import logging
import os
import sys

import numpy as np
import torch
import pandas as pd
from pyfaidx import Fasta

DEFAULT_MODEL_SRC = "/home1/smaruj/akita_pytorch/"

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))
from utils.data_utils import one_hot_encode_sequence
from utils.model_utils import (load_model, run_model)
from utils.df_utils import load_bed_fold


def make_output_dirs(out_dir: str, fold: int) -> tuple[str, str]:
    """Create output subdirectories for sequences and targets.

    Parameters
    ----------
    out_dir:
        Root output directory.
    fold:
        Fold index, used in subdirectory names.

    Returns
    -------
    tuple[str, str]
        Paths to (sequences_dir, targets_dir).
    """
    seq_dir = os.path.join(out_dir, f"ohe_X_fold{fold}")
    tgt_dir = os.path.join(out_dir, f"genomic_targets_fold{fold}")
    os.makedirs(seq_dir, exist_ok=True)
    os.makedirs(tgt_dir, exist_ok=True)
    return seq_dir, tgt_dir


# ===========================================================================
# Main pipeline
# ===========================================================================

def generate_ledidi_files(
    fold: int,
    bed_file: str,
    fasta_file: str,
    model_weights: str,
    out_dir: str,
    device: torch.device,
) -> None:
    """Generate and save OHE sequences and model targets for one fold.

    Saved tensors
    -------------
    X  : (1, 4, seq_len)      float32  — Ledidi input shape (1, n_channels, length)
    y  : (1, 1, num_contacts) float32  — Ledidi target shape (1, *)

    Both tensors are saved on CPU for portability.

    Parameters
    ----------
    fold:
        Fold index to process.
    bed_file:
        Path to sequences.bed.
    fasta_file:
        Path to genome FASTA.
    model_weights:
        Path to model state dict.
    out_dir:
        Root output directory. Subdirectories are created automatically.
    device:
        Torch device for model inference.
    """
    df_fold = load_bed_fold(bed_file, fold)
    df_fold.to_csv(os.path.join(out_dir, f"df_select_fold{fold}.tsv"), sep="\t", index=False)

    seq_dir, tgt_dir = make_output_dirs(out_dir, fold)
    genome = Fasta(fasta_file)
    model = load_model(model_weights, device)

    for i, row in enumerate(df_fold.itertuples(index=False)):
        chrom, start, end = row.chrom, row.start, row.end
        tag = f"{chrom}_{start}_{end}"

        # --- sequence ---
        sequence = genome[chrom][start:end]
        ohe = one_hot_encode_sequence(sequence)          # (1, 4, seq_len)
        X_tensor = torch.tensor(ohe)                     # (1, 4, seq_len)

        # --- model prediction ---
        y_tensor = run_model(model, X_tensor, device)    # (1, 1, num_contacts) on CPU

        # --- save ---
        torch.save(X_tensor, os.path.join(seq_dir, f"{tag}_X.pt"))
        torch.save(y_tensor, os.path.join(tgt_dir, f"{tag}_target.pt"))

        if (i + 1) % 50 == 0 or i == 0:
            logging.info(
                f"  [{i+1}/{len(df_fold)}] {tag} | "
                f"X: {tuple(X_tensor.shape)} | y: {tuple(y_tensor.shape)}"
            )

    logging.info("Done.")


# ===========================================================================
# CLI
# ===========================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate OHE sequences and Akita targets for Ledidi optimisation."
    )
    parser.add_argument("--fold", type=int, required=True, help="Fold index to process.")
    parser.add_argument("--bed_file", type=str, required=True, help="Path to sequences.bed.")
    parser.add_argument("--fasta_file", type=str, required=True, help="Path to genome FASTA.")
    parser.add_argument("--model_weights", type=str, required=True, help="Path to model .pt file.")
    parser.add_argument("--out_dir", type=str, required=True, help="Root output directory.")
    parser.add_argument(
        "--model_src",
        type=str,
        default=DEFAULT_MODEL_SRC,
        help="Path to the akita_model package root (added to sys.path).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Torch device string (e.g. 'cuda:0', 'cpu'). Defaults to cuda if available.",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)s  %(message)s",
        datefmt="%H:%M:%S",
    )

    args = parse_args()

    # Add model package to path before importing SeqNN
    sys.path.append(os.path.abspath(args.model_src))

    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    logging.info(f"Using device: {device}")

    generate_ledidi_files(
        fold=args.fold,
        bed_file=args.bed_file,
        fasta_file=args.fasta_file,
        model_weights=args.model_weights,
        out_dir=args.out_dir,
        device=device,
    )


if __name__ == "__main__":
    main()