"""
run_background_to_genomic.py

Runs Ledidi rewiring from BACKGROUND (flat) sequences into genomic contact maps.
For each optimization, a background sequence is picked at random (with
replacement) from a pool of 10, and Akita v2 is optimised so that its predicted
contact map resembles the genomic target for that row.

Outputs
-------
- One .pt file per target containing the optimised OHE sequence
- A TSV summarising the optimization (last accepted step + background used per row)
"""

import argparse
import os
import sys
import numpy as np
import pandas as pd
import torch

from ledidi import ledidi
from utils.model_utils import load_model
from utils.df_utils import build_optimization_table

# ==========================================================================
# Helpers
# ==========================================================================

def load_tensor(path: str, device: torch.device) -> torch.Tensor:
    return torch.load(path, weights_only=True, map_location=device)

# --- FASTA reading + one-hot encoding -------------------------------
NUC_TO_IDX = {"A": 0, "C": 1, "G": 2, "T": 3}

def read_fasta(path: str) -> dict:
    """Minimal FASTA parser -> {name: sequence}."""
    seqs, name, chunks = {}, None, []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if name is not None:
                    seqs[name] = "".join(chunks)
                name, chunks = line[1:].split()[0], []
            else:
                chunks.append(line)
        if name is not None:
            seqs[name] = "".join(chunks)
    return seqs


def one_hot_encode(seq: str, device: torch.device) -> torch.Tensor:
    """(seq_str) -> (1, 4, seq_len). Non-ACGT bases become all-zero columns."""
    arr = np.frombuffer(seq.upper().encode("ascii"), dtype=np.uint8)
    ohe = np.zeros((4, arr.shape[0]), dtype=np.float32)
    for base, j in NUC_TO_IDX.items():
        ohe[j, arr == ord(base)] = 1.0
    return torch.from_numpy(ohe).unsqueeze(0).to(device)  # (1, 4, seq_len)


# ==========================================================================
# CLI
# ==========================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Ledidi rewiring from background sequences into genomic loci."
    )
    parser.add_argument("--fold",               type=int, required=True)
    parser.add_argument("--model_path",         type=str, required=True)
    parser.add_argument("--input_dir",          type=str, required=True)
    parser.add_argument("--output_dir",         type=str, required=True)
    parser.add_argument("--background_fasta",   type=str, required=True,   # NEW
                        help="FASTA of background sequences to start from")
    parser.add_argument("--seed",               type=int, default=0,       # NEW
                        help="Seed for reproducible background selection")
    parser.add_argument("--max_iter",           type=int, default=2000)
    parser.add_argument("--early_stopping_iter",type=int, default=2000)
    parser.add_argument("--l",                  type=float, default=0.05,
                        help="Input/output loss mixing weight")
    return parser.parse_args()


# ==========================================================================
# Main
# ==========================================================================

def main() -> None:
    args = parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = load_model(args.model_path, device)

    FOLD       = args.fold
    input_dir  = args.input_dir
    output_dir = args.output_dir

    results_dir = os.path.join(output_dir, f"results_fold{FOLD}")
    os.makedirs(results_dir, exist_ok=True)

    # --- NEW: load + encode the background pool once -------------------
    bg_seqs = read_fasta(args.background_fasta)
    bg_names = list(bg_seqs.keys())
    lengths = {len(s) for s in bg_seqs.values()}
    assert len(lengths) == 1, f"Background sequences differ in length: {lengths}"
    backgrounds = [one_hot_encode(bg_seqs[n], device) for n in bg_names]
    print(f"Loaded {len(backgrounds)} background sequences (len={lengths.pop()})")

    rng = np.random.default_rng(args.seed)  # reproducible selection

    df = pd.read_csv(f"{input_dir}/df_select_fold{FOLD}.tsv", sep="\t")
    df = build_optimization_table(df)
    df.to_csv(
        f"{output_dir}/genomic_optimization_fold{FOLD}.tsv",
        sep="\t", index=False,
    )

    for i, row in enumerate(df.itertuples(index=False)):
        chrom,  pred_start,   pred_end   = row.chrom,        row.start,        row.end
        tchrom, target_start, target_end = row.target_chrom, row.target_start, row.target_end

        # --- NEW: random background as the starting sequence ----------
        bg_idx = int(rng.integers(0, len(backgrounds)))
        X = backgrounds[bg_idx].clone()  # clone so the pool tensor isn't mutated
        df.at[i, "background_name"]  = bg_names[bg_idx]
        df.at[i, "background_index"] = bg_idx

        print(f"\n[{i+1}/{len(df)}] background '{bg_names[bg_idx]}'  →  {tchrom}:{target_start}-{target_end}")

        y_bar = load_tensor(
            f"{input_dir}/genomic_targets_fold{FOLD}/{tchrom}_{target_start}_{target_end}_target.pt",
            device,
        )  # (1, 1, 130305)

        generated_seq, history = ledidi(
            model, X, y_bar,
            batch_size          = 1,
            l                   = args.l,
            max_iter            = args.max_iter,
            early_stopping_iter = args.early_stopping_iter,
            input_loss          = torch.nn.L1Loss(reduction="sum"),
            output_loss         = torch.nn.L1Loss(reduction="sum"),
            return_history      = True,
            verbose             = True,
            device              = device,
        )

        total_losses = history["total_loss"]
        last_accepted = int(min(range(len(total_losses)), key=lambda k: total_losses[k]))
        df.at[i, "last_accepted_step"] = last_accepted

        torch.save(
            generated_seq.cpu(),
            os.path.join(results_dir, f"{tchrom}_{target_start}_{target_end}_seq.pt"),
        )

        del generated_seq, history, X, y_bar
        torch.cuda.empty_cache()

    df.to_csv(
        f"{output_dir}/genomic_background_fold{FOLD}_with_steps.tsv",
        sep="\t", index=False,
    )
    print("\nDone.")


if __name__ == "__main__":
    main()