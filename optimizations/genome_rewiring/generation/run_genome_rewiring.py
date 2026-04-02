"""
run_genome_rewiring.py

Runs Ledidi genome rewiring on all windows of a given fold.
Each source window is optimised so that Akita v2 predicts a contact
map resembling that of the next window in the fold (circular shift).

Outputs
-------
- One .pt file per window containing the optimised OHE sequence
- A TSV summarising the optimization (last accepted step per window)

Usage
-----
See run_genome_rewiring.sh
"""

import argparse
import os
import sys

import pandas as pd
import torch

# --- path setup -----------------------------------------------------------
# Cloned ledidi repo takes priority over any pip-installed version
sys.path.insert(0, os.path.abspath("/home1/smaruj/ledidi/"))

# Akita model package — appended so its utils/ does NOT shadow ledidi_akita/utils/
sys.path.append(os.path.abspath("/home1/smaruj/pytorch_akita/"))

# Project root — gives access to utils/
sys.path.insert(0, os.path.abspath("/home1/smaruj/ledidi_akita/"))
# --------------------------------------------------------------------------

from ledidi import ledidi
from utils.model_utils import load_model
from utils.df_utils import build_optimization_table

# ==========================================================================
# Helpers
# ==========================================================================

def load_tensor(path: str, device: torch.device) -> torch.Tensor:
    return torch.load(path, weights_only=True, map_location=device)


# ==========================================================================
# CLI
# ==========================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Ledidi genome rewiring on selected genomic loci."
    )
    parser.add_argument("--fold",               type=int, required=True)
    parser.add_argument("--model_path",         type=str, required=True)
    parser.add_argument("--input_dir",          type=str, required=True)
    parser.add_argument("--output_dir",         type=str, required=True)
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

    df = pd.read_csv(f"{input_dir}/df_select_fold{FOLD}.tsv", sep="\t")
    df = build_optimization_table(df)
    df.to_csv(
        f"{output_dir}/genomic_optimization_fold{FOLD}.tsv",
        sep="\t", index=False,
    )

    for i, row in enumerate(df.itertuples(index=False)):
        chrom,  pred_start,   pred_end   = row.chrom,        row.start,        row.end
        tchrom, target_start, target_end = row.target_chrom, row.target_start, row.target_end

        print(f"\n[{i+1}/{len(df)}] {chrom}:{pred_start}-{pred_end}  →  {tchrom}:{target_start}-{target_end}")

        X = load_tensor(
            f"{input_dir}/ohe_X_fold{FOLD}/{chrom}_{pred_start}_{pred_end}_X.pt",
            device,
        )  # (1, 4, seq_len)

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
        # generated_seq : (1, 4, seq_len)
        # history       : dict — input_loss, output_loss, total_loss, edits, batch_size

        # last_accepted_step: index of the final improvement before early stopping
        total_losses = history["total_loss"]
        last_accepted = int(min(range(len(total_losses)), key=lambda k: total_losses[k]))
        df.at[i, "last_accepted_step"] = last_accepted

        torch.save(
            generated_seq.cpu(),
            os.path.join(results_dir, f"{chrom}_{pred_start}_{pred_end}_seq.pt"),
        )
        
        del generated_seq, history, X, y_bar
        torch.cuda.empty_cache()
        
    df.to_csv(
        f"{output_dir}/genomic_optimization_results_fold{FOLD}_with_steps.tsv",
        sep="\t", index=False,
    )
    print("\nDone.")


if __name__ == "__main__":
    main()