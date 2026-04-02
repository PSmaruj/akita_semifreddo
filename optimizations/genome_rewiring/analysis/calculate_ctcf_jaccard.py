"""
calculate_ctcf_jaccard.py

For each genomic window in a fold, computes the Jaccard index of CTCF binding
sites between:
  - the initial (genomic) sequence and the Ledidi-optimised sequence
  - the optimised sequence and the target sequence

Results are written to a TSV file.
"""
import os
import sys
import argparse
import torch
import pandas as pd
from pyfaidx import Fasta
from memelite import fimo

# --- path setup -----------------------------------------------------------
# Cloned ledidi repo takes priority over any pip-installed version
sys.path.insert(0, os.path.abspath("/home1/smaruj/ledidi/"))

# Akita model package — appended so its utils/ does NOT shadow akita_semifreddo/utils/
sys.path.append(os.path.abspath("/home1/smaruj/akita_pytorch/"))

# Project root — gives access to utils/
sys.path.insert(0, os.path.abspath("/home1/smaruj/akita_semifreddo/"))
# --------------------------------------------------------------------------

from utils.df_utils import build_optimization_table
from utils.fimo_utils import read_meme_pwm, hits_to_site_set, jaccard_index


# =============================================================================
# Helpers
# =============================================================================


def scan_ctcf(tensor: torch.Tensor, motifs_dict: dict, threshold: float) -> set:
    """Run FIMO on a (1, 4, L) tensor and return the set of CTCF binding sites."""
    hits = fimo(
        motifs=motifs_dict,
        sequences=tensor.detach().cpu().numpy(),  # ← added .detach()
        threshold=threshold,
        reverse_complement=True,
    )[0]
    return hits_to_site_set(hits)


# =============================================================================
# CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute CTCF Jaccard indices.")
    parser.add_argument("--fold",         type=int,   default=0)
    parser.add_argument("--input_dir",    type=str,   required=True)
    parser.add_argument("--results_dir",  type=str,   required=True,
                        help="Directory containing the optimised seq .pt files")
    parser.add_argument("--output",       type=str,   required=True,
                        help="Path for the output TSV")
    parser.add_argument("--genome",       type=str,
                        default="/project2/fudenber_735/genomes/mm10/mm10.fa")
    parser.add_argument("--ctcf_pwm",     type=str,
                        default="/home1/smaruj/ledidi_akita/data/pwm/MA0139.1.meme")
    parser.add_argument("--threshold",    type=float, default=1e-4)
    return parser.parse_args()


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    args = parse_args()

    df = pd.read_csv(f"{args.input_dir}/df_select_fold{args.fold}.tsv", sep="\t")
    df = build_optimization_table(df)

    pwm_tensor = read_meme_pwm(args.ctcf_pwm).float()    
    motifs     = {"CTCF": pwm_tensor}

    df["Jaccard_init_opt"]    = 0.0
    df["Jaccard_opt_target"]  = 0.0

    for idx, row in df.iterrows():
        # Initial (genomic) sequence
        init_path   = f"{args.input_dir}/ohe_X_fold{args.fold}/{row['chrom']}_{row['start']}_{row['end']}_X.pt"
        init_tensor = torch.load(init_path, weights_only=True)
        init_sites  = scan_ctcf(init_tensor, motifs, args.threshold)
        
        # Optimised sequence
        opt_path   = f"{args.results_dir}/{row['chrom']}_{row['start']}_{row['end']}_seq.pt"
        opt_tensor = torch.load(opt_path, weights_only=True)
        
        if opt_tensor.dim() == 2:          # handle (4, L) saved without batch dim
            opt_tensor = opt_tensor.unsqueeze(0)
        opt_sites  = scan_ctcf(opt_tensor, motifs, args.threshold)

        # Target (genomic) sequence
        tgt_path   = f"{args.input_dir}/ohe_X_fold{args.fold}/{row['target_chrom']}_{row['target_start']}_{row['target_end']}_X.pt"
        tgt_tensor = torch.load(tgt_path, weights_only=True)
        tgt_sites  = scan_ctcf(tgt_tensor, motifs, args.threshold)
        
        df.at[idx, "Jaccard_init_opt"]   = jaccard_index(init_sites, opt_sites)
        df.at[idx, "Jaccard_opt_target"] = jaccard_index(opt_sites,  tgt_sites)

        print(
            f"[{idx+1}/{len(df)}] {row['chrom']}:{row['start']}-{row['end']} "
            f"init↔opt={df.at[idx,'Jaccard_init_opt']:.3f}  "
            f"opt↔target={df.at[idx,'Jaccard_opt_target']:.3f}"
        )

    df.to_csv(args.output, sep="\t", index=False)
    print(f"\nSaved results to {args.output}")

if __name__ == "__main__":
    main()