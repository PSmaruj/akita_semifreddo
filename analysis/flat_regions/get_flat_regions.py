"""
get_flat_regions.py

Pipeline for identifying flat insulation regions in predicted Hi-C maps.

Iterates over all folds, selects high-quality predictions (PearsonR > species threshold),
computes insulation profiles, detects flat regions, recenters windows, and saves results.

Usage:
    python get_flat_regions.py --species mouse --folds 0 1 2 3 4 5 6 7
    python get_flat_regions.py --species human --folds 0 1 2 3 4 5 6 7
"""

import argparse
import os
import sys
import glob

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

# Project root — gives access to utils/
sys.path.insert(0, os.path.abspath("/home1/smaruj/ledidi_akita/"))
# --------------------------------------------------------------------------

from utils.dataset_utils import HiCDataset
from utils.data_utils import from_upper_triu
from utils.insulation_utils import (insulation_full,
                                    find_longest_flat_region,
                                    recenter_flat_region,
                                    remove_close_regions
)
from scipy.stats import pearsonr

# Fixed PearsonR thresholds — average across all folds per species
PEARSON_THRESHOLD = {
    "mouse": 0.663,
    "human": 0.674,
}

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Detect flat insulation regions fold by fold.")

    parser.add_argument(
        "--species",
        choices=["mouse", "human"],
        required=True,
        help="Species to run (mouse or human).",
    )
    parser.add_argument(
        "--folds",
        nargs="+",
        type=int,
        default=list(range(8)),
        help="Fold indices to process (default: 0-7).",
    )

    # Paths — override defaults if needed
    parser.add_argument(
        "--akita_repo",
        default="/home1/smaruj/pytorch_akita/",
        help="Path to akita repo (for model import).",
    )
    parser.add_argument(
        "--model_path_mouse",
        default=(
            "/home1/smaruj/pytorch_akita/models/finetuned/mouse/Hsieh2019_mESC/checkpoints/Akita_v2_mouse_Hsieh2019_mESC_model0_finetuned.pth"
        ),
        help="Path to finetuned mouse model checkpoint.",
    )
    parser.add_argument(
        "--model_path_human",
        default=(
            "/home1/smaruj/pytorch_akita/models/finetuned/human/Krietenstein2019_HFF/checkpoints/Akita_v2_human_Krietenstein2019_HFF_model0_finetuned.pth"
        ),
        help="Path to finetuned human model checkpoint.",
    )
    parser.add_argument(
        "--bed_file_mouse",
        default="/project2/fudenber_735/tensorflow_models/akita/v2/data/mm10/sequences.bed",
        help="BED file with mm10 sequence windows.",
    )
    parser.add_argument(
        "--bed_file_human",
        default="/project2/fudenber_735/tensorflow_models/akita/v2/data/hg38/sequences.bed",
        help="BED file with hg38 sequence windows.",
    )
    parser.add_argument(
        "--data_dir_mouse",
        default="/project2/fudenber_735/smaruj/sequence_design/ledidi_semifreddo_akita/Akita_pytorch_training_data/mouse_training_data/Hsieh2019_mESC",
        help="Directory containing mouse .pt data files.",
    )
    parser.add_argument(
        "--data_dir_human",
        default="/project2/fudenber_735/smaruj/sequence_design/ledidi_semifreddo_akita/Akita_pytorch_training_data/human_training_data/Krietenstein2019_HFF",
        help="Directory containing human .pt data files.",
    )
    parser.add_argument(
        "--output_dir",
        default="/project2/fudenber_735/smaruj/sequence_design/ledidi_semifreddo_akita/analysis/flat_regions/human_flat_regions_tsv",
        help="Directory to write output TSV files.",
    )

    # Detection parameters
    parser.add_argument("--matrix_len", type=int, default=512)
    parser.add_argument("--num_diags", type=int, default=2)
    parser.add_argument("--insulation_window", type=int, default=16,
                        help="Half-width of diamond window for insulation score.")
    parser.add_argument("--std_window", type=int, default=40)
    parser.add_argument("--std_threshold", type=float, default=0.025)
    parser.add_argument("--min_flat_length", type=int, default=100)
    parser.add_argument("--edge_margin", type=int, default=50)
    parser.add_argument("--min_spacing", type=int, default=300_000,
                        help="Min genomic spacing (bp) between retained windows.")
    parser.add_argument("--cropping", type=int, default=64)
    parser.add_argument("--bin_size", type=int, default=2048)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Per-fold pipeline
# ---------------------------------------------------------------------------

def process_fold(fold, model, device, df_all, args, pearson_threshold):
    print(f"\n{'='*60}")
    print(f"Processing fold {fold}")
    print(f"{'='*60}")

    # Select genomic windows for this fold
    df_fold = df_all[df_all["fold"] == f"fold{fold}"].reset_index(drop=True)

    # Infer the files from the directory
    data_dir = args.data_dir_mouse if args.species == "mouse" else args.data_dir_human

    search_pattern = os.path.join(data_dir, f"fold{fold}_*.pt")
    data_files = sorted(glob.glob(search_pattern))

    if not data_files:
        print(f"  Warning: No data files found for fold {fold} in {data_dir}")
        return None

    print(f"  Found {len(data_files)} data files for fold {fold}")

    dataset = HiCDataset(data_files)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # Run model
    all_preds, all_targets = [], []
    with torch.no_grad():
        for ohe_seq, hic_vec in loader:
            outputs = model(ohe_seq.to(device))
            all_preds.append(outputs.cpu())
            all_targets.append(hic_vec.cpu())

    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    targets_np = all_targets.numpy()
    preds_np = all_preds.numpy()

    # Per-sample Pearson R
    r_vals = np.array([
        pearsonr(targets_np[i].flatten(), preds_np[i].flatten())[0]
        for i in range(len(targets_np))
    ])

    # Align df with predictions
    n = min(len(df_fold), len(r_vals))
    df_fold = df_fold.iloc[:n].copy()
    df_fold["PearsonR"] = r_vals[:n]

    # Filter high-quality predictions using fixed species threshold
    highR_df = df_fold[df_fold["PearsonR"] > pearson_threshold].copy()
    print(f"  PearsonR threshold: {pearson_threshold}  ({len(highR_df)} / {len(df_fold)} maps pass)")

    # Detect flat regions
    flat_starts, flat_ends = [], []
    for idx in highR_df.index:
        pred_mat = from_upper_triu(preds_np[idx], args.matrix_len, args.num_diags)
        pred_insul = insulation_full(pred_mat, args.insulation_window)
        fs, fe = find_longest_flat_region(
            pred_insul,
            std_window=args.std_window,
            std_threshold=args.std_threshold,
            min_length=args.min_flat_length,
            edge_margin=args.edge_margin,
        )
        flat_starts.append(fs)
        flat_ends.append(fe)

    highR_df["flat_start"] = flat_starts
    highR_df["flat_end"] = flat_ends

    valid_flats_df = highR_df.dropna(subset=["flat_start", "flat_end"]).copy()
    valid_flats_df["flat_start"] = valid_flats_df["flat_start"].astype(int)
    valid_flats_df["flat_end"] = valid_flats_df["flat_end"].astype(int)
    print(f"  {len(valid_flats_df)} windows have a detected flat region")

    # Recenter windows
    recentered = valid_flats_df.apply(
        recenter_flat_region,
        axis=1,
        cropping=args.cropping,
        map_size=args.matrix_len,
        bin_size=args.bin_size,
    )
    valid_flats_df = pd.concat([valid_flats_df, recentered], axis=1)

    # Remove overlapping windows
    non_overlapping_df = remove_close_regions(
        valid_flats_df, min_spacing=args.min_spacing, seed=args.seed
    )
    print(f"  {len(non_overlapping_df)} windows remain after spacing filter")

    # Save output
    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(
        args.output_dir,
        f"fold{fold}_selected_genomic_windows_centered.tsv",
    )
    non_overlapping_df.to_csv(out_path, sep="\t", index=False)
    print(f"  Saved → {out_path}")

    return non_overlapping_df, all_preds, all_targets, df_fold


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    # Add akita repo to path and import model
    sys.path.append(os.path.abspath(args.akita_repo))
    from akita.model import SeqNN

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load model
    model_path = (
        args.model_path_mouse if args.species == "mouse" else args.model_path_human
    )
    model = SeqNN()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    print(f"Loaded model: {model_path}")

    # Load BED file
    bed_file = (
        args.bed_file_mouse if args.species == "mouse" else args.bed_file_human
    )
    df_all = pd.read_csv(bed_file, sep="\t", header=None,
                         names=["chrom", "start", "end", "fold"])

    pearson_threshold = PEARSON_THRESHOLD[args.species]
    print(f"Using fixed PearsonR threshold: {pearson_threshold} ({args.species})")

    # Process each fold
    for fold in args.folds:
        process_fold(fold, model, device, df_all, args, pearson_threshold)

    print("\nDone.")


if __name__ == "__main__":
    main()