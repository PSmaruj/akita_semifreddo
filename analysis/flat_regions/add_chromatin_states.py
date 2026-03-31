"""
add_chromatin_states.py

Adds chromatin state (ChromHMM) annotations to flat-region windows.

Usage:
    python add_chromatin_states.py --folds 0 1 2 3 4 5 6 7
"""

import argparse
import os

import bioframe as bf
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(
        description="Annotate flat-region windows with ChromHMM states."
    )
    parser.add_argument(
        "--folds",
        nargs="+",
        type=int,
        default=list(range(8)),
        help="Fold indices to annotate (default: 0–7).",
    )
    parser.add_argument(
        "--input_dir",
        default="/project2/fudenber_735/smaruj/sequence_design/ledidi_semifreddo_akita/analysis/flat_regions/mouse_flat_regions_tsv",
        help="Directory containing fold TSVs from get_flat_regions.py.",
    )
    parser.add_argument(
        "--output_dir",
        default="/project2/fudenber_735/smaruj/sequence_design/ledidi_semifreddo_akita/analysis/flat_regions/mouse_flat_regions_chrom_states_tsv",
        help="Directory to write annotated TSV files.",
    )
    parser.add_argument(
        "--chromhmm_bed",
        default=(
            "/project2/fudenber_735/smaruj/sequence_design/"
            "ledidi_semifreddo_akita/analysis/flat_regions/"
            "mESC_mm10_3states_H3K27ac_9ac_9me3_chromHMM.bed"
        ),
        help="ChromHMM BED file (3-state: 1=active, 2=neutral, 3=repressive).",
    )
    parser.add_argument("--cropping", type=int, default=64)
    parser.add_argument("--bin_size", type=int, default=2048)
    return parser.parse_args()


STATE_MAP = {1: "active", 2: "neutral", 3: "repressive"}


def load_chromhmm(bed_path):
    chromhmm_df = pd.read_csv(bed_path, sep="\t", header=None)
    chromhmm_df.columns = [
        "chrom", "start", "end", "state",
        "score", "strand", "thickStart", "thickEnd", "rgb",
    ]
    chromhmm_df["state_label"] = chromhmm_df["state"].map(STATE_MAP)
    return chromhmm_df


def annotate_fold(fold, chromhmm_df, args):
    print(f"\nAnnotating fold {fold} ...")

    in_path = os.path.join(
        args.input_dir,
        f"fold{fold}_selected_genomic_windows_centered.tsv",
    )
    if not os.path.exists(in_path):
        print(f"  Input not found, skipping: {in_path}")
        return

    df = pd.read_csv(in_path, sep="\t")

    # Rename original start/end to avoid collision with flat-region coordinates
    df = df.rename(columns={"start": "og_start", "end": "og_end"})

    # Compute genomic coordinates of the flat region itself
    df["start"] = (
        df["centered_start"]
        + (df["centered_flat_start"] + args.cropping) * args.bin_size
    )
    df["end"] = (
        df["centered_start"]
        + (df["centered_flat_end"] + args.cropping) * args.bin_size
    )

    # Overlap with ChromHMM
    overlap_df = bf.overlap(
        df,
        chromhmm_df,
        return_index=True,
        suffixes=("_query", "_chromhmm"),
    )

    # Keep relevant columns
    keep_cols = [
        "chrom_query",
        "fold_query",
        "PearsonR_query",
        "centered_start_query",
        "centered_end_query",
        "centered_flat_start_query",
        "centered_flat_end_query",
        "state_chromhmm",
        "state_label_chromhmm",
    ]
    result = overlap_df[keep_cols]

    # Count bins per state per window
    grouped = (
        result.groupby(
            [c for c in keep_cols if c not in ("state_chromhmm", "state_label_chromhmm")]
            + ["state_chromhmm", "state_label_chromhmm"]
        )
        .size()
        .reset_index(name="count")
    )

    index_cols = [
        "chrom_query",
        "fold_query",
        "PearsonR_query",
        "centered_start_query",
        "centered_end_query",
        "centered_flat_start_query",
        "centered_flat_end_query",
    ]

    pivoted = grouped.pivot_table(
        index=index_cols,
        columns="state_label_chromhmm",
        values="count",
        fill_value=0,
    ).reset_index()
    pivoted.columns.name = None

    # Ensure all three state columns exist
    for label in ["active", "neutral", "repressive"]:
        if label not in pivoted.columns:
            pivoted[label] = 0
    pivoted = pivoted.rename(columns={
        "active": "active_count",
        "neutral": "neutral_count",
        "repressive": "repressive_count",
    })

    pivoted["total"] = pivoted[
        ["active_count", "neutral_count", "repressive_count"]
    ].sum(axis=1)

    for label in ["active", "neutral", "repressive"]:
        pivoted[f"{label}_fraction"] = pivoted[f"{label}_count"] / pivoted["total"]

    pivoted.drop(
        columns=["active_count", "neutral_count", "repressive_count", "total"],
        inplace=True,
    )

    pivoted.rename(
        columns={
            "chrom_query": "chrom",
            "fold_query": "fold",
            "PearsonR_query": "PearsonR",
            "centered_start_query": "centered_start",
            "centered_end_query": "centered_end",
            "centered_flat_start_query": "centered_flat_start",
            "centered_flat_end_query": "centered_flat_end",
        },
        inplace=True,
    )

    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(
        args.output_dir,
        f"fold{fold}_selected_genomic_windows_centered_chrom_states.tsv",
    )
    pivoted.to_csv(out_path, sep="\t", index=False)
    print(f"  Saved → {out_path}  ({len(pivoted)} windows)")


def main():
    args = parse_args()
    chromhmm_df = load_chromhmm(args.chromhmm_bed)
    print(f"Loaded ChromHMM: {len(chromhmm_df)} entries")

    for fold in args.folds:
        annotate_fold(fold, chromhmm_df, args)

    print("\nDone.")


if __name__ == "__main__":
    main()