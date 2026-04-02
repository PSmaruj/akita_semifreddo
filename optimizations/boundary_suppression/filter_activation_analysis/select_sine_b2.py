"""
select_sine_b2.py

Select 300 B2_Mm2 SINE elements with and without overlapping CTCF motifs
from the mm10 RepeatMasker annotation, and export them as TSV files.

Outputs:
    sineB2_with_ctcf_300.tsv
    sineB2_no_ctcf_300.tsv
"""

import argparse
import bioframe
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
RMSK_FILE    = "/project2/fudenber_735/genomes/mm10/database/rmsk.txt.gz"
CTCF_FILE    = "/project2/fudenber_735/motifs/mm10/jaspar/MA0139.1.tsv.gz"
OUT_WITH_CTCF = "/home1/smaruj/akita_semifreddo/data/sine_b2_tables/sineB2_with_ctcf_300.tsv"
OUT_NO_CTCF   = "/home1/smaruj/akita_semifreddo/data/sine_b2_tables/sineB2_no_ctcf_300.tsv"

RMSK_COLUMNS = [
    "bin", "swScore", "milliDiv", "milliDel", "milliIns",
    "genoName", "genoStart", "genoEnd", "genoLeft", "strand",
    "repName", "repClass", "repFamily", "repStart", "repEnd", "repLeft", "id",
]
STANDARD_CHROMS = [f"chr{i}" for i in range(1, 20)] + ["chrX", "chrY"]
TARGET_REPEAT   = "B2_Mm2"
N_SAMPLE        = 300
RANDOM_STATE    = 42

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_rmsk(path: str) -> pd.DataFrame:
    """Load RepeatMasker flat file (plain or gzipped) and return a DataFrame."""
    compression = "gzip" if path.endswith(".gz") else None
    df = pd.read_csv(path, sep="\t", compression=compression, header=None)
    if df.shape[1] == len(RMSK_COLUMNS):
        df.columns = RMSK_COLUMNS
    else:
        raise ValueError(
            f"Expected {len(RMSK_COLUMNS)} columns in rmsk file, got {df.shape[1]}"
        )
    return df


def load_ctcf(path: str) -> pd.DataFrame:
    """Load JASPAR CTCF motif hits."""
    df = pd.read_csv(
        path, sep="\t", compression="gzip", header=None,
        names=["chrom", "start", "end", "name", "score", "score2", "strand"],
    )
    df["start"] = df["start"].astype(int)
    df["end"]   = df["end"].astype(int)
    return df


def split_by_ctcf_overlap(
    repeats: pd.DataFrame, ctcf: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split *repeats* into two DataFrames based on overlap with *ctcf* intervals.

    Returns
    -------
    with_ctcf, no_ctcf : DataFrames (original index preserved)
    """
    overlap = bioframe.overlap(
        repeats, ctcf,
        how="left",
        suffixes=("", "_ctcf"),
        cols1=("genoName", "genoStart", "genoEnd"),
        cols2=("chrom", "start", "end"),
        return_index=True,
    )
    has_ctcf_mask = overlap["chrom_ctcf"].notna().values
    with_ctcf = repeats.loc[overlap.loc[has_ctcf_mask,  "index"]].copy()
    no_ctcf   = repeats.loc[overlap.loc[~has_ctcf_mask, "index"]].copy()
    return with_ctcf, no_ctcf


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args):
    # Load data
    print("Loading RepeatMasker annotations...")
    rmsk = load_rmsk(args.rmsk)
    print(f"  {len(rmsk):,} total repeat annotations loaded")

    print("Loading CTCF motif hits...")
    ctcf = load_ctcf(args.ctcf)
    print(f"  {len(ctcf):,} CTCF motif hits loaded")

    # Filter to target repeat element and standard chromosomes
    sine_b2 = (
        rmsk[rmsk["repName"] == TARGET_REPEAT]
        .reset_index()
        .rename(columns={"index": "rmsk_index"})
    )
    sine_b2 = sine_b2[sine_b2["genoName"].isin(STANDARD_CHROMS)]
    print(f"\n{TARGET_REPEAT} elements on standard chroms: {len(sine_b2):,}")

    # Split by CTCF overlap
    with_ctcf, no_ctcf = split_by_ctcf_overlap(sine_b2, ctcf)
    print(f"  With CTCF:    {len(with_ctcf):,}")
    print(f"  Without CTCF: {len(no_ctcf):,}")

    # Sample
    with_ctcf_300 = with_ctcf.sample(args.n, random_state=args.seed)
    no_ctcf_300   = no_ctcf.sample(args.n,   random_state=args.seed)

    # Report element lengths
    for label, df in [("with CTCF", with_ctcf_300), ("without CTCF", no_ctcf_300)]:
        lengths = df["genoEnd"] - df["genoStart"]
        print(
            f"\n{TARGET_REPEAT} {label} (n={len(df)}): "
            f"min={lengths.min()}, median={lengths.median():.0f}, max={lengths.max()}"
        )

    # Export
    with_ctcf_300.to_csv(args.out_with_ctcf, sep="\t", index=False)
    no_ctcf_300.to_csv(args.out_no_ctcf,     sep="\t", index=False)
    print(f"\nSaved {len(with_ctcf_300)} rows → {args.out_with_ctcf}")
    print(f"Saved {len(no_ctcf_300)} rows → {args.out_no_ctcf}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rmsk",          default=RMSK_FILE,      help="RepeatMasker TSV (plain or .gz)")
    parser.add_argument("--ctcf",          default=CTCF_FILE,      help="CTCF motif hits TSV.gz")
    parser.add_argument("--out-with-ctcf", default=OUT_WITH_CTCF,  help="Output path for with-CTCF table")
    parser.add_argument("--out-no-ctcf",   default=OUT_NO_CTCF,    help="Output path for no-CTCF table")
    parser.add_argument("--n",             default=N_SAMPLE, type=int, help="Samples per group (default: 300)")
    parser.add_argument("--seed",          default=RANDOM_STATE, type=int, help="Random seed (default: 42)")
    args = parser.parse_args()
    main(args)