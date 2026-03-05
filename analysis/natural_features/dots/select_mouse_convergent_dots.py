"""
select_mouse_convergent_dots.py

Filters mouse Hi-C loop dots (Mustache calls) to retain high-confidence,
model-compatible entries, then annotates each anchor with overlapping CTCF
motif orientations. Keeps only dots whose anchors carry CTCFs in convergent
orientation (anchor1: '+', anchor2: '-'), as expected for canonical cohesin-
extruded loops.

Output: TSV with anchor coordinates, CTCF strand annotations, prediction
window coordinates, and per-anchor bin positions in the contact map.

Usage:
    python select_mouse_convergent_dots.py
"""

import pandas as pd
import bioframe as bf

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

CHROM_SIZES_FILE = "/project2/fudenber_735/genomes/mm10/mm10.chrom.sizes.reduced"
DOT_FILE = (
    "/project2/fudenber_735/GEO/bonev_2017_GSE96107/distiller-0.3.1_mm10"
    "/results/coolers/features/mustache_HiC_ES.mm10.mapq_30.10000.tsv"
)
JASPAR_FILE = "/project2/fudenber_735/motifs/mm10/jaspar/MA0139.1.tsv.gz"
OUTPUT_TSV = "/project2/fudenber_735/smaruj/sequence_design/ledidi_semifreddo_akita/analysis/natural_features/dots/mouse_convergent_dots.tsv"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SEQ_LENGTH      = 1_310_720
BIN_SIZE        = 2_048
CROPPING_BINS   = 64
AUTOSOMES_ONLY  = True
CHROMS_TO_DROP  = ["chrX", "chrY", "chrM"]

# Anchor 2 is fixed at bin 384 in the prediction window.
# Dots called at 10 kb resolution → ~5 bins wide; anchor starts 2 bins before center.
ANCHOR2_CENTER_BIN  = 384
ANCHOR2_START_BIN   = ANCHOR2_CENTER_BIN - 2           # = 382
MAX_ANCHOR_DIST_BINS = ANCHOR2_CENTER_BIN               # 3/4 of map width

# Genomic offset of anchor2's start from the window start
REL_ANCHOR2_START = (ANCHOR2_START_BIN + CROPPING_BINS) * BIN_SIZE  # bp


# ---------------------------------------------------------------------------
# Filtering helpers
# ---------------------------------------------------------------------------

def drop_chroms(df: pd.DataFrame, chroms: list[str], col: str = "chrom") -> pd.DataFrame:
    """Remove rows whose *col* value is in *chroms*."""
    return df[~df[col].isin(chroms)].copy()


def filter_within_chrom_bounds(
    df: pd.DataFrame,
    chrom_sizes_file: str,
    buffer_bp: int = 0,
) -> pd.DataFrame:
    """
    Remove dots where either anchor exceeds chromosome bounds (± buffer_bp).

    Expects columns: chrom, BIN1_START, BIN1_END, BIN2_START, BIN2_END.
    """
    sizes = pd.read_csv(chrom_sizes_file, sep="\t", header=None, names=["chrom", "size"])
    size_map = dict(zip(sizes["chrom"], sizes["size"]))

    def _valid(row):
        chrom_len = size_map.get(row["chrom"])
        if chrom_len is None:
            return False
        lo, hi = buffer_bp, chrom_len - buffer_bp
        return (
            lo <= row["BIN1_START"] < row["BIN1_END"] <= hi
            and lo <= row["BIN2_START"] < row["BIN2_END"] <= hi
        )

    return df[df.apply(_valid, axis=1)].copy()


def filter_by_anchor_distance(df: pd.DataFrame, max_bins: int) -> pd.DataFrame:
    """
    Keep dots where the inter-anchor distance is within the model's prediction
    window (≤ max_bins × BIN_SIZE bp).
    """
    max_dist_bp = max_bins * BIN_SIZE
    return df[abs(df["BIN1_START"] - df["BIN2_START"]) <= max_dist_bp].copy()


def enforce_anchor_order(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure BIN1_START ≤ BIN2_START; swap anchor columns where needed."""
    df = df.copy()
    mask = df["BIN1_START"] > df["BIN2_START"]
    for c1, c2 in [("BIN1_START", "BIN2_START"), ("BIN1_END", "BIN2_END")]:
        df.loc[mask, [c1, c2]] = df.loc[mask, [c2, c1]].values
    return df


# ---------------------------------------------------------------------------
# CTCF annotation
# ---------------------------------------------------------------------------

def annotate_ctcf_strands(
    dots_df: pd.DataFrame,
    jaspar_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Intersect each dot anchor with JASPAR CTCF motifs and attach the
    strand(s) of any overlapping motifs as comma-separated strings.

    Convergent orientation means anchor1 carries '+' and anchor2 carries '-'.
    Uses bioframe.overlap (no bedtools binary required).
    """
    ctcf = jaspar_df[["chrom", "start", "end", "strand"]].copy()

    def _intersect_strands(dots_df, start_col, end_col, col_name):
        anchor = dots_df[["chrom", start_col, end_col]].rename(
            columns={start_col: "start", end_col: "end"}
        )
        hits = bf.overlap(anchor, ctcf, suffixes=("", "_ctcf"), return_index=True)
        strands = (
            hits.groupby("index")["strand_ctcf"]
            .agg(lambda x: ",".join(sorted(set(x.dropna()))))
            .rename(col_name)
        )
        return strands

    a1_strands = _intersect_strands(dots_df, "anchor1_start", "anchor1_end", "anchor1_ctcf_strand")
    a2_strands = _intersect_strands(dots_df, "anchor2_start", "anchor2_end", "anchor2_ctcf_strand")

    annotated = dots_df.join(a1_strands, how="left").join(a2_strands, how="left")
    annotated["anchor1_ctcf_strand"] = annotated["anchor1_ctcf_strand"].fillna("None")
    annotated["anchor2_ctcf_strand"] = annotated["anchor2_ctcf_strand"].fillna("None")
    return annotated


def filter_convergent(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep only dots with convergent CTCF orientation:
    anchor1 overlaps '+' strand and anchor2 overlaps '-' strand.
    Both anchors may also carry additional motifs of the other strand;
    the requirement is that the expected strand is present.
    """
    has_fwd = df["anchor1_ctcf_strand"].str.contains(r"\+", regex=True)
    has_rev = df["anchor2_ctcf_strand"].str.contains(r"\-", regex=True)
    return df[has_fwd & has_rev].copy()


# ---------------------------------------------------------------------------
# Window coordinate computation
# ---------------------------------------------------------------------------

def add_window_coords(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute prediction window [window_start, window_end) by placing anchor2
    at ANCHOR2_START_BIN (bin 382) within the 512-bin output map.

    Also adds:
      anchor2_center_bin  – fixed at ANCHOR2_CENTER_BIN (384)
      anchors_dist        – inter-anchor distance in bins
      anchor1_center_bin  – bin position of anchor1 within the window
    """
    df = df.copy()
    df["window_start"] = df["anchor2_start"] - REL_ANCHOR2_START
    df["window_end"]   = df["window_start"] + SEQ_LENGTH

    df["anchor2_center_bin"] = ANCHOR2_CENTER_BIN
    df["anchors_dist"]       = (df["anchor2_start"] - df["anchor1_start"]) // BIN_SIZE
    df["anchor1_center_bin"] = df["anchor2_center_bin"] - df["anchors_dist"]

    assert (df["window_end"] - df["window_start"] == SEQ_LENGTH).all(), \
        "Window length mismatch — check anchor coordinate computation."
    assert (df["anchor1_center_bin"] >= 0).all(), \
        "Negative anchor1 bin — some dots exceed the map's left edge."

    return df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    # -- Load dots -----------------------------------------------------------
    dots = pd.read_csv(DOT_FILE, sep="\t")
    print(f"Loaded {len(dots):,} dots")

    # Mustache outputs BIN2_CHROMOSOME redundantly; drop and standardise
    dots = dots.drop(columns=["BIN2_CHROMOSOME"]).rename(columns={"BIN1_CHR": "chrom"})

    # -- Filter --------------------------------------------------------------
    if AUTOSOMES_ONLY:
        dots = drop_chroms(dots, CHROMS_TO_DROP)

    dots = filter_within_chrom_bounds(dots, CHROM_SIZES_FILE, buffer_bp=SEQ_LENGTH)
    dots = filter_by_anchor_distance(dots, max_bins=MAX_ANCHOR_DIST_BINS)
    dots = enforce_anchor_order(dots)
    dots = dots.reset_index(drop=True)
    print(f"After coordinate filters: {len(dots):,} dots")

    dots = dots.rename(columns={
        "BIN1_START": "anchor1_start", "BIN1_END": "anchor1_end",
        "BIN2_START": "anchor2_start", "BIN2_END": "anchor2_end",
    })

    # -- CTCF annotation and convergent filter -------------------------------
    jaspar_df = bf.read_table(JASPAR_FILE, schema="jaspar", skiprows=1)
    if AUTOSOMES_ONLY:
        jaspar_df = drop_chroms(jaspar_df, CHROMS_TO_DROP)
    jaspar_df = jaspar_df.reset_index(drop=True)

    dots = annotate_ctcf_strands(dots, jaspar_df)
    dots = filter_convergent(dots)
    dots = dots.reset_index(drop=True)
    print(f"After convergent CTCF filter: {len(dots):,} dots")

    # -- Window coordinates --------------------------------------------------
    dots = add_window_coords(dots)

    # -- Save ----------------------------------------------------------------
    dots.to_csv(OUTPUT_TSV, sep="\t", index=False)
    print(f"Saved to:\n  {OUTPUT_TSV}")


if __name__ == "__main__":
    main()