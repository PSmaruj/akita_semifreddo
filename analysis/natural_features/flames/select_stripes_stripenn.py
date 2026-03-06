"""
select_stripes_stripenn.py

Filter and annotate stripes detected by StripeNN, retaining only those with
low missing-data fraction and positive Stripiness scores.

Outputs a TSV of selected stripes with added annotation columns.

python select_stripes_stripenn.py --cool /project2/fudenber_735/GEO/Hsieh2019/4DN/mESC_mm10_4DNFILZ1CPT8.mapq_30.8192.cool \
--stripes /project2/fudenber_735/smaruj/sequence_design/ledidi_semifreddo_akita/analysis/natural_features/flames/stripenn_output/result_filtered.tsv \
--out /project2/fudenber_735/smaruj/sequence_design/ledidi_semifreddo_akita/analysis/natural_features/flames/mouse_selected_stripes.tsv

"""

import argparse

import cooler
import numpy as np
import pandas as pd
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Missing-data annotation
# ---------------------------------------------------------------------------

def annotate_missing_fraction_bbox(
    df: pd.DataFrame,
    cool_path: str,
    res: int,
    halfwin: int = 5,
    use_pixels: bool = False,
    balance: bool = True,
    show_progress: bool = True,
) -> tuple[pd.DataFrame, str]:
    """Annotate each stripe with the fraction of missing data in its bounding box.

    For every intra-chromosomal stripe defined by
    (chr, pos1, pos2) x (chr2, pos3, pos4), computes the fraction of missing
    pixels inside the rectangle, expanded by ±halfwin bins on all four sides.

    Two modes:
      use_pixels=False  — fast proxy based on bins.weight being NaN.
      use_pixels=True   — exact, based on NaN pixels in the (balanced) matrix
                          slice (slower).

    Parameters
    ----------
    df : pd.DataFrame
        Stripenn output table.
    cool_path : str
        Path to the .cool file.
    res : int
        Resolution of the cooler (bin size in bp).
    halfwin : int
        Number of bins to expand the bounding box on each side.
    use_pixels : bool
        If True, fetch the 2-D matrix slice to count NaN pixels.
        If False, use bins.weight as a fast proxy.
    balance : bool
        Whether to use balanced weights when use_pixels=True.
    show_progress : bool
        Show a tqdm progress bar.

    Returns
    -------
    df : pd.DataFrame
        Input DataFrame with an additional annotation column.
    label : str
        Name of the new column.
    """
    clr = cooler.Cooler(cool_path)
    chrom_extents = {chrom: clr.extent(chrom) for chrom in clr.chromnames}

    if not use_pixels:
        bins = clr.bins()[:]
        bad_bins = (
            bins["weight"].isna()
            if "weight" in bins.columns
            else pd.Series(False, index=bins.index)
        )

    def _pos_to_bin_start(chrom: str, pos: int) -> int:
        start, _ = chrom_extents[chrom]
        return start + (pos // res)

    def _pos_to_bin_end(chrom: str, pos: int) -> int:
        """Inclusive bin index for an end coordinate."""
        start, _ = chrom_extents[chrom]
        return start + ((pos - 1) // res)

    rows = df.itertuples(index=False)
    if show_progress:
        rows = tqdm(rows, total=len(df), desc="Annotating missing fractions (bbox)")

    fracs = []
    for row in rows:
        chrom = row.chr
        chrom_start, chrom_end = chrom_extents[chrom]

        i0 = _pos_to_bin_start(chrom, row.pos1)
        i1 = _pos_to_bin_end(chrom, row.pos2)
        j0 = _pos_to_bin_start(chrom, row.pos3)
        j1 = _pos_to_bin_end(chrom, row.pos4)

        # Expand by halfwin, clamped to chromosome extent.
        li = max(i0 - halfwin, chrom_start)
        hi = min(i1 + halfwin, chrom_end - 1)
        lj = max(j0 - halfwin, chrom_start)
        hj = min(j1 + halfwin, chrom_end - 1)

        if hi < li or hj < lj:
            fracs.append(np.nan)
            continue

        if use_pixels:
            sub = clr.matrix(balance=balance, sparse=True)[li : hi + 1, lj : hj + 1].toarray()
            frac_missing = np.isnan(sub).mean()
        else:
            bad_r = bad_bins.iloc[li : hi + 1].to_numpy()
            bad_c = bad_bins.iloc[lj : hj + 1].to_numpy()
            nr, nc = bad_r.size, bad_c.size
            good_r = nr - bad_r.sum()
            good_c = nc - bad_c.sum()
            # A pixel is missing if either its row OR column bin is bad.
            frac_missing = 1.0 - (good_r * good_c) / (nr * nc)

        fracs.append(frac_missing)

    label = (
        f"frac_missing_bbox_pm{halfwin}bins_"
        + ("pixels" if use_pixels else "bins")
    )
    df = df.copy()
    df[label] = fracs
    return df, label


# ---------------------------------------------------------------------------
# Stripe annotation helpers
# ---------------------------------------------------------------------------

def add_geometry_columns(df: pd.DataFrame, akita_window_bins: int = 320, bin_size: int = 2048) -> pd.DataFrame:
    """Add midpoint, triangular-half, and Akita window columns.

    Parameters
    ----------
    df : pd.DataFrame
        Stripe table (after missing-data annotation).
    akita_window_bins : int
        Half-width of the Akita input window in bins (default 320 → 640 total).
    bin_size : int
        Akita bin size in bp (default 2048 bp).
    """
    df = df.copy()

    half_window_bp = akita_window_bins * bin_size

    df["x_mid"] = (df["pos1"] + df["pos2"]) / 2
    df["y_mid"] = (df["pos3"] + df["pos4"]) / 2
    df["triangular_half"] = np.where(df["y_mid"] < df["x_mid"], "upper", "lower")
    df["window_start"] = df["pos1"] - half_window_bp
    df["window_end"]   = df["pos1"] + half_window_bp

    return df


# ---------------------------------------------------------------------------
# Filtering
# ---------------------------------------------------------------------------

def filter_stripes(
    df: pd.DataFrame,
    missing_col: str,
    max_missing_frac: float = 0.1,
    min_stripiness: float = 0.0,
) -> pd.DataFrame:
    """Retain stripes with low missing-data fraction and positive Stripiness.

    Parameters
    ----------
    df : pd.DataFrame
        Annotated stripe table.
    missing_col : str
        Name of the missing-fraction column produced by
        ``annotate_missing_fraction_bbox``.
    max_missing_frac : float
        Upper bound on the missing-data fraction (exclusive).
    min_stripiness : float
        Lower bound on Stripiness (exclusive).
    """
    mask = (df[missing_col] < max_missing_frac) & (df["Stripiness"] > min_stripiness)
    return df[mask].reset_index(drop=True)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Filter Stripenn stripes by missing-data fraction and Stripiness."
    )
    p.add_argument("--cool", required=True, help="Path to .cool file.")
    p.add_argument("--stripes", required=True, help="Path to Stripenn result_filtered.tsv.")
    p.add_argument("--out", required=True, help="Path for the output TSV.")
    p.add_argument("--res", type=int, default=8192, help="Cooler resolution in bp (default: 8192).")
    p.add_argument("--halfwin", type=int, default=5, help="Bins to expand bounding box (default: 5).")
    p.add_argument("--max-missing", type=float, default=0.1,
                   help="Max allowed missing-data fraction (default: 0.1).")
    p.add_argument("--min-stripiness", type=float, default=0.0,
                   help="Minimum Stripiness score (default: 0.0, i.e. > 0).")
    p.add_argument("--use-pixels", action="store_true",
                   help="Use exact pixel-level NaN detection (slower).")
    p.add_argument("--no-balance", action="store_true",
                   help="Disable balancing when --use-pixels is set.")
    p.add_argument("--akita-window-bins", type=int, default=320,
                   help="Akita half-window size in bins (default: 320 → 640 total).")
    p.add_argument("--akita-bin-size", type=int, default=2048,
                   help="Akita bin size in bp (default: 2048).")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    print(f"Loading stripes from: {args.stripes}")
    stripes_df = pd.read_csv(args.stripes, sep="\t")
    print(f"  {len(stripes_df):,} stripes loaded.")

    print(f"Loading cooler: {args.cool}")
    annotated_df, missing_col = annotate_missing_fraction_bbox(
        df=stripes_df,
        cool_path=args.cool,
        res=args.res,
        halfwin=args.halfwin,
        use_pixels=args.use_pixels,
        balance=not args.no_balance,
        show_progress=True,
    )

    selected_df = filter_stripes(
        annotated_df,
        missing_col=missing_col,
        max_missing_frac=args.max_missing,
        min_stripiness=args.min_stripiness,
    )
    print(f"  {len(selected_df):,} stripes pass filters.")

    selected_df = add_geometry_columns(
        selected_df,
        akita_window_bins=args.akita_window_bins,
        bin_size=args.akita_bin_size,
    )

    selected_df.to_csv(args.out, sep="\t", index=False)
    print(f"Saved to: {args.out}")


if __name__ == "__main__":
    main()