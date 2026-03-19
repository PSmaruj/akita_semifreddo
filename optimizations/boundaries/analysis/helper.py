"""
helper.py
=========
Utility functions for the AkitaSF independent-runs analysis of boundary
optimization results.  Functions are organised into five sections:

  1. Shared helpers      – parsing, filtering, sequence decoding
  2. PWM scoring         – score DNA sequences against a position-weight matrix
  3. CTCF site analysis  – collect, count, and score CTCF sites across runs
  4. Jaccard index       – pairwise reproducibility of CTCF placement
  5. Plotting            – all visualisation functions

Constants
---------
REGION_COLS      : columns that uniquely identify a genomic region
N_RUNS           : expected number of independent runs
CENTER_BIN       : index of the central 2048 bp bin optimized by AkitaSF
BIN_SIZE         : size of one Akita bin in bp
CENTER_BIN_START : genomic offset of the central bin start within the full sequence
CENTER_BIN_END   : genomic offset of the central bin end
"""

import ast
import itertools

from tqdm import tqdm
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrow, Patch
from pathlib import Path
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.ndimage import gaussian_filter1d
from scipy.stats import pearsonr


# ── constants ─────────────────────────────────────────────────────────────────

REGION_COLS = ["chrom", "centered_start", "centered_end"]
N_RUNS = 10
CENTER_BIN       = 320
BIN_SIZE         = 2048
CENTER_BIN_START = CENTER_BIN * BIN_SIZE       # 655360
CENTER_BIN_END   = CENTER_BIN_START + BIN_SIZE  # 657408

IDX_TO_BASE = {0: "A", 1: "C", 2: "G", 3: "T"}


# ── 1. Shared helpers ─────────────────────────────────────────────────────────
# Low-level parsing and filtering utilities used across multiple sections.

def _parse_positions(series):
    """Ensure the 'positions' column contains lists, not strings."""
    return series.apply(
        lambda x: ast.literal_eval(x) if isinstance(x, str) else x
    )


def _parse_orientations(series):
    """
    Ensure the 'orientation' column contains lists, not strings.

    Handles three storage formats:
      - already a list  → returned as-is
      - stringified list, e.g. "['+', '-']"  → ast.literal_eval
      - bare symbol string, e.g. "+−" or "--"  → list of characters
    """
    def _parse(x):
        if isinstance(x, list):
            return x
        if isinstance(x, str):
            if x.startswith("["):
                return ast.literal_eval(x)
            return list(x)
        return list(x)
    return series.apply(_parse)


def _filter_region(df, region):
    """
    Return rows matching a single region tuple (chrom, centered_start, centered_end).
    Raises ValueError if no rows are found.
    """
    chrom, start, end = region
    out = df[
        (df["chrom"] == chrom) &
        (df["centered_start"] == start) &
        (df["centered_end"] == end)
    ].copy()
    if out.empty:
        raise ValueError(f"No rows found for region {chrom}:{start}-{end}")
    return out


def decode_ohe(ohe_array, rev_comp=False):
    """
    Decode a (4, L) one-hot array to a DNA string.

    Parameters
    ----------
    ohe_array : np.ndarray
        Shape (4, L), channel order A/C/G/T.
    rev_comp : bool
        If True, return the reverse complement.

    Returns
    -------
    str
    """
    bases = np.array(["A", "C", "G", "T"])
    seq = "".join(bases[np.argmax(ohe_array, axis=0)])
    if rev_comp:
        comp = str.maketrans("ACGT", "TGCA")
        seq = seq.translate(comp)[::-1]
    return seq


# ── 2. PWM scoring ────────────────────────────────────────────────────────────
# Score DNA sequences (string or one-hot) against a position-weight matrix
# using a log2 likelihood ratio relative to background nucleotide frequencies.

def seq_score_ohe(
    ohe: np.ndarray,
    pwm: np.ndarray,
    bg: dict | None = None,
    pseudocount: float = 1e-9,
) -> float:
    """
    Log-likelihood score of a one-hot encoded sequence against a PWM.

    Parameters
    ----------
    ohe : np.ndarray
        Shape (4, L), channel order A/C/G/T.
    pwm : np.ndarray
        Shape (L, 4), column order A/C/G/T.
    bg : dict, optional
        Background nucleotide probabilities. Defaults to uniform 0.25.
    pseudocount : float
        Added to PWM probabilities before taking log.

    Returns
    -------
    float
        Scalar log2-likelihood ratio score.
    """
    if bg is None:
        bg = {"A": 0.25, "C": 0.25, "G": 0.25, "T": 0.25}
    if hasattr(ohe, "numpy"):
        ohe = ohe.numpy()

    bg_arr = np.array([bg["A"], bg["C"], bg["G"], bg["T"]], dtype=np.float32)
    pwm_scores = np.sum(ohe.T * np.log2(pwm + pseudocount))
    bg_scores  = np.sum(ohe.T * np.log2(bg_arr))
    return float(pwm_scores - bg_scores)


# ── 3. CTCF site analysis ─────────────────────────────────────────────────────
# Collect CTCF sites placed during optimization, count how reproducibly each
# site appears across independent runs, and score the underlying pre-optimization
# sequence against the CTCF PWM.

def collect_ctcf_sites(df):
    """
    Explode per-run CTCF site lists into one row per (region, run, site).

    Parses 'positions' and 'orientation' columns from string storage format
    if necessary.

    Parameters
    ----------
    df : pd.DataFrame
        Output of load_indep_runs_results().

    Returns
    -------
    pd.DataFrame
        Columns: chrom, centered_start, centered_end, rel_start, rel_end,
        orientation, run.
    """
    df = df.copy()
    df["positions"]   = _parse_positions(df["positions"])
    df["orientation"] = _parse_orientations(df["orientation"])

    records = []
    for _, row in df.iterrows():
        for (rel_start, rel_end), ori in zip(row["positions"], row["orientation"]):
            records.append({
                "chrom":          row["chrom"],
                "centered_start": row["centered_start"],
                "centered_end":   row["centered_end"],
                "rel_start":      rel_start,
                "rel_end":        rel_end,
                "orientation":    ori,
                "run":            row["run"],
            })
    return pd.DataFrame(records)


def count_ctcf_reproducibility(ctcf_df, n_runs=N_RUNS):
    """
    Count how many runs each unique CTCF site appears in and compute its
    fraction out of the total number of runs.

    Parameters
    ----------
    ctcf_df : pd.DataFrame
        Output of collect_ctcf_sites().
    n_runs : int
        Total number of independent runs (denominator for fraction).

    Returns
    -------
    pd.DataFrame
        One row per unique site with columns: site columns, runs_present,
        fraction_across_runs.
    """
    site_cols = ["chrom", "centered_start", "centered_end", "rel_start", "rel_end", "orientation"]
    return (
        ctcf_df
        .groupby(site_cols)
        .agg(runs_present=("run", "nunique"))
        .reset_index()
        .assign(fraction_across_runs=lambda d: d["runs_present"] / n_runs)
    )


def _load_center_bin(seq_dir, chrom, centered_start, centered_end, fold):
    """
    Load the central 2048 bp bin from a pre-optimization one-hot .pt file.

    Returns a (4, 2048) numpy array (channel order A/C/G/T).
    """
    import torch
    path = Path(seq_dir) / f"fold{fold}" / f"{chrom}_{centered_start}_{centered_end}_X.pt"
    X = torch.load(path, map_location="cpu")  # (1, 4, 1310720)
    return X[0, :, CENTER_BIN_START:CENTER_BIN_END].detach().cpu().numpy()  # (4, 2048)


def score_ctcf_sites(ctcf_df, df_runs, ctcf_counts, seq_dir, pwm, bg):
    """
    Score each unique CTCF site against a PWM using the pre-optimization sequence.

    For each (site, fold) combination the subsequence is fetched from the saved
    one-hot encoded .pt file and scored.  Scores are averaged across folds and
    merged back into ctcf_counts.  Sites that fall partially outside the central
    bin (shape != (4, 19)) are silently skipped.

    Parameters
    ----------
    ctcf_df : pd.DataFrame
        Output of collect_ctcf_sites().
    df_runs : pd.DataFrame
        Output of load_indep_runs_results(), used to map runs to folds.
    ctcf_counts : pd.DataFrame
        Output of count_ctcf_reproducibility().
    seq_dir : str or Path
        Parent directory containing fold0/, fold1/, ... subdirectories.
    pwm : np.ndarray
        Shape (L, 4).
    bg : dict
        Background nucleotide probabilities.

    Returns
    -------
    pd.DataFrame
        ctcf_counts with an added 'score' column (mean across folds).
    """
    site_cols = ["chrom", "centered_start", "centered_end", "rel_start", "rel_end", "orientation"]

    fold_info = (
        ctcf_df[site_cols + ["run"]]
        .merge(
            df_runs[["run", "fold"] + REGION_COLS].drop_duplicates(),
            on=["run"] + REGION_COLS,
            how="left",
        )
        [site_cols + ["fold"]]
        .drop_duplicates()
    )

    cache = {}  # (chrom, centered_start, centered_end, fold) → (4, 2048) numpy array

    records = []
    for _, row in tqdm(fold_info.iterrows(), total=len(fold_info)):
        key = (row["chrom"], row["centered_start"], row["centered_end"], row["fold"])
        if key not in cache:
            cache[key] = _load_center_bin(seq_dir, *key)
        bin_ohe = cache[key]  # (4, 2048)

        site_ohe = bin_ohe[:, row["rel_start"]:row["rel_end"]]  # (4, 19)
        if row["orientation"] == "-":
            site_ohe = site_ohe[::-1, ::-1]  # reverse complement in one-hot space

        if site_ohe.shape == (4, 19):
            records.append({**{c: row[c] for c in site_cols}, "score": seq_score_ohe(site_ohe, pwm, bg=bg)})

    scores_df = (
        pd.DataFrame(records)
        .groupby(site_cols)["score"]
        .mean()
        .reset_index()
    )
    return ctcf_counts.merge(scores_df, on=site_cols, how="left")


# ── 4. Jaccard index ──────────────────────────────────────────────────────────
# Measure reproducibility of CTCF placement across runs by computing the
# pairwise Jaccard index between runs' sets of (position, orientation) tuples.

def _motif_set(row):
    """Represent a run's CTCF sites as a frozenset of (start, end, strand) tuples."""
    return frozenset((s, e, o) for (s, e), o in zip(row["positions"], row["orientation"]))


def _jaccard(set1, set2):
    """Jaccard index between two sets; returns NaN if both are empty."""
    union = len(set1 | set2)
    return len(set1 & set2) / union if union > 0 else float("nan")


def compute_jaccard(df):
    """
    Compute pairwise Jaccard indices between runs for every region.

    Parameters
    ----------
    df : pd.DataFrame
        Output of load_indep_runs_results(), with 'positions' and 'orientation'.

    Returns
    -------
    jaccard_df : pd.DataFrame
        Columns: chrom, centered_start, centered_end, run1, run2, jaccard.
    avg_jaccard_df : pd.DataFrame
        Per-region mean Jaccard index across all run pairs.
    """
    df = df.copy()
    df["positions"] = _parse_positions(df["positions"])
    df["motif_set"] = df.apply(_motif_set, axis=1)

    records = []
    for region, region_df in df.groupby(REGION_COLS):
        run_to_sites = dict(zip(region_df["run"], region_df["motif_set"]))
        for run1, run2 in itertools.combinations(sorted(run_to_sites), 2):
            records.append({
                "chrom": region[0],
                "centered_start": region[1],
                "centered_end": region[2],
                "run1": run1,
                "run2": run2,
                "jaccard": _jaccard(run_to_sites[run1], run_to_sites[run2]),
            })

    jaccard_df = pd.DataFrame(records)
    avg_jaccard_df = (
        jaccard_df
        .groupby(REGION_COLS)["jaccard"]
        .mean()
        .reset_index()
        .rename(columns={"jaccard": "avg_jaccard"})
    )
    return jaccard_df, avg_jaccard_df


def get_jaccard_matrix(region, jaccard_df):
    """
    Build a symmetric run × run Jaccard matrix for a single region.

    Parameters
    ----------
    region : tuple
        (chrom, centered_start, centered_end).
    jaccard_df : pd.DataFrame
        Output of compute_jaccard().

    Returns
    -------
    pd.DataFrame
        Square symmetric matrix with 1s on the diagonal.
    """
    sub = _filter_region(jaccard_df, region)
    runs = sorted(set(sub["run1"]) | set(sub["run2"]))
    mat = pd.DataFrame(np.eye(len(runs)), index=runs, columns=runs)
    for _, row in sub.iterrows():
        mat.loc[row["run1"], row["run2"]] = row["jaccard"]
        mat.loc[row["run2"], row["run1"]] = row["jaccard"]
    return mat


# ── 5. Plotting ───────────────────────────────────────────────────────────────
# All visualisation functions follow the same conventions:
#   - Accept an optional `ax` to embed in a larger figure.
#   - Accept an optional `savepath` / `savedir` to save to disk.
#   - Return (fig, ax) or (fig, axes).

def plot_ctcf_orientations(
    df,
    region,
    *,
    region_length=2048,
    flank=60,
    bin_size=10,
    figsize=(12, 4),
    savepath=None,
    ax=None,
):
    """
    Plot CTCF motif positions and orientations across independent runs for a
    single genomic region.  Runs are ordered by hierarchical clustering on
    their smoothed position tracks so similar patterns appear adjacent.

    Parameters
    ----------
    df : pd.DataFrame
        Output of load_indep_runs_results().
    region : tuple
        (chrom, centered_start, centered_end).
    region_length : int
        Length of the designed sequence in bp (default 2048).
    flank : int
        Flanking bp to show on each side (default 60).
    bin_size : int
        Resolution for clustering tracks in bp (default 10).
    figsize : tuple
        Passed to plt.subplots; ignored when ax is provided.
    savepath : str or Path, optional
        Save path; format inferred from suffix.
    ax : matplotlib.Axes, optional

    Returns
    -------
    fig, ax
    """
    region_df = _filter_region(df, region)
    region_df["positions"] = _parse_positions(region_df["positions"])
    chrom, start, end = region

    bins = np.arange(-flank, region_length + flank + 1, bin_size)
    run_order = _cluster_runs(region_df, bins, flank, region_length)
    y_pos = {run: idx for idx, run in enumerate(run_order)}
    n_runs = len(run_order)

    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    for run in run_order:
        ax.plot(
            [-flank, region_length + flank], [y_pos[run]] * 2,
            color="lightgrey", linewidth=0.8, linestyle="--", alpha=0.8, zorder=1,
        )

    for _, row in region_df.iterrows():
        y = y_pos[row["run"]]
        for (s, e_), ori in zip(row["positions"], row["orientation"]):
            color = "blue" if ori == "+" else "red"
            dx = (e_ - s) * (1 if ori == "+" else -1)
            ax.add_patch(
                FancyArrow(
                    s, y, dx, 0,
                    width=0.2,
                    head_width=0.3,
                    head_length=min(20, abs(dx) * 0.6),
                    length_includes_head=True,
                    color=color,
                    zorder=2,
                )
            )

    ax.set_xlim(0, region_length)
    ax.set_ylim(-1, n_runs)
    ax.set_yticks(list(y_pos.values()))
    ax.set_yticklabels([f"Run {r}" for r in run_order])
    ax.set_xlabel("Relative position (bp)")
    ax.set_title(f"CTCF motif orientations – {chrom}:{start}-{end}")
    ax.grid(axis="x", linestyle="--", alpha=0.4)
    ax.legend(
        handles=[
            Patch(color="blue", label="Orientation +"),
            Patch(color="red",  label="Orientation −"),
        ],
        loc="lower left",
    )

    if own_fig:
        fig.tight_layout()

    if savepath is not None:
        savepath = Path(savepath)
        fmt = savepath.suffix.lstrip(".") or "svg"
        fig.savefig(savepath, format=fmt, bbox_inches="tight")

    return fig, ax


def plot_jaccard_matrix(
    region,
    jaccard_df,
    *,
    figsize=(5, 4),
    ax=None,
    savedir=None,
):
    """
    Plot a run × run Jaccard index heatmap for a single region.

    Parameters
    ----------
    region : tuple
        (chrom, centered_start, centered_end).
    jaccard_df : pd.DataFrame
        Output of compute_jaccard().
    figsize : tuple
        Passed to plt.subplots; ignored when ax is provided.
    ax : matplotlib.Axes, optional
    savedir : str or Path, optional
        Directory in which to save an SVG named after the region.

    Returns
    -------
    fig, ax
    """
    chrom, start, end = region
    mat = get_jaccard_matrix(region, jaccard_df)

    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    sns.heatmap(
        mat, annot=True, fmt=".2f", cmap="viridis", vmin=0, vmax=1, square=True,
        xticklabels=mat.columns, yticklabels=mat.index,
        cbar_kws={"label": "Jaccard index"}, ax=ax,
    )
    ax.set_title(f"{chrom}:{start}-{end}")
    ax.set_xlabel("Run")
    ax.set_ylabel("Run")

    if own_fig:
        fig.tight_layout()

    if savedir is not None:
        path = Path(savedir) / f"jaccard_matrix_{chrom}_{start}_{end}.svg"
        fig.savefig(path, format="svg", bbox_inches="tight")

    return fig, ax


def plot_jaccard_matrices(regions, jaccard_df, *, savedir=None):
    """Plot Jaccard heatmaps for a list of region tuples, one figure each."""
    for region in regions:
        plot_jaccard_matrix(region, jaccard_df, savedir=savedir)
        plt.show()


def plot_score_vs_fraction(ctcf_counts, *, ax=None, figsize=(8, 6), savepath=None):
    """
    Boxplot of pre-optimization PWM score vs. fraction of runs a CTCF site
    appears in.  Higher scores for more reproducible sites would suggest that
    sequence context influences where CTCFs are inserted.

    Parameters
    ----------
    ctcf_counts : pd.DataFrame
        Output of score_ctcf_sites(); must have 'score' and
        'fraction_across_runs'.
    ax : matplotlib.Axes, optional
    figsize : tuple
    savepath : str or Path, optional

    Returns
    -------
    fig, ax
    """
    df = ctcf_counts.copy()
    df["fraction_str"] = df["fraction_across_runs"].map(lambda x: f"{x:.1f}")
    fraction_order = [
        f"{x:.1f}" for x in sorted(df["fraction_across_runs"].unique(), reverse=True)
    ]

    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    sns.boxplot(data=df, x="score", y="fraction_str", order=fraction_order, ax=ax)
    ax.set_xlabel("Pre-optimization motif score")
    ax.set_ylabel("Fraction of runs")
    ax.set_title("CTCF motif score vs. reproducibility across runs")

    if own_fig:
        fig.tight_layout()

    if savepath is not None:
        savepath = Path(savepath)
        fig.savefig(savepath, format=savepath.suffix.lstrip(".") or "svg", bbox_inches="tight")

    return fig, ax


def _plot_single_pair(ax, x, y, run1, run2, value_col):
    """Draw one scatter panel for a pair of runs onto an existing Axes."""
    r, _ = pearsonr(x, y)
    lim = [min(x.min(), y.min()), max(x.max(), y.max())]
    ax.scatter(x, y, alpha=0.5, s=10, edgecolors="none")
    ax.plot(lim, lim, "r--", linewidth=0.8)
    ax.set_xlabel(f"Run {run1}")
    ax.set_ylabel(f"Run {run2}")
    ax.set_title(f"r = {r:.3f}", fontsize=9)
    ax.grid(alpha=0.3, linestyle="--")


def plot_pairwise_runs(
    df,
    value_col="insul_score_diff",
    *,
    success_only=True,
    runs_subset=None,
    figsize_per_panel=3,
    ax=None,
    savepath=None,
):
    """
    Scatter plot of value_col for all pairs of runs, one panel per pair.

    For a single pair with ax provided, draws directly into that Axes.
    With 10 runs the full grid covers all 45 pairs (4 columns × 12 rows).

    Parameters
    ----------
    df : pd.DataFrame
        Output of load_indep_runs_results().
    value_col : str
        Column to compare across runs (default 'insul_score_diff').
    success_only : bool
        If True, restrict to rows where optimization_success == True.
    runs_subset : list of int, optional
        Restrict comparison to these runs. Pass [r1, r2] for a single panel.
    figsize_per_panel : int
        Size of each square panel in inches; ignored when ax is provided.
    ax : matplotlib.Axes, optional
        Used only when runs_subset defines exactly one pair.
    savepath : str or Path, optional

    Returns
    -------
    fig, axes
    """
    if success_only:
        df = df[df["optimization_success"]]

    runs = sorted(runs_subset if runs_subset is not None else df["run"].unique())
    pairs = list(itertools.combinations(runs, 2))
    n = len(pairs)

    # single-pair shortcut
    if n == 1 and ax is not None:
        run1, run2 = pairs[0]
        pivot = (
            df[df["run"].isin([run1, run2])]
            .pivot_table(index=REGION_COLS, columns="run", values=value_col)
            .dropna()
        )
        _plot_single_pair(ax, pivot[run1], pivot[run2], run1, run2, value_col)
        fig = ax.get_figure()
        fig.tight_layout()
        if savepath is not None:
            savepath = Path(savepath)
            fig.savefig(savepath, format=savepath.suffix.lstrip(".") or "svg", bbox_inches="tight")
        return fig, ax

    ncols = min(n, 4)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(figsize_per_panel * ncols, figsize_per_panel * nrows),
        squeeze=False,
    )

    for ax, (run1, run2) in zip(axes.flat, pairs):
        pivot = (
            df[df["run"].isin([run1, run2])]
            .pivot_table(index=REGION_COLS, columns="run", values=value_col)
            .dropna()
        )
        if pivot.empty or pivot.shape[1] < 2:
            ax.set_visible(False)
            continue

        x, y = pivot[run1], pivot[run2]
        _plot_single_pair(ax, x, y, run1, run2, value_col)

    for ax in axes.flat[n:]:
        ax.set_visible(False)

    fig.suptitle(f"{value_col} – pairwise run comparison", y=1.01)
    fig.tight_layout()

    if savepath is not None:
        savepath = Path(savepath)
        fig.savefig(savepath, format=savepath.suffix.lstrip(".") or "svg", bbox_inches="tight")

    return fig, axes


# ── internal clustering helper (used by plot_ctcf_orientations) ───────────────

def _cluster_runs(region_df, bins, flank, region_length):
    """Return runs ordered by hierarchical clustering on smoothed position tracks."""
    tracks, run_ids = [], []
    for run in sorted(region_df["run"].unique()):
        row = region_df[region_df["run"] == run].iloc[0]
        signal = np.zeros(len(bins))
        for (start, _end), ori in zip(row["positions"], row["orientation"]):
            start = np.clip(start, -flank, region_length + flank)
            idx = np.argmin(np.abs(bins - start))
            signal[idx] = 1 if ori == "+" else -1
        tracks.append(gaussian_filter1d(signal, sigma=1))
        run_ids.append(run)

    tracks = np.array(tracks)
    if len(tracks) > 1:
        dist = pdist(tracks, metric="cosine")
        link = linkage(dist, method="average")
        order = leaves_list(link)
    else:
        order = [0]

    return [run_ids[i] for i in order]