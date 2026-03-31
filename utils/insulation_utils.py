"""
utils/insulation_utils.py

Insulation score computation and flat-region detection utilities used to
identify suitable genomic windows for AkitaSF sequence optimization.

Functions
---------
calculate_insulation_profile : compute insulation scores along a contact map diagonal
insulation_full              : insulation profile padded with NaNs to match map size
masked_pearson               : Pearson R between two arrays, ignoring NaN positions

find_longest_flat_region     : detect the longest low-variance region in an insulation profile
recenter_flat_region         : shift a genomic window so the flat region falls at map center
remove_close_regions         : deduplicate regions closer than a minimum spacing
"""

import numpy as np
import pandas as pd
from scipy.stats import pearsonr


# ---------------------------------------------------------------------------
# Insulation score
# ---------------------------------------------------------------------------


def calculate_insulation_profile(contact_map, window=16):
    """Compute the insulation score profile using a diamond window along the diagonal.

    For each bin i, the score is the mean contact frequency in the
    diamond-shaped region [i-window:i, i:i+window].

    Parameters
    ----------
    contact_map : np.ndarray
        Square contact matrix of shape (N, N).
    window : int
        Half-width of the diamond window (default 16).

    Returns
    -------
    np.ndarray
        1-D array of length N - 2*window.
    """
    map_size = contact_map.shape[0]
    scores = []
    for i in range(window, map_size - window):
        diamond = contact_map[i - window : i, i : i + window]
        scores.append(np.nanmean(diamond))
    return np.array(scores)


def insulation_full(contact_map, window):
    """Compute the insulation profile padded with NaNs to match the map size.

    Parameters
    ----------
    contact_map : np.ndarray
        Square contact matrix of shape (N, N).
    window : int
        Half-width of the diamond window, passed to calculate_insulation_profile.

    Returns
    -------
    np.ndarray
        1-D array of length N with NaN at the first and last `window` positions.
    """
    map_size = contact_map.shape[0]
    raw = calculate_insulation_profile(contact_map, window)
    full = np.full(map_size, np.nan)
    full[window : map_size - window] = raw
    return full


# ---------------------------------------------------------------------------
# PearsonR
# ---------------------------------------------------------------------------

def masked_pearson(target, pred):
    """Compute Pearson R between two arrays, ignoring NaN positions.

    Parameters
    ----------
    target : np.ndarray
        Reference array; may contain NaNs.
    pred : np.ndarray
        Predicted array; may contain NaNs.

    Returns
    -------
    float
        Pearson correlation coefficient over non-NaN positions,
        or 0.0 if fewer than 2 valid positions exist.
    """
    t = target.flatten()
    p = pred.flatten()
    mask = ~np.isnan(t) & ~np.isnan(p)
    if np.sum(mask) < 2: # Pearson requires at least 2 points
        return 0.0
    return pearsonr(t[mask], p[mask])[0]


# ---------------------------------------------------------------------------
# Flat region detection
# ---------------------------------------------------------------------------


def find_longest_flat_region(
    insul_profile,
    std_window=40,
    std_threshold=0.025,
    min_length=100,
    edge_margin=50,
):
    """Detect the longest contiguous flat region in an insulation profile.

    Uses a sliding-window standard deviation filter: a bin is considered flat
    if the std of its local neighbourhood is below std_threshold.

    Parameters
    ----------
    insul_profile : np.ndarray
        1-D insulation score profile, may contain NaNs.
    std_window : int
        Width of the sliding std window in bins (default 40).
    std_threshold : float
        Maximum local std to be considered flat (default 0.025).
    min_length : int
        Minimum flat region length in bins (default 100).
    edge_margin : int
        Minimum distance from either end of the profile (default 50).

    Returns
    -------
    flat_start, flat_end : int or None
        Start and end indices of the longest flat region in full-profile
        coordinates, or (None, None) if no qualifying region is found.
    """
    valid = ~np.isnan(insul_profile)
    y = insul_profile[valid]
    indices = np.where(valid)[0]

    flat_mask = np.zeros(len(y), dtype=bool)
    half = std_window // 2
    for i in range(half, len(y) - half):
        if np.std(y[i - half : i + half]) < std_threshold:
            flat_mask[i] = True

    max_len = 0
    best_start = best_end = None

    i = 0
    while i < len(flat_mask):
        if flat_mask[i]:
            start = i
            while i < len(flat_mask) and flat_mask[i]:
                i += 1
            end = i
            region_start = indices[start]
            region_end = indices[end - 1]
            region_len = region_end - region_start
            if (
                region_len >= min_length
                and region_start >= edge_margin
                and region_end <= len(insul_profile) - edge_margin
                and region_len > max_len
            ):
                max_len = region_len
                best_start = region_start
                best_end = region_end
        else:
            i += 1

    return best_start, best_end


# ---------------------------------------------------------------------------
# Recentering
# ---------------------------------------------------------------------------


def recenter_flat_region(row, cropping=64, map_size=512, bin_size=2048):
    """Shift a genomic window so the flat region falls at the map center.

    Computes the offset between the flat region's center and the map's
    center bin, then shifts the window coordinates accordingly.

    Parameters
    ----------
    row : pd.Series
        Row with columns [start, end, flat_start, flat_end].
    cropping : int
        Bins cropped from each side by Akita (default 64).
    map_size : int
        Side length of the contact map in bins (default 512).
    bin_size : int
        Bin size in bp (default 2048).

    Returns
    -------
    pd.Series
        New series with columns: centered_start, centered_end,
        centered_flat_start, centered_flat_end.
    """
    halfpoint = (map_size // 2) + cropping

    flat_region_half = (row["flat_end"] - row["flat_start"]) // 2
    flat_region_center = row["flat_start"] + flat_region_half + cropping

    difference = flat_region_center - halfpoint

    new_start = int(row["start"] + (difference * bin_size))
    new_end = int(row["end"] + (difference * bin_size))

    centered_flat_start = int(halfpoint - flat_region_half - cropping)
    centered_flat_end = int(halfpoint + flat_region_half - cropping)

    return pd.Series(
        {
            "centered_start": new_start,
            "centered_end": new_end,
            "centered_flat_start": centered_flat_start,
            "centered_flat_end": centered_flat_end,
        }
    )


# ---------------------------------------------------------------------------
# De-duplication / spacing filter
# ---------------------------------------------------------------------------


def remove_close_regions(df, chrom_col="chrom", start_col="centered_start",
                         min_spacing=300_000, seed=None):
    """Deduplicate genomic regions closer than a minimum spacing.

    Iterates over each chromosome in sorted order, keeping at most one region
    per min_spacing bp window. When two regions fall within the window,
    one is chosen randomly (50/50).

    Parameters
    ----------
    df : pd.DataFrame
        Table with genomic region rows.
    chrom_col : str
        Column name for chromosome (default 'chrom').
    start_col : str
        Column name for start coordinate (default 'centered_start').
    min_spacing : int
        Minimum distance in bp between retained regions (default 300,000).
    seed : int or None
        Random seed for reproducibility (default None).

    Returns
    -------
    pd.DataFrame
        Filtered DataFrame with at most one region per min_spacing window,
        reset index.
    """
    if seed is not None:
        np.random.seed(seed)

    df_sorted = df.sort_values([chrom_col, start_col]).reset_index(drop=True)
    keep_indices = []
    last_chrom = None
    last_kept_start = -np.inf

    for i, row in df_sorted.iterrows():
        chrom = row[chrom_col]
        start = row[start_col]
        if chrom != last_chrom or start - last_kept_start >= min_spacing:
            keep_indices.append(i)
            last_chrom = chrom
            last_kept_start = start
        else:
            if np.random.rand() < 0.5:
                keep_indices[-1] = i
                last_kept_start = start

    return df_sorted.loc[keep_indices].reset_index(drop=True)