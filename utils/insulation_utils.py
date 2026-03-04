import numpy as np
import pandas as pd
from scipy.stats import pearsonr

# ---------------------------------------------------------------------------
# Insulation score
# ---------------------------------------------------------------------------

def calculate_insulation_profile(contact_map, window=16):
    """
    Compute insulation score using a diamond window along the diagonal.

    Parameters
    ----------
    contact_map : np.ndarray  (N x N)
    window : int
        Half-width of the diamond.

    Returns
    -------
    np.ndarray of length N - 2*window
    """
    map_size = contact_map.shape[0]
    scores = []
    for i in range(window, map_size - window):
        diamond = contact_map[i - window : i, i : i + window]
        scores.append(np.nanmean(diamond))
    return np.array(scores)


def insulation_full(contact_map, window):
    """Return insulation padded with NaNs to match map size."""
    map_size = contact_map.shape[0]
    raw = calculate_insulation_profile(contact_map, window)
    full = np.full(map_size, np.nan)
    full[window : map_size - window] = raw
    return full


# ---------------------------------------------------------------------------
# PearsonR
# ---------------------------------------------------------------------------

def masked_pearson(target, pred):
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
    """
    Detect the longest contiguous flat region in an insulation profile
    using a sliding-window standard deviation filter.

    Returns
    -------
    (flat_start, flat_end) in full-profile coordinates, or (None, None).
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
    """
    Shift the genomic window so the flat region falls at the map center.
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
    """
    Keep at most one region per `min_spacing` bp window along each chromosome.
    When two regions are within the window, one is kept randomly.
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