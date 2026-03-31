"""
utils/scores_utils.py

Contact map scoring functions for AkitaSF-designed sequences.

Each function operates on a batch of symmetric contact matrices of shape
(B, 512, 512) and returns per-sequence scores.

Functions
---------
insulation_score      : mean contact in a specified upper-right map region (boundary)
compute_dot_scores    : mean contact in a square window around a dot position
compute_flame_scores  : mean contact in a vertical stripe centred on the map column
"""

import numpy as np


MAP_CENTER = 256   # centre bin of the 512-bin contact map


def compute_insulation_scores(
    maps: np.ndarray,
    row_slice: slice,
    col_slice: slice,
) -> list[float]:
    """Compute the mean contact in a rectangular region for each map in a batch.

    Used as a boundary insulation score: the specified region is typically
    the upper-right block adjacent to the boundary position, where insulation
    manifests as reduced contact frequency.

    Parameters
    ----------
    maps : np.ndarray
        Shape (B, 512, 512); batch of symmetric contact maps.
    row_slice : slice
        Row range of the region to average.
    col_slice : slice
        Column range of the region to average.

    Returns
    -------
    list of float
        Per-sequence mean contact values, length B.
    """
    return np.nanmean(
        maps[:, row_slice, col_slice],
        axis=(1, 2)
    ).tolist()


def compute_dot_scores(
    maps: np.ndarray,
    dot_row: int,
    dot_col: int,
    half_widths: list[int],
) -> dict[str, list[float]]:
    """Compute mean contact in a square window around a dot position.

    Scores are computed at multiple window sizes, allowing comparison
    across resolutions.

    Parameters
    ----------
    maps : np.ndarray
        Shape (B, 512, 512); batch of symmetric contact maps.
    dot_row : int
        Row index of the dot centre in map coordinates.
    dot_col : int
        Column index of the dot centre in map coordinates.
    half_widths : list of int
        Half-widths to measure; window size = 2*hw + 1.

    Returns
    -------
    dict of str → list of float
        Maps f"dot{2*hw+1}" to a list of B per-sequence mean values.
    """
    scores = {}
    for hw in half_widths:
        key = f"dot{2 * hw + 1}"
        scores[key] = np.nanmean(
            maps[:, dot_row - hw : dot_row + hw + 1,
                    dot_col - hw : dot_col + hw + 1],
            axis=(1, 2),
        ).tolist()
    return scores


def compute_flame_scores(
    maps: np.ndarray,
    half_widths: list[int],
    map_center: int = MAP_CENTER,
) -> dict[str, list[float]]:
    """Compute mean contact in a vertical stripe centred on the map column.

    A flame is a vertical stripe of elevated contact in the upper half of
    the map, centred on the middle column. Scores are computed at multiple
    stripe widths.

    Parameters
    ----------
    maps : np.ndarray
        Shape (B, 512, 512); batch of symmetric contact maps.
    half_widths : list of int
        Half-widths to measure; stripe width = 2*hw + 1.
    map_center : int
        Centre bin index, used as both the column centre and the upper
        row boundary (default 256).

    Returns
    -------
    dict of str → list of float
        Maps f"flame{2*hw+1}" to a list of B per-sequence mean values.
    """
    scores = {}
    for hw in half_widths:
        key = f"flame{2 * hw + 1}"
        scores[key] = np.nanmean(
            maps[:, :map_center, map_center - hw : map_center + hw + 1],
            axis=(1, 2),
        ).tolist()
    return scores