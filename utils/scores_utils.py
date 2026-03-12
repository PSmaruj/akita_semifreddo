import numpy as np

MAP_CENTER = 256   # centre bin of the 512-bin contact map


def insulation_score(
    maps: np.ndarray,
    row_slice: slice,
    col_slice: slice,
) -> list[float]:
    """Per-sequence mean of the specified upper-right region of contact maps."""
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
    """Compute mean dot score at multiple window sizes for a batch of maps.

    Parameters
    ----------
    maps : np.ndarray, shape (B, 512, 512)
    dot_row, dot_col : int
        Row and column of the dot centre in map coordinates.
    half_widths : list[int]
        Half-widths to measure; window size = 2*hw + 1.

    Returns
    -------
    dict mapping f"dot{2*hw+1}" → list of per-sequence mean values.
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
    """Compute mean flame score at multiple column widths for a batch of maps.

    A flame is a vertical stripe of elevated contact in the upper half of the
    map, centred on the middle column. For each half-width hw, the score is
    the mean over rows 0..map_center-1 and cols map_center-hw..map_center+hw.

    Parameters
    ----------
    maps : np.ndarray, shape (B, 512, 512)
    half_widths : list[int]
        Half-widths to measure; stripe width = 2*hw + 1.
    map_center : int
        Centre bin index (default 256).

    Returns
    -------
    dict mapping f"flame{2*hw+1}" → list of per-sequence mean values.
    """
    scores = {}
    for hw in half_widths:
        key = f"flame{2 * hw + 1}"
        scores[key] = np.nanmean(
            maps[:, :map_center, map_center - hw : map_center + hw + 1],
            axis=(1, 2),
        ).tolist()
    return scores