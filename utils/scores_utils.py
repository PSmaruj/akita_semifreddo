import numpy as np

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