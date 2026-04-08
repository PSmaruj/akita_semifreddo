import numpy as np
import pandas as pd
import torch

MAP_SIZE  = 512
HALF      = MAP_SIZE // 2
NUM_DIAGS = 2   # diagonals excluded from the upper-tri representation

def place_pileup_at_center(pileup, shape=(512, 512), center=(256, 281)):
    """
    Embed a 15x15 pileup matrix into a larger mask, centered at a specific
    location. Ensures symmetry along the main diagonal.

    Parameters
    ----------
    pileup : np.ndarray, shape (15, 15)
    shape  : tuple, shape of the output array
    center : (row, col) where the pileup centre should be placed

    Returns
    -------
    mask : np.ndarray, shape == shape
    """
    assert pileup.shape == (15, 15), "Pileup must be 15x15"

    H, W = shape
    r0, c0 = center
    half_size = 7  # 15x15 → half = 7

    mask = np.zeros((H, W), dtype=float)

    r_start, r_end = r0 - half_size, r0 + half_size + 1
    c_start, c_end = c0 - half_size, c0 + half_size + 1

    if not (0 <= r_start < r_end <= H) or not (0 <= c_start < c_end <= W):
        raise ValueError(
            f"Pileup does not fit inside mask shape {shape} at center {center}."
        )

    mask[r_start:r_end, c_start:c_end] = pileup

    # Mirror across the main diagonal
    for r in range(H):
        for c in range(W):
            if r < W and c < H and mask[r, c] != 0:
                mask[c, r] = mask[r, c]

    return mask


def dot_center_from_distance(distance: int, half: int = HALF):
    """
    Convert an inter-anchor distance (in bins) to a 2-D map centre.

    For distance d the two anchors sit at HALF ± d//2, placing the dot at
    (HALF - d//2, HALF + d//2) in the upper triangle.
    """
    offset = distance // 2
    return (half - offset, half + offset)


def make_dot_mask(pileup: np.ndarray, distance: int):
    """
    Build the full-map dot mask and return the flat upper-tri indices and the
    full upper-tri value vector (same convention as make_boundary_mask /
    make_flame_mask_indices).

    Returns
    -------
    indices : LongTensor of shape (K,)         — flat positions within the
              upper-tri vector where the dot mask is nonzero.
    vector  : FloatTensor of shape (N_triu,)   — full upper-tri vector with
              dot values at masked positions and 0 elsewhere.
    """
    center = dot_center_from_distance(distance)
    full_mask = place_pileup_at_center(pileup, shape=(MAP_SIZE, MAP_SIZE), center=center)

    triu_rows, triu_cols = np.triu_indices(MAP_SIZE, k=NUM_DIAGS)
    upper_tri_values = full_mask[triu_rows, triu_cols]                    # (N_triu,)

    indices = torch.tensor(np.nonzero(upper_tri_values)[0], dtype=torch.long)
    vector  = torch.tensor(upper_tri_values, dtype=torch.float32)         # full length
    return indices, vector