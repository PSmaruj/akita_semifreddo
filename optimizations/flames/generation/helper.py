import numpy as np
import pandas as pd
import torch

MAP_SIZE    = 512
HALF        = MAP_SIZE // 2
NUM_DIAGS   = 2   # diagonals excluded from the upper-tri representation
FLAME_WIDTH = 3

def create_flame_mask(
    shape: tuple[int, int] = (MAP_SIZE, MAP_SIZE),
    center: tuple[int, int] | None = None,
    flame_width: int = FLAME_WIDTH,
    value: float = 0.5,
) -> np.ndarray:
    """
    Creates a flame/stripe-shaped mask: vertical stripe (top half of map) and
    horizontal stripe (left half of map) forming an 'L' shape.

    Parameters
    ----------
    shape       : (rows, cols) of the mask.
    center      : (row, col) origin of the flame; defaults to map centre.
    flame_width : Total width of each stripe in bins.
    value       : Fill value within the stripe.

    Returns
    -------
    mask : 2-D float array of the requested shape.
    """
    H, W = shape
    half_r, half_c = center if center is not None else (H // 2, W // 2)
    half_w = flame_width // 2

    mask = np.zeros((H, W), dtype=float)
    mask[:half_r, half_c - half_w : half_c + half_w] = value   # vertical stripe
    mask[half_r - half_w : half_r + half_w, :half_c] = value   # horizontal stripe
    return mask


def make_flame_mask_indices(value: float) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Build the flat upper-tri index tensor and full upper-tri value vector for
    the flame mask, matching the format expected by make_target().

    Returns
    -------
    f_indices : LongTensor of shape (K,) — flat positions within the upper-tri
                vector where the flame mask is nonzero.
    f_vector  : FloatTensor of shape (N_upper_tri,) — full upper-tri vector with
                flame values at masked positions and 0 elsewhere.
    """
    full_mask = create_flame_mask(value=value)

    rows, cols = np.triu_indices(MAP_SIZE, k=NUM_DIAGS)
    upper_tri_values = full_mask[rows, cols]                          # (N_upper_tri,)

    f_indices = torch.tensor(np.nonzero(upper_tri_values)[0], dtype=torch.long)
    f_vector  = torch.tensor(upper_tri_values, dtype=torch.float32)  # full length
    return f_indices, f_vector