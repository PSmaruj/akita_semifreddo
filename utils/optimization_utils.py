import torch
import numpy as np
from utils.data_utils import upper_triangular_to_vector, fragment_indices_in_upper_triangular


def strength_tag(strength: float) -> str:
    """Convert a float boundary strength to a filesystem-safe string.
    e.g. -0.5 -> 'neg0p5', 1.0 -> 'pos1p0', 0.0 -> '0p0'
    """
    # Use :.1f to ensure at least one decimal, or :.4g if you prefer precision.
    # We'll use a helper to determine the prefix.
    prefix = "neg" if strength < 0 else ("pos" if strength > 0 else "")
    
    # format to a string with a guaranteed decimal point
    # 'abs' prevents double negatives in the string formatting
    val_str = f"{abs(strength):.1f}".replace(".", "p")
    
    return f"{prefix}{val_str}"


def make_boundary_mask(
    strength: float,
    map_size: int = 512,
    num_diags: int = 2,
):
    """Build the boundary mask that suppresses inter-compartment contacts.

    The top-right and bottom-left quadrants of the contact map are set to
    `strength` (typically negative); all other entries are 0.

    Returns
    -------
    indices : (N,) LongTensor   — positions in the flattened upper-tri vector
    vector  : (M,) FloatTensor  — full upper-tri vector with mask values filled in
    """
    half = map_size // 2

    matrix = np.zeros((map_size, map_size))
    matrix[:half, half:] = strength
    matrix[half:, :half] = strength

    fragment_bool = np.zeros((map_size, map_size), dtype=bool)
    fragment_bool[:half, half:] = True
    fragment_bool[half:, :half] = True

    vector  = upper_triangular_to_vector(matrix, num_diags)
    indices = fragment_indices_in_upper_triangular(
        matrix_size=map_size, fragment_mask=fragment_bool
    )

    return torch.tensor(indices), torch.tensor(vector).float()


def store_tower_output(X_tensor: torch.Tensor, model, path: str) -> None:
    """Run the convolutional tower on X and save activations to disk."""
    with torch.no_grad():
        x = model.conv_block_1(X_tensor)
        x = model.conv_tower(x)
    torch.save(x.cpu(), path)
    torch.cuda.empty_cache()
    

def make_target(
    model,
    X_tensor: torch.Tensor,
    feature_mask: torch.Tensor,
    feature_vector: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """Run the full model on X, then overwrite the boundary positions with mask values."""
    with torch.no_grad():
        y = model(X_tensor.to(device))

    y_bar = y.clone()
    y_bar[0, 0, feature_mask] = feature_vector[feature_mask].to(device)
    return y_bar.cpu()