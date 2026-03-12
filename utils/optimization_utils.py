import torch
import numpy as np


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


def build_stem(chrom: str, start: int, end: int) -> str:
    return f"{chrom}_{start}_{end}"


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


def last_accepted_step(history: dict) -> int:
    """Return the index of the last iteration that introduced a new edited position.

    Each entry in history["edits"] is a tuple of three tensors:
        (batch_indices, nucleotide_indices, position_indices)
    We track the cumulative set of positions seen across steps 0..i-1 and
    return the last i where history["edits"][i][2] contains a position not
    previously seen.
    """
    edits = history["edits"]
    seen  = set()
    last  = 0
    for i, edit in enumerate(edits):
        positions = set(edit[2].cpu().tolist())
        if positions - seen:
            last = i
            seen |= positions
    return last


def count_edits(original_X: torch.Tensor, generated_full: torch.Tensor) -> int:
    """Number of nucleotide positions that differ between original and generated."""
    return int(
        (torch.argmax(generated_full, dim=1) != torch.argmax(original_X, dim=1))
        .sum()
        .item()
    )

