"""
helper.py
----------------------
Utilities for reconstructing full DNA sequences from one-hot .pt tensors,
splicing in one or more edited bins, and writing FASTA files.

Supports boundary, flame (single edited bin) and dot (two edited bins)
designs through a unified interface.
"""

import numpy as np
import torch
from pathlib import Path
from typing import Optional

# ── Constants ─────────────────────────────────────────────────────────────────

BIN_SIZE  = 2048    # bp per Akita bin
TRIM_BP   = 131072  # bases trimmed from each end to match Alpha Genome window
ALPHABET  = "ACGT"

# ── Core functions ─────────────────────────────────────────────────────────────

def load_and_splice(
    main_path: str,
    edits: list[tuple[int, str | torch.Tensor]],
    bin_size: int = BIN_SIZE,
    alphabet: str = ALPHABET,
    device: Optional[str] = None,
) -> torch.Tensor:
    """
    Load a one-hot tensor and replace one or more bins with edited slices.

    Parameters
    ----------
    main_path : str
        Path to the full-context one-hot tensor, shape [1, 4, L].
    edits : list of (bin_index, slice)
        Each tuple specifies which bin to replace. The slice can be either:
        - a file path (str) to a .pt tensor, or
        - a pre-loaded torch.Tensor (e.g. a split chunk from a combined tensor).
        Pass an empty list to return the original sequence unchanged.
    bin_size : int
        Bases per bin (default: 2048).
    alphabet : str
        Nucleotide order for the one-hot channels (default: "ACGT").
    device : str or None
        Torch device string, e.g. "cuda:0". None = CPU.

    Returns
    -------
    torch.Tensor
        Modified (or original) tensor, shape [1, 4, L].
    """
    main = torch.load(main_path, map_location=device)

    if main.dim() != 3 or main.size(1) != len(alphabet):
        raise ValueError(
            f"Expected main tensor shape [1, {len(alphabet)}, L], got {tuple(main.shape)}"
        )

    for bin_idx, slice_or_path in edits:
        start = bin_idx * bin_size
        end   = start + bin_size

        if end > main.size(2):
            raise ValueError(
                f"Bin {bin_idx} replacement range [{start}:{end}] "
                f"exceeds sequence length {main.size(2)}"
            )

        slc = (
            torch.load(slice_or_path, map_location=device)
            if isinstance(slice_or_path, str)
            else slice_or_path.to(main.device)
        )

        if slc.dim() != 3 or slc.size(1) != len(alphabet) or slc.size(2) != bin_size:
            raise ValueError(
                f"Slice tensor for bin {bin_idx} must have shape "
                f"[1, {len(alphabet)}, {bin_size}], got {tuple(slc.shape)}"
            )

        main[:, :, start:end] = slc

    return main


def trim_and_decode(
    tensor: torch.Tensor,
    trim_bp: int = TRIM_BP,
    alphabet: str = ALPHABET,
) -> str:
    """
    Trim `trim_bp` bases from each end of a one-hot tensor and decode to string.

    Parameters
    ----------
    tensor : torch.Tensor
        One-hot tensor, shape [1, 4, L].
    trim_bp : int
        Bases to remove from each side (default: 131072).
    alphabet : str
        Nucleotide order for decoding (default: "ACGT").

    Returns
    -------
    str
        Decoded nucleotide sequence.
    """
    L = tensor.size(2)
    if 2 * trim_bp >= L:
        raise ValueError(
            f"trim_bp ({trim_bp}) is too large for sequence length {L}"
        )

    trimmed = tensor[:, :, trim_bp : L - trim_bp]
    idx     = trimmed.argmax(dim=1).cpu().numpy()
    letters = np.array(list(alphabet))
    seq_arr = letters[idx].ravel()
    return "".join(seq_arr.tolist())


def save_to_fasta(seq: str, fasta_path: str, header: str = "sequence") -> None:
    """
    Write a nucleotide sequence to a FASTA file (80 bp per line).

    Parameters
    ----------
    seq : str
        Nucleotide sequence (A/C/G/T/N).
    fasta_path : str
        Destination file path. Parent directories are created if needed.
    header : str
        FASTA record header (without the leading '>').
    """
    Path(fasta_path).parent.mkdir(parents=True, exist_ok=True)
    with open(fasta_path, "w") as f:
        f.write(f">{header}\n")
        for i in range(0, len(seq), 80):
            f.write(seq[i : i + 80] + "\n")