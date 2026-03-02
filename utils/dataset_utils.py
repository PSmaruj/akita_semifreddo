"""
utils/dataset_utils.py

PyTorch Dataset classes for loading genomic sequences and contact maps.
"""

import os
import torch
from torch.utils.data import Dataset


class OriginalDataset(Dataset):
    """Load one-hot encoded initial (genomic) sequences from pre-saved .pt files.

    Each file is named  <chrom>_<start>_<end>_X.pt  and contains a
    (1, 4, L) tensor.  The batch dimension is squeezed out so the
    DataLoader collates into (B, 4, L).
    """

    def __init__(self, coord_df, seq_dir: str):
        self.coords  = coord_df
        self.seq_dir = seq_dir

    def __len__(self):
        return len(self.coords)

    def __getitem__(self, idx):
        row = self.coords.iloc[idx]
        path = os.path.join(
            self.seq_dir,
            f"{row['chrom']}_{row['start']}_{row['end']}_X.pt",
        )
        return torch.load(path, weights_only=True).squeeze(0)  # (4, L)


class GenomicDataset(Dataset):
    """Load Ledidi-optimised sequences from pre-saved .pt files.

    Each file is named  <chrom>_<start>_<end>_seq.pt  and may contain
    either a (1, 4, L) or (4, L) tensor.  Returned as-is so callers can
    squeeze as needed (the optimisation loop saves with a batch dim).
    """

    def __init__(self, coord_df, seq_dir: str):
        self.coords  = coord_df
        self.seq_dir = seq_dir

    def __len__(self):
        return len(self.coords)

    def __getitem__(self, idx):
        row = self.coords.iloc[idx]
        path = os.path.join(
            self.seq_dir,
            f"{row['chrom']}_{row['start']}_{row['end']}_seq.pt",
        )
        return torch.load(path, weights_only=True)  # (1, 4, L) or (4, L)


class TriuMatrixDataset(Dataset):
    """Load pre-computed upper-triangular Hi-C target vectors.

    Each file is named  <target_chrom>_<target_start>_<target_end>_target.pt
    and contains a flat upper-tri vector (the format Akita outputs).
    Coordinates are read from the  target_chrom / target_start / target_end
    columns of the DataFrame.
    """

    def __init__(self, coord_df, map_dir: str):
        self.coords  = coord_df
        self.map_dir = map_dir

    def __len__(self):
        return len(self.coords)

    def __getitem__(self, idx):
        row = self.coords.iloc[idx]
        path = os.path.join(
            self.map_dir,
            f"{row['target_chrom']}_{row['target_start']}_{row['target_end']}_target.pt",
        )
        return torch.load(path, map_location="cpu")  # (1, N_triu) or (N_triu,)