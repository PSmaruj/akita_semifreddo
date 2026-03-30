"""
utils/dataset_utils.py

PyTorch Dataset classes for loading genomic sequences and contact maps.
"""

import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from pyfaidx import Fasta
import seqpro as sp

from .data_utils import one_hot_encode_sequence


class SequenceDataset(Dataset):
    """Load one-hot encoded sequences from pre-saved .pt files.

    Each file is named  <chrom>_<start>_<end>_<suffix>.pt  and contains
    a (1, 4, L) or (4, L) tensor.  The batch dimension is squeezed out
    so the DataLoader collates into (B, 4, L).

    Parameters
    ----------
    coord_df : pd.DataFrame
        Table with columns 'chrom', 'start', 'end'.
    seq_dir : str
        Directory containing the .pt files.
    chrom_col : str
        Column name for chromosome, e.g. 'chrom' or 'target_chrom'.
    start_col : str
        Column name for start coordinate, e.g. 'centered_start' or 'target_start'.
    end_col : str
        Column name for end coordinate, e.g. 'centered_end' or 'target_end'.
    suffix : str
        File suffix, e.g. 'target'.
    suffix : str
        File suffix, e.g. 'X' for genomic sequences or 'gen_seq' for
        Ledidi-optimised central bins.
    """

    def __init__(self, coord_df: pd.DataFrame, 
                 seq_dir: str, 
                 chrom_col: str = "chrom",
                 start_col: str = "start",
                 end_col: str   = "end",
                 suffix: str   = "X"):
        
        self.coords  = coord_df
        self.seq_dir = seq_dir
        self.chrom_col = chrom_col
        self.start_col = start_col
        self.end_col   = end_col
        self.suffix  = suffix

    def __len__(self) -> int:
        return len(self.coords)

    def __getitem__(self, idx: int) -> torch.Tensor:
        row  = self.coords.iloc[idx]
        path = os.path.join(
            self.seq_dir,
            f"{row[self.chrom_col]}_{row[self.start_col]}_{row[self.end_col]}_{self.suffix}.pt",
        )
        return torch.load(path, weights_only=True).squeeze(0)  # (4, L)


class TriuMatrixDataset(Dataset):
    """Load pre-computed upper-triangular Hi-C target vectors.

    Each file is named  <chrom>_<start>_<end>_target.pt  and contains
    a flat upper-tri vector (the format Akita outputs).

    Parameters
    ----------
    coord_df : pd.DataFrame
        Table with coordinate columns.
    map_dir : str
        Directory containing the .pt files.
    chrom_col : str
        Column name for chromosome, e.g. 'chrom' or 'target_chrom'.
    start_col : str
        Column name for start coordinate, e.g. 'centered_start' or 'target_start'.
    end_col : str
        Column name for end coordinate, e.g. 'centered_end' or 'target_end'.
    suffix : str
        File suffix, e.g. 'target'.
    """

    def __init__(
        self,
        coord_df: pd.DataFrame,
        map_dir: str,
        chrom_col: str = "chrom",
        start_col: str = "start",
        end_col: str   = "end",
        suffix: str    = "target",
    ):
        self.coords    = coord_df
        self.map_dir   = map_dir
        self.chrom_col = chrom_col
        self.start_col = start_col
        self.end_col   = end_col
        self.suffix    = suffix

    def __len__(self) -> int:
        return len(self.coords)

    def __getitem__(self, idx: int) -> torch.Tensor:
        row  = self.coords.iloc[idx]
        path = os.path.join(
            self.map_dir,
            f"{row[self.chrom_col]}_{row[self.start_col]}_{row[self.end_col]}_{self.suffix}.pt",
        )
        return torch.load(path, map_location="cpu")  # (1, N_triu) or (N_triu,)


class HiCDataset(Dataset):
    def __init__(self, data_files):
        self.data = []
        for file in data_files:
            print(f"  Loading: {file}")
            file_data = torch.load(file, weights_only=True)
            for ohe_sequence, hic_vector in file_data:
                ohe_sequence = ohe_sequence.squeeze(0)
                assert ohe_sequence.shape[0] == 4
                assert ohe_sequence.ndim == 2
                self.data.append((ohe_sequence, hic_vector))
        print(f"  Total sequences loaded: {len(self.data)}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class FeatureDataset(Dataset):
    """
    Returns one-hot encoded sequences centred on each dot's / flame's prediction window.

    Args:
        coord_df:     DataFrame with columns [chrom, window_start, window_end].
        genome_fasta: pyfaidx.Fasta object for the reference genome.
    """

    def __init__(self, coord_df: pd.DataFrame, genome_fasta: Fasta,
                 target_len) -> None:
        self.coords       = coord_df
        self.genome       = genome_fasta
        self.target_len = target_len
        
    def __len__(self) -> int:
        return len(self.coords)

    def __getitem__(self, idx: int) -> torch.Tensor:
        row   = self.coords.iloc[idx]
        chrom = row["chrom"]
        seq   = self.genome[chrom][row["window_start"]:row["window_end"]].seq.upper()

        # Guard against off-chromosome windows: truncate or N-pad
        if len(seq) != self.target_len:
            seq = seq[:self.target_len].ljust(self.target_len, "N")

        one_hot = one_hot_encode_sequence(seq)   # (1, 4, target_len) or (4, target_len)
        one_hot = np.squeeze(one_hot)            # → (4, target_len)
        return torch.from_numpy(one_hot.copy())


class CentralInsertionDataset(Dataset):
    """Reconstruct full sequences by inserting the optimised central bin."""

    def __init__(self, coord_df: pd.DataFrame, seq_path: str, slice_path: str,
                 edit_start: int, edit_end: int):
        self.coords     = coord_df
        self.seq_path   = seq_path
        self.slice_path = slice_path
        self.edit_start = edit_start
        self.edit_end = edit_end
        
    def __len__(self) -> int:
        return len(self.coords)

    def __getitem__(self, idx: int) -> torch.Tensor:
        row   = self.coords.iloc[idx]
        chrom = row["chrom"]
        start = int(row["centered_start"])
        end   = int(row["centered_end"])
        stem  = f"{chrom}_{start}_{end}"

        X     = torch.load(f"{self.seq_path}{stem}_X.pt",        weights_only=True)
        slice_ = torch.load(f"{self.slice_path}{stem}_gen_seq.pt", weights_only=True)

        edited = X.clone()
        edited[:, :, self.edit_start:self.edit_end] = slice_
        return edited.squeeze(0)


class DoubleInsertionDataset(Dataset):
    """Reconstruct full sequences by inserting two optimised anchor bins.

    The saved _gen_seq.pt contains a (1, 4, 2*BIN_SIZE) tensor with the
    lo anchor in [:, :, :BIN_SIZE] and the hi anchor in [:, :, BIN_SIZE:].
    """

    def __init__(
        self,
        coord_df: pd.DataFrame,
        seq_path: str,
        slice_path: str,
        bp_lo_start: int,
        bp_lo_end: int,
        bp_hi_start: int,
        bp_hi_end: int,
        bin_size: int = 2048,
    ):
        self.coords      = coord_df
        self.seq_path    = seq_path
        self.slice_path  = slice_path
        self.bp_lo_start = bp_lo_start
        self.bp_lo_end   = bp_lo_end
        self.bp_hi_start = bp_hi_start
        self.bp_hi_end   = bp_hi_end
        self.bin_size    = bin_size

    def __len__(self) -> int:
        return len(self.coords)

    def __getitem__(self, idx: int) -> torch.Tensor:
        row   = self.coords.iloc[idx]
        chrom = row["chrom"]
        start = int(row["centered_start"])
        end   = int(row["centered_end"])
        stem  = f"{chrom}_{start}_{end}"

        X      = torch.load(f"{self.seq_path}{stem}_X.pt",        weights_only=True)
        slice_ = torch.load(f"{self.slice_path}{stem}_gen_seq.pt", weights_only=True)

        edited = X.clone()
        edited[:, :, self.bp_lo_start:self.bp_lo_end] = slice_[:, :, :self.bin_size]
        edited[:, :, self.bp_hi_start:self.bp_hi_end] = slice_[:, :, self.bin_size:]
        return edited.squeeze(0)   # (4, L)
