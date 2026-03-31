"""
utils/dataset_utils.py

PyTorch Dataset classes for loading genomic sequences and contact maps
used throughout the AkitaSF pipeline.

Datasets
--------
SequenceDataset               : load pre-saved OHE sequence tensors (.pt)
TriuMatrixDataset             : load pre-saved upper-tri Hi-C target vectors (.pt)
HiCDataset                    : load (sequence, Hi-C) pairs from batched .pt files
FeatureDataset                : OHE-encode sequences on-the-fly from a Fasta
CentralInsertionDataset       : splice optimised central bin into full sequences
DoubleInsertionDataset        : splice two optimised anchor bins into full sequences
ShuffledCentralInsertionDataset : splice dinucleotide-shuffled optimised bin (control)
"""

import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from pyfaidx import Fasta
import seqpro as sp # dinucleotide shuffle
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
    """Load (sequence, Hi-C) pairs from pre-batched .pt files.

    Each file is expected to contain a list of (ohe_sequence, hic_vector) tuples,
    where ohe_sequence has shape (1, 4, L) and is squeezed to (4, L) on load.
    All files are concatenated into a single flat list at construction time.

    Parameters
    ----------
    data_files : list[str]
        Paths to .pt files, each containing a list of (ohe_sequence, hic_vector) tuples.
    """
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
    """Load one-hot encoded sequences centred on each feature's prediction window.

    Sequences are extracted on-the-fly from a reference genome Fasta.
    Off-chromosome windows are truncated and N-padded to target_len.

    Parameters
    ----------
    coord_df : pd.DataFrame
        Table with columns [chrom, window_start, window_end].
    genome_fasta : pyfaidx.Fasta
        Reference genome.
    target_len : int
        Expected sequence length in bp; used to guard against
        off-chromosome windows.
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
    """Reconstruct full sequences by splicing in the optimised central bin.

    Loads the full original sequence (_X.pt) and the Ledidi-optimised central
    bin (_gen_seq.pt), replaces the edit region in the original, and returns
    the full edited sequence.

    Parameters
    ----------
    coord_df : pd.DataFrame
        Table with columns [chrom, centered_start, centered_end].
    seq_path : str
        Directory containing {stem}_X.pt full sequence tensors.
    slice_path : str
        Directory containing {stem}_gen_seq.pt optimised bin tensors.
    edit_start : int
        bp start of the editable bin within the full sequence.
    edit_end : int
        bp end of the editable bin within the full sequence.
    """
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
    """Reconstruct full sequences by splicing in two optimised anchor bins.

    Used for dot optimization, where two anchor bins (lo and hi) flanking the
    dot position are optimised simultaneously by Ledidi. The saved _gen_seq.pt
    contains a (1, 4, 2*bin_size) tensor with the lo anchor in [:, :, :bin_size]
    and the hi anchor in [:, :, bin_size:], which are split and inserted at their
    respective positions in the full original sequence.

    Parameters
    ----------
    coord_df : pd.DataFrame
        Table with columns [chrom, centered_start, centered_end].
    seq_path : str
        Directory containing {stem}_X.pt full sequence tensors.
    slice_path : str
        Directory containing {stem}_gen_seq.pt optimised anchor tensors.
    bp_lo_start : int
        bp start of the lo anchor bin within the full sequence.
    bp_lo_end : int
        bp end of the lo anchor bin within the full sequence.
    bp_hi_start : int
        bp start of the hi anchor bin within the full sequence.
    bp_hi_end : int
        bp end of the hi anchor bin within the full sequence.
    bin_size : int
        Size of each anchor bin in bp (default 2048).
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


# =================================================================================================
# dinucletide-preserving shuffling
# =================================================================================================

# Shuffle constants
K           = 2
BASES       = np.array(["A", "C", "G", "T"])
BASE_TO_IDX = {"A": 0, "C": 1, "G": 2, "T": 3}


def ohe_to_bytes(ohe: np.ndarray) -> bytes:
    """Convert a (4, L) OHE array to a byte string.

    Returns
    -------
    bytes
        ASCII-encoded DNA string, e.g. b"ACGTACGT...".
    """
    indices = np.argmax(ohe, axis=0)           # (L,)
    return "".join(BASES[indices]).encode()


def bytes_to_ohe(seq: bytes, length: int) -> np.ndarray:
    """Convert a byte string back to a (4, L) OHE float32 array.

    Parameters
    ----------
    seq : bytes
        ASCII-encoded DNA string, e.g. b"ACGTACGT...".
    length : int
        Expected sequence length (used to pre-allocate the output array).

    Returns
    -------
    np.ndarray
        Shape (4, length), dtype float32, channel order A=0, C=1, G=2, T=3.
    """
    ohe = np.zeros((4, length), dtype=np.float32)
    for i, base in enumerate(seq.decode()):
        ohe[BASE_TO_IDX[base], i] = 1.0
    return ohe


class ShuffledCentralInsertionDataset(Dataset):
    """Reconstruct full sequences with a dinucleotide-shuffled optimised bin.

    For each window, the optimised central bin (_gen_seq.pt) is shuffled using
    seqpro's dinucleotide-preserving shuffle, then spliced back into the full
    original sequence at the same position as the optimised bin.

    Parameters
    ----------
    coord_df   : DataFrame with chrom, centered_start, centered_end columns.
    seq_path   : Directory containing {stem}_X.pt full sequence tensors.
    slice_path : Directory containing {stem}_gen_seq.pt optimised bin tensors.
    edit_start : bp start of the central bin in the full sequence.
    edit_end   : bp end   of the central bin in the full sequence.
    """

    def __init__(self, coord_df: pd.DataFrame, seq_path: str, slice_path: str,
                 edit_start: int, edit_end: int):
        self.coords     = coord_df
        self.seq_path   = seq_path
        self.slice_path = slice_path
        self.edit_start = edit_start
        self.edit_end   = edit_end

    def __len__(self) -> int:
        return len(self.coords)

    def __getitem__(self, idx: int) -> torch.Tensor:
        row   = self.coords.iloc[idx]
        chrom = row["chrom"]
        start = int(row["centered_start"])
        end   = int(row["centered_end"])
        stem  = f"{chrom}_{start}_{end}"

        X      = torch.load(f"{self.seq_path}{stem}_X.pt",         weights_only=True)
        slice_ = torch.load(f"{self.slice_path}{stem}_gen_seq.pt",  weights_only=True)

        # Dinucleotide-preserving shuffle: OHE → bytes → k_shuffle → OHE
        ohe_np      = slice_.squeeze(0).numpy()          # (4, BIN_SIZE)
        seq_bytes   = ohe_to_bytes(ohe_np)               # b"ACGT..."
        shuffled_bytes = b"".join(sp.k_shuffle(seq_bytes, k=K, alphabet=b"ACGT"))
        shuffled_ohe   = bytes_to_ohe(shuffled_bytes, 2048)  # (4, BIN_SIZE)
        shuffled = torch.from_numpy(shuffled_ohe).unsqueeze(0)   # (1, 4, BIN_SIZE)

        edited = X.clone()
        edited[:, :, self.edit_start:self.edit_end] = shuffled
        return edited.squeeze(0)