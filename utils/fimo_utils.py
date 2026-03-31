"""
utils/fimo_utils.py

FIMO-based motif scanning and PWM scoring utilities for the AkitaSF pipeline.

FIMO / CTCF utilities
---------------------
run_fimo                         : run FIMO on a (1, 4, L) sequence tensor
ctcf_hits_per_seq                : summarize FIMO hits per sequence in a batch
ctcf_hits_from_fimo              : bin FIMO hits by strand into per-bin arrays
hits_to_site_set                 : convert hits DataFrame to a set of (position, strand) tuples
jaccard_index                    : Jaccard similarity between two site sets

PWM I/O and sequence utilities
-------------------------------
get_sequence                     : extract an uppercase DNA string from a pyfaidx Fasta
reverse_complement               : reverse-complement a DNA string
reverse_complement_pwm           : reverse-complement a (L, 4) PWM array
read_meme_pwm                    : parse a MEME file → (4, L) torch tensor
read_meme_pwm_as_numpy           : parse a MEME file → (L, 4) numpy array

Scoring
-------
estimate_background_probs        : estimate per-base background frequencies from genomic windows
seq_score                        : log-likelihood score of a sequence window against a PWM
sliding_scores                   : sliding-window PWM scores over a sequence
aggregated_positive_motif_score  : sum of positive-strand PWM scores across both strands
compute_aggregated_positive_motif_scores : compute and store motif scores for all rows in a DataFrame
"""

import numpy as np
import torch
from memelite import fimo
import pandas as pd
from pyfaidx import Fasta
from tqdm import tqdm
from collections import Counter

from .data_utils import get_sequence


# ---------------------------------------------------------------------------
# FIMO / CTCF utilities
# ---------------------------------------------------------------------------


def run_fimo(seq_tensor, motifs_dict, threshold=1e-4):
    """Run FIMO motif scanning on a batch of one-hot encoded sequences.

    Parameters
    ----------
    seq_tensor : torch.Tensor
        Shape (1, 4, L); converted to NumPy internally.
    motifs_dict : dict
        Motif dictionary as expected by memelite.fimo,
        e.g. {"CTCF": pwm_tensor} with PWM shape (4, motif_len).
    threshold : float
        p-value threshold for reporting hits (default 1e-4).

    Returns
    -------
    pd.DataFrame
        FIMO hits table with columns [sequence_name, start, end, strand, score].
    """
    arr = seq_tensor.cpu().detach().numpy()
    return fimo(motifs=motifs_dict, sequences=arr,
                threshold=threshold, reverse_complement=True)[0]


def ctcf_hits_per_seq(hits: pd.DataFrame, batch_size: int) -> list[dict]:
    """Summarize FIMO hits per sequence index within a batch.

    Parameters
    ----------
    hits : pd.DataFrame
        FIMO hits table with columns [sequence_name, start, end, strand, score].
    batch_size : int
        Number of sequences in the batch; used to iterate over sequence indices.

    Returns
    -------
    list of dict
        One dict per sequence with keys:
        n, score_sum, score_max, positions (list of (start, end) tuples), strands.
        Sequences with no hits get n=0, score_sum=0.0, score_max=0.0.
    """
    records = []
    for seq_idx in range(batch_size):
        eh = hits[hits["sequence_name"] == seq_idx]
        if eh.empty:
            records.append(dict(n=0, score_sum=0.0, score_max=0.0,
                                positions=[], strands="no"))
        else:
            eh = eh.sort_values("start")
            records.append(dict(
                n         = len(eh),
                score_sum = float(eh["score"].sum()),
                score_max = float(eh["score"].max()),
                positions = [(int(s), int(e)) for s, e in zip(eh["start"], eh["end"])],
                strands   = "".join(eh["strand"].tolist()),
            ))
    return records


def ctcf_hits_from_fimo(fimo_df, seq_len=1310720, bin_size=2048):
    """Bin FIMO hits by strand into per-bin hit count arrays.

    Parameters
    ----------
    fimo_df : pd.DataFrame
        FIMO hits table with columns [start, strand].
    seq_len : int
        Full sequence length in bp (default 1,310,720).
    bin_size : int
        Bin size in bp (default 2048).

    Returns
    -------
    hits_plus : np.ndarray
        Shape (n_bins,); hit counts on the + strand per bin.
    hits_minus : np.ndarray
        Shape (n_bins,); hit counts on the − strand per bin.
    """
    n_bins = seq_len // bin_size
    hits_plus  = np.zeros(n_bins)
    hits_minus = np.zeros(n_bins)
    for _, row in fimo_df.iterrows():
        bin_idx = int(row["start"] // bin_size)
        if bin_idx >= n_bins:
            continue
        if row["strand"] == "+":
            hits_plus[bin_idx]  += 1
        else:
            hits_minus[bin_idx] += 1
    return hits_plus, hits_minus


def hits_to_site_set(hits_df, bin_size=10):
    """Convert a FIMO hits DataFrame to a set of binned (position, strand) tuples.

    Each hit's center position is rounded to the nearest bin_size boundary,
    producing a position that is comparable across runs for Jaccard analysis.

    Parameters
    ----------
    hits_df : pd.DataFrame
        FIMO hits table with columns [start, end, strand].
    bin_size : int
        Bin width in bp for position rounding (default 10).

    Returns
    -------
    set of (int, str)
        Each element is a (binned_center_position, strand) tuple.
    """
    site_set = set()
    for _, row in hits_df.iterrows():
        start, end, strand = row['start'], row['end'], row['strand']
        center = (start + end) // 2
        binned_pos = round(center / bin_size) * bin_size
        site_set.add((binned_pos, strand))
    return site_set


def jaccard_index(set1, set2):
    """Compute the Jaccard similarity index between two sets.

    Returns 1.0 if both sets are empty (identical by convention).

    Parameters
    ----------
    set1, set2 : set
        Sets to compare, typically from hits_to_site_set.

    Returns
    -------
    float
        |set1 ∩ set2| / |set1 ∪ set2|, or 1.0 if both are empty.
    """
    if not set1 and not set2:
        return 1.0
    union = set1 | set2
    return len(set1 & set2) / len(union) if union else 0.0


# ---------------------------------------------------------------------------
# PWM I/O and sequence utilities
# ---------------------------------------------------------------------------


def reverse_complement(seq):
    """Return the reverse complement of a DNA string.

    Handles both upper and lower case bases.

    Parameters
    ----------
    seq : str
        DNA string, e.g. 'ACGTacgt'.

    Returns
    -------
    str
        Reverse-complemented DNA string.
    """
    comp = str.maketrans("ACGTacgt", "TGCAtgca")
    return seq.translate(comp)[::-1]


def reverse_complement_pwm(pwm: np.ndarray) -> np.ndarray:
    """Return the reverse complement of a PWM.

    Parameters
    ----------
    pwm : np.ndarray
        Shape (L, 4), column order A=0, C=1, G=2, T=3.

    Returns
    -------
    np.ndarray
        Reverse-complemented PWM of the same shape.
    """
    return np.flipud(pwm)[:, [3, 2, 1, 0]].copy()


def read_meme_pwm_as_numpy(filename: str) -> np.ndarray:
    """Parse a MEME-format file and return the first PWM as a (L, 4) array.

    Counterpart to read_meme_pwm, which returns a (4, L) torch tensor.

    Parameters
    ----------
    filename : str
        Path to a MEME-format motif file.

    Returns
    -------
    np.ndarray
        Shape (motif_len, 4), dtype float32, column order A=0, C=1, G=2, T=3.
    """
    rows, in_matrix = [], False
    with open(filename) as fh:
        for line in fh:
            line = line.strip()
            if line.startswith("letter-probability matrix"):
                in_matrix = True
                continue
            if in_matrix and line.startswith("MOTIF"):
                break
            if in_matrix and line:
                rows.append([float(v) for v in line.split()])
    return np.array(rows, dtype=np.float32)


def read_meme_pwm(filename: str) -> torch.Tensor:
    """Parse a MEME-format file and return the first PWM as a (4, L) tensor.

    Parameters
    ----------
    filename : str
        Path to a MEME-format motif file.

    Returns
    -------
    torch.Tensor
        Shape (4, motif_len), dtype float32, row order A=0, C=1, G=2, T=3.
    """
    pwm = read_meme_pwm_as_numpy(filename)   # (L, 4)
    return torch.from_numpy(pwm.T)            # (4, L)


# ---------------------------------------------------------------------------
# Background probabilities
# ---------------------------------------------------------------------------


def estimate_background_probs(
    df,
    genome: Fasta,
    chrom_col: str = "chrom",
    start_col: str = "centered_start",
    end_col: str = "centered_end",
) -> dict[str, float]:
    """Estimate nucleotide background probabilities from a set of genomic windows.

    Counts A/C/G/T across all sequences in df (ignoring Ns), then normalises.

    Parameters
    ----------
    df : pd.DataFrame
        Table with genomic window rows.
    genome : pyfaidx.Fasta
        Reference genome.
    chrom_col : str
        Column name for chromosome (default 'chrom').
    start_col : str
        Column name for window start coordinate (default 'centered_start').
    end_col : str
        Column name for window end coordinate (default 'centered_end').

    Returns
    -------
    dict of str → float
        Mapping each base in "ACGT" to its relative frequency.
    """
    counts = Counter()
    for _, row in df.iterrows():
        seq = get_sequence(genome, row[chrom_col], row[start_col], row[end_col])
        counts.update(b for b in seq if b in "ACGT")

    total = sum(counts.values())
    if total == 0:
        raise ValueError("No valid (A/C/G/T) bases found across the provided windows.")

    return {b: counts[b] / total for b in "ACGT"}


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

_BASE_IDX = {"A": 0, "C": 1, "G": 2, "T": 3}


def seq_score(
    seq: str,
    pwm: np.ndarray,
    bg: dict[str, float] | None = None,
    pseudocount: float = 1e-9,
) -> float:
    """
    Log-likelihood score of a sequence window against a PWM.

    Args:
        seq:         DNA string of length == pwm.shape[0].
        pwm:         (L, 4) probability matrix, column order A/C/G/T.
        bg:          Background nucleotide probabilities. Defaults to uniform.
        pseudocount: Added to PWM probabilities before taking log.

    Returns:
        Scalar log2-likelihood ratio score.
    """
    if bg is None:
        bg = {"A": 0.25, "C": 0.25, "G": 0.25, "T": 0.25}

    score = 0.0
    for i, b in enumerate(seq.upper()):
        if b not in _BASE_IDX:
            continue
        p = pwm[i, _BASE_IDX[b]] + pseudocount
        score += np.log2(p / bg[b])
    return score


def sliding_scores(
    seq: str,
    pwm: np.ndarray,
    bg: dict[str, float] | None = None,
    step: int = 1,
) -> np.ndarray:
    """
    Sliding-window PWM scores over a sequence.

    Args:
        seq:  DNA string.
        pwm:  (L, 4) PWM array.
        bg:   Background probabilities passed to seq_score.
        step: Stride between windows.

    Returns:
        1-D array of scores, length = (len(seq) - L) // step + 1.
    """
    L = pwm.shape[0]
    return np.array(
        [seq_score(seq[i: i + L], pwm, bg) for i in range(0, len(seq) - L + 1, step)]
    )
    
    
def aggregated_positive_motif_score(
    seq: str,
    pwm: np.ndarray,
    bg: dict[str, float] | None = None,
    step: int = 1,
) -> float:
    """Sum of positive-valued sliding-window scores across both strands.

    For each position the strand score is max(forward_score, rc_score),
    and only positions with a combined score > 0 contribute to the sum.

    Parameters
    ----------
    seq : str
        DNA string.
    pwm : np.ndarray
        Shape (L, 4) forward PWM array.
    bg : dict of str → float, optional
        Background probabilities passed to seq_score.
    step : int
        Stride for the sliding window (default 1).

    Returns
    -------
    float
        Aggregated positive motif score.
    """
    pwm_rc = reverse_complement_pwm(pwm)
    fwd = sliding_scores(seq, pwm, bg=bg, step=step)
    rev = sliding_scores(seq, pwm_rc, bg=bg, step=step)
    combined = np.maximum(fwd, rev)
    return float(combined[combined > 0].sum())


def compute_aggregated_positive_motif_scores(
    df,
    genome: Fasta,
    pwm: np.ndarray,
    seq_start_offset: int,
    seq_end_offset: int,
    bg: dict[str, float] | None = None,
    step: int = 1,
    chrom_col: str = "chrom",
    start_col: str = "centered_start",
    score_col: str = "sum_positive_scores",
) -> None:
    """Compute aggregated positive motif scores for every row in df (in-place).

    The extracted sequence spans
    genome[chrom][start + seq_start_offset : start + seq_end_offset].

    Parameters
    ----------
    df : pd.DataFrame
        Table with genomic window rows; modified in-place.
    genome : pyfaidx.Fasta
        Reference genome.
    pwm : np.ndarray
        Shape (L, 4) PWM array.
    seq_start_offset : int
        Offset from start_col to begin sequence extraction.
    seq_end_offset : int
        Offset from start_col to end sequence extraction.
    bg : dict of str → float, optional
        Background probabilities (None → uniform).
    step : int
        Sliding-window stride (default 1).
    chrom_col : str
        Column name for chromosome (default 'chrom').
    start_col : str
        Column name for window start coordinate (default 'centered_start').
    score_col : str
        Name of the new score column written into df
        (default 'sum_positive_scores').
    """
    df[score_col] = 0.0
    for i, row in tqdm(df.iterrows(), total=len(df)):
        seq = get_sequence(
            genome,
            row[chrom_col],
            row[start_col] + seq_start_offset,
            row[start_col] + seq_end_offset,
        )
        df.at[i, score_col] = aggregated_positive_motif_score(seq, pwm, bg=bg, step=step)
