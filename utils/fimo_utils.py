import numpy as np
import torch
from memelite import fimo
import pandas as pd
from pyfaidx import Fasta
from tqdm import tqdm


def run_fimo(seq_tensor, motifs_dict, threshold=1e-4):
    """Run FIMO on a (1, 4, L) tensor; returns the hits DataFrame."""
    arr = seq_tensor.cpu().detach().numpy()
    return fimo(motifs=motifs_dict, sequences=arr,
                threshold=threshold, reverse_complement=True)[0]


def ctcf_hits_per_seq(hits: pd.DataFrame, batch_size: int) -> list[dict]:
    """Summarise FIMO hits per sequence index within a batch."""
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
    """Bin FIMO hits by strand → (hits_plus, hits_minus) arrays of length n_bins."""
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
    site_set = set()
    for _, row in hits_df.iterrows():
        start, end, strand = row['start'], row['end'], row['strand']
        center = (start + end) // 2
        binned_pos = round(center / bin_size) * bin_size
        site_set.add((binned_pos, strand))
    return site_set


def jaccard_index(set1, set2):
    if not set1 and not set2:
        return 1.0
    union = set1 | set2
    return len(set1 & set2) / len(union) if union else 0.0


def get_sequence(genome: Fasta, chrom: str, start: int, end: int) -> str:
    """Extract an uppercase DNA sequence from a pyfaidx Fasta object."""
    return genome[chrom][start:end].seq.upper()


def reverse_complement(seq):
    comp = str.maketrans("ACGTacgt", "TGCAtgca")
    return seq.translate(comp)[::-1]


def reverse_complement_pwm(pwm: np.ndarray) -> np.ndarray:
    """
    Return the reverse-complement of a PWM.
    Assumes column order A=0, C=1, G=2, T=3.
    """
    return np.flipud(pwm)[:, [3, 2, 1, 0]].copy()


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
    """
    Estimate nucleotide background probabilities from a set of genomic windows.

    Counts A/C/G/T across all sequences in df (ignoring Ns), then normalises.

    Args:
        df:        DataFrame with genomic window rows.
        genome:    pyfaidx Fasta object.
        chrom_col: Column name for chromosome.
        start_col: Column name for window start coordinate.
        end_col:   Column name for window end coordinate.

    Returns:
        Dict mapping each base in "ACGT" to its relative frequency.
    """
    from collections import Counter

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


def read_meme_pwm_as_numpy(filename: str) -> np.ndarray:
    """
    Parse a MEME-format file and return the PWM as a (L, 4) float32 array.
    (Counterpart to read_meme_pwm, which returns a (4, L) torch tensor.)
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
    """Parse a MEME-format file and return the PWM as a (4, L) float32 tensor."""
    pwm = read_meme_pwm_as_numpy(filename)   # (L, 4)
    return torch.from_numpy(pwm.T)            # (4, L)


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
    """
    Sum of all positive-valued sliding-window scores across both strands.

    For each position the strand score is max(forward_score, rc_score),
    and only positions with a combined score > 0 contribute to the sum.

    Args:
        seq:  DNA string.
        pwm:  (L, 4) forward PWM array.
        bg:   Background probabilities passed to seq_score.
        step: Stride for the sliding window.

    Returns:
        Scalar aggregated positive motif score.
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
    """
    Compute aggregated positive motif scores for every row in df (in-place).

    The extracted sequence is genome[chrom][centered_start + seq_start_offset :
                                            centered_start + seq_end_offset].

    Args:
        df:               DataFrame with genomic window rows.
        genome:           pyfaidx Fasta object.
        pwm:              (L, 4) PWM array.
        seq_start_offset: Offset from centered_start to begin sequence extraction.
        seq_end_offset:   Offset from centered_start to end sequence extraction.
        bg:               Background probabilities (None → uniform).
        step:             Sliding-window stride.
        chrom_col:        Column name for chromosome.
        start_col:        Column name for window start coordinate.
        score_col:        Name of the new score column written into df.
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
