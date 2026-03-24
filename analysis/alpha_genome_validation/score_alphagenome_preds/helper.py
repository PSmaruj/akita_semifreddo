"""
alpha_genome_utils.py
---------------------
Shared helpers for running Alpha Genome predictions and computing
feature scores from contact maps.

Imported by run_boundary_alphagenome.py, run_flame_alphagenome.py,
and run_dot_alphagenome.py.
"""

import numpy as np
import pandas as pd
from typing import Callable
from alphagenome.models import dna_client

# ── Constants ──────────────────────────────────────────────────────────────────

ORGANISM = dna_client.Organism.MUS_MUSCULUS
ONTOLOGY = ["EFO:0004038"]  # mESC

# ── FASTA I/O ──────────────────────────────────────────────────────────────────

def read_fasta(fasta_path: str) -> str:
    """Read a single-record FASTA and return the sequence as one uppercase string."""
    seq_lines = []
    with open(fasta_path) as f:
        for line in f:
            if not line.startswith(">"):
                seq_lines.append(line.strip())
    return "".join(seq_lines).upper()

# ── Alpha Genome prediction ────────────────────────────────────────────────────

def predict_contact_map(dna_model, seq: str) -> np.ndarray:
    """
    Run Alpha Genome on a sequence and return the 2D contact map as a numpy array.

    Parameters
    ----------
    dna_model :
        Alpha Genome model instance from dna_client.create().
    seq : str
        Nucleotide sequence string.

    Returns
    -------
    np.ndarray
        2D contact map array (shape: [N, N]).
    """
    output = dna_model.predict_sequence(
        organism=ORGANISM,
        sequence=seq,
        requested_outputs=[dna_client.OutputType.CONTACT_MAPS],
        ontology_terms=ONTOLOGY,
    )
    return output.contact_maps.values[:, :, 0]

# ── Scoring functions ──────────────────────────────────────────────────────────

def boundary_score(matrix: np.ndarray) -> float:
    """URQ mean insulation score for boundary designs."""
    return float(np.nanmean(matrix[0:250, 260:512]))


def dot_score(
    matrix: np.ndarray,
    dot_dist: int = 50,
    dot_width_half: int = 7,
    map_center: int = 256,
) -> float:
    """
    Mean contact frequency in a square window centred on the expected dot position.

    Parameters
    ----------
    matrix : np.ndarray
        2D contact map array (shape: [N, N]).
    dot_dist : int
        Distance between the two dot anchors in bins (default: 50).
    dot_width_half : int
        Half-width of the scoring window in bins (default: 7, giving a 14×14 window).
    map_center : int
        Centre bin of the contact map (default: 256 for a 512×512 map).
    """
    dist_half = dot_dist // 2
    dot_r = map_center - dist_half
    dot_c = map_center + dist_half
    return float(np.nanmean(
        matrix[dot_r - dot_width_half : dot_r + dot_width_half,
               dot_c - dot_width_half : dot_c + dot_width_half]
    ))


def flame_score(
    matrix: np.ndarray,
    flame_width: int = 3,
    map_size: int = 512,
) -> float:
    """
    Mean contact frequency in a narrow vertical stripe in the upper half
    of the contact map, centred on the map's column midpoint.

    Parameters
    ----------
    matrix : np.ndarray
        2D contact map array (shape: [N, N]).
    flame_width : int
        Width of the scoring stripe in bins (default: 3).
    map_size : int
        Height/width of the contact map (default: 512).
    """
    half_r = map_size // 2
    half_c = map_size // 2
    half_flame_width = flame_width // 2
    return float(np.nanmean(
        matrix[:half_r, half_c - half_flame_width : half_c + half_flame_width]
    ))


# ── Score loop ─────────────────────────────────────────────────────────────────

def score_fasta_dir(
    dna_model,
    df: pd.DataFrame,
    fasta_dir: str,
    scoring_fn: Callable[[np.ndarray], float],
    label: str = "",
) -> list[float]:
    """
    Predict and score contact maps for all loci in df.

    Parameters
    ----------
    dna_model :
        Alpha Genome model instance.
    df : pd.DataFrame
        Table with columns chrom, centered_start, centered_end.
    fasta_dir : str
        Directory containing per-locus FASTA files named {chrom}_{start}_{end}.fasta.
    scoring_fn : callable
        Function that takes a 2D contact map array and returns a float score.
        e.g. boundary_score, dot_score, flame_score.
    label : str
        Short label for progress printing (e.g. "original", "designed").

    Returns
    -------
    list of float
        Score per locus; NaN for any locus that failed.
    """
    scores   = []
    failures = []

    for i, row in enumerate(df.itertuples(index=False)):
        chrom, start, end = row.chrom, row.centered_start, row.centered_end
        locus      = f"{chrom}_{start}_{end}"
        fasta_path = f"{fasta_dir}/{locus}.fasta"
        print(f"  [{i:>4}] {label} {locus}", end=" ... ", flush=True)

        try:
            seq    = read_fasta(fasta_path)
            matrix = predict_contact_map(dna_model, seq)
            score  = scoring_fn(matrix)
            scores.append(score)
            print(f"{score:.4f}")
        except Exception as e:
            print(f"FAILED ({e})")
            failures.append(locus)
            scores.append(float("nan"))

    if failures:
        print(f"\n  WARNING: {len(failures)} loci failed for {label}:")
        for loc in failures:
            print(f"    {loc}")

    return scores