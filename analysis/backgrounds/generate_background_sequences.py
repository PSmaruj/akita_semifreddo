"""
generate_background_sequences.py

Generates negative control (background) sequences for the AkitaSF sequence
design pipeline by k-mer-shuffling mouse genomic regions and filtering for
minimal predicted genome folding.

Input genomic regions are drawn from a TSV of 50 pre-selected mouse loci
spanning a range of GC content. Each region is shuffled with seqpro's
k-mer-preserving shuffle (k=8), one-hot encoded, and passed through the
AkitaPT model (Akita v2 fine-tuned on mouse mESC Hsieh2019 Hi-C). Candidate
sequences are retained only when both:
  - SCD  (root-sum-of-squares of predicted contact map values) < 30
  - Total variation (sum of |Δ| between adjacent bins on both axes)  < 1300

For each input locus, shuffling is retried up to MAX_TRIES_PER_SEQ times.
Accepted sequences are written to a FASTA file with per-sequence QC metrics
embedded in the header.

Usage
-----
    python generate_background_sequences.py

Outputs
-------
    FASTA file at OUTPUT_FASTA, one record per accepted background sequence.
    Header format:
        >shuffled_<i>_<chrom>_<start>_<end>_scd<SCD>_totvar<totvar>
"""

import os
import sys
import random

import numpy as np
import pandas as pd
import torch
import seqpro as sp
from pyfaidx import Fasta

sys.path.append(os.path.abspath("/home1/smaruj/pytorch_akita/"))
sys.path.insert(0, os.path.abspath("/home1/smaruj/ledidi_akita/"))

from utils.model_utils import load_model
from utils.data_utils import one_hot_encode_sequence, from_upper_triu


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
MODEL_CKPT = (
    "/home1/smaruj/pytorch_akita/models/finetuned/mouse/Hsieh2019_mESC"
    "/checkpoints/Akita_v2_mouse_Hsieh2019_mESC_model0_finetuned.pth"
)
GENOME_FASTA = "/project2/fudenber_735/genomes/mm10/mm10.fa"
TSV_PATH = (
    "/home1/smaruj/akitaV2-analyses/experiments/background_generation"
    "/background_generation/input_data/50seqs_GCuniform_maxSCD35.tsv"
)
OUTPUT_FASTA = (
    "/project2/fudenber_735/smaruj/sequence_design/ledidi_semifreddo_akita"
    "/analysis/background_generation/background_sequences_scd30_totvar1300.fasta"
)

# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------
K                    = 8      # k-mer size for seqpro shuffle
SEQ_LENGTH           = 1310720  # 1.3 Mb context window
MAX_TRIES_PER_SEQ    = 20     # shuffling attempts per locus before skipping
TARGET_COUNT         = 10     # number of background sequences to collect
SCD_THRESHOLD        = 30     # max allowed SCD score
TOTVAR_THRESHOLD     = 1300   # max allowed total variation


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def total_variation(contact_map: np.ndarray) -> float:
    """
    Sum of absolute differences between adjacent bins along both axes of a
    predicted contact map, ignoring NaN entries (off-diagonal padding).

    Parameters
    ----------
    contact_map : np.ndarray, shape (L, L)
        Symmetrised contact map reconstructed from upper-triangular predictions.

    Returns
    -------
    float
        Total variation score.
    """
    dx = contact_map[:, 1:] - contact_map[:, :-1]
    dy = contact_map[1:, :] - contact_map[:-1, :]
    return float(np.sum(np.abs(dx[~np.isnan(dx)])) + np.sum(np.abs(dy[~np.isnan(dy)])))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    # --- Device ---
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Model ---
    model = load_model(MODEL_CKPT, device)
    model.eval()

    # --- Data ---
    df = pd.read_csv(TSV_PATH, sep="\t")
    genome = Fasta(GENOME_FASTA)

    seen_indices = set()
    saved_seqs   = []
    scd_scores   = []
    totvar_scores = []

    # --- Shuffling loop ---
    print(f"Targeting {TARGET_COUNT} background sequences "
          f"(SCD < {SCD_THRESHOLD}, total variation < {TOTVAR_THRESHOLD})\n")

    with torch.no_grad():
        while len(saved_seqs) < TARGET_COUNT and len(seen_indices) < len(df):
            # Pick an unseen locus at random
            remaining = df.index.difference(seen_indices)
            idx = random.choice(remaining)
            seen_indices.add(idx)

            row = df.loc[idx]
            chrom, start, end = row["chrom"], row["start"], row["end"]

            # Load and pad genomic sequence to fixed length
            seq = genome[chrom][start:end].seq.upper()
            seq = seq[:SEQ_LENGTH].ljust(SEQ_LENGTH, "N")

            for attempt in range(1, MAX_TRIES_PER_SEQ + 1):
                # k-mer-preserving shuffle
                shuffled_seq = b"".join(sp.k_shuffle(seq.encode(), k=K, alphabet=b"ACGT")).decode()

                # Model forward pass
                one_hot = one_hot_encode_sequence(shuffled_seq)
                batch   = torch.from_numpy(one_hot).unsqueeze(0).to(device)  # (1, 4, L)
                preds   = model(batch).cpu()  # (1, n_triu)

                # QC metrics
                scd      = torch.sqrt((preds ** 2).sum()).item()
                pred_map = from_upper_triu(preds.squeeze(0), matrix_len=512, num_diags=2)
                tot_var  = total_variation(pred_map)

                print(f"  [idx={idx} | attempt {attempt:>2}/{MAX_TRIES_PER_SEQ}] "
                      f"SCD={scd:.2f}  tot_var={tot_var:.2f}")

                if scd < SCD_THRESHOLD and tot_var < TOTVAR_THRESHOLD:
                    saved_seqs.append((chrom, start, end, shuffled_seq))
                    scd_scores.append(scd)
                    totvar_scores.append(tot_var)
                    print(f"  → Accepted #{len(saved_seqs):>2} "
                          f"(idx={idx}, SCD={scd:.2f}, tot_var={tot_var:.2f})\n")
                    break
            else:
                print(f"  → idx={idx} exhausted {MAX_TRIES_PER_SEQ} attempts — skipping.\n")

    # --- Save FASTA ---
    os.makedirs(os.path.dirname(OUTPUT_FASTA), exist_ok=True)
    with open(OUTPUT_FASTA, "w") as f:
        for i, (chrom, start, end, seq) in enumerate(saved_seqs):
            header = (f">shuffled_{i}_{chrom}_{start}_{end}"
                      f"_scd{scd_scores[i]:.2f}_totvar{totvar_scores[i]:.2f}")
            f.write(f"{header}\n{seq}\n")

    print(f"Done. {len(saved_seqs)} / {TARGET_COUNT} sequences saved → {OUTPUT_FASTA}")


if __name__ == "__main__":
    main()