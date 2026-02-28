#!/usr/bin/env python
"""
Insert a strong CTCF in the middle of a background sequence,
then flank it with varying numbers of SINE B2 + CTCF elements
and measure Akita predictions.

Conditions (for each of the top 100 CTCFs):
1. Background only (once)
2. Background + central CTCF
3. Background + central CTCF +  2 SINE B2s (1 per side)
4. Background + central CTCF +  4 SINE B2s (2 per side)
5. Background + central CTCF +  6 SINE B2s (3 per side)
6. Background + central CTCF +  8 SINE B2s (4 per side)
7. Background + central CTCF + 10 SINE B2s (5 per side)

SINE B2s are placed symmetrically around the CTCF with 100 bp spacing.
Top 5 SINE B2 sequences are used cyclically.

Measures: upper-right quarter mean of the predicted contact map.
"""

import numpy as np
import pandas as pd
from Bio import SeqIO
from Bio.Seq import Seq
import pysam
import torch
import sys
import os

sys.path.insert(0, "/home1/smaruj/pytorch_akita")
from akita_model.model import SeqNN  # adjust import path if needed

# ── Paths ──────────────────────────────────────────────────────────────────────
GENOME_FASTA = "/project2/fudenber_735/genomes/mm10/mm10.fa"
BACKGROUND_FASTA = (
    "/project2/fudenber_735/smaruj/sequence_design/"
    "ledidi_semifreddo_akita/background_generation/"
    "background_sequences_scd30_totvar1300.fasta"
)
CTCF_TSV = (
    "/home1/smaruj/akitaV2-analyses/input_data/select_top20percent/"
    "output/CTCFs_jaspar_filtered_mm10_top20percent.tsv"
)
SINEB2_WITH_CTCF = "sineB2_with_ctcf_300.tsv"
MODEL_PATH = (
    "/home1/smaruj/pytorch_akita/models/finetuned/mouse/"
    "Hsieh2019_mESC/checkpoints/"
    "Akita_v2_mouse_Hsieh2019_mESC_model0_finetuned.pth"
)

# ── Parameters ─────────────────────────────────────────────────────────────────
SEQ_LEN = 1_310_720       # 1.3 Mb, Akita input size
MATRIX_LEN = 512          # Akita output map size
NUM_DIAGS = 2             # number of diagonals set to NaN
TOP_N_CTCF = 100          # use top 100 CTCFs
TOP_N_SINEB2 = 5          # use top 5 SINE B2s (cycled through)
SINEB2_COUNTS = [2, 4, 6, 8, 10]  # total SINE B2s to test (split evenly per side)
SPACING_BP = 100          # spacing between element edges


# ── Helper: one-hot encode ─────────────────────────────────────────────────────
def one_hot_encode(seq_str):
    """One-hot encode a DNA sequence string to (L, 4) array. Order: ACGT."""
    mapping = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    seq_str = seq_str.upper()
    arr = np.zeros((len(seq_str), 4), dtype=np.float32)
    for i, nt in enumerate(seq_str):
        if nt in mapping:
            arr[i, mapping[nt]] = 1.0
        else:
            arr[i, :] = 0.25
    return arr


# ── Helper: extract genomic sequence ──────────────────────────────────────────
def fetch_sequence(fasta_path, chrom, start, end, strand="+"):
    """Fetch sequence from an indexed fasta."""
    fasta = pysam.FastaFile(fasta_path)
    seq = fasta.fetch(chrom, start, end).upper()
    fasta.close()
    if strand == "-":
        seq = str(Seq(seq).reverse_complement())
    return seq


# ── Helper: upper triangle to matrix ─────────────────────────────────────────
def set_diag(matrix, value, k):
    """Set diagonal `k` of a matrix to `value`."""
    rows, cols = matrix.shape
    for i in range(rows):
        if 0 <= i + k < cols:
            matrix[i, i + k] = value


def from_upper_triu_batch(batch_vectors, matrix_len=MATRIX_LEN, num_diags=NUM_DIAGS):
    """Convert a batch of upper-triangular vectors into symmetric matrices
    with np.nan on diagonals."""
    if isinstance(batch_vectors, torch.Tensor):
        batch_vectors = batch_vectors.detach().cpu().numpy()

    batch_size = len(batch_vectors)
    matrices = np.zeros((batch_size, matrix_len, matrix_len), dtype=np.float32)
    triu_indices = np.triu_indices(matrix_len, num_diags)

    for i in range(batch_size):
        matrices[i][triu_indices] = batch_vectors[i][0, :]
        matrices[i] = matrices[i] + matrices[i].T
        for k in range(-num_diags + 1, num_diags):
            set_diag(matrices[i], np.nan, k)

    return matrices  # shape: [B, 512, 512]


# ── Load background sequence (first one) ──────────────────────────────────────
def load_background():
    """Load the first background sequence from the fasta."""
    records = list(SeqIO.parse(BACKGROUND_FASTA, "fasta"))
    bg_seq = str(records[0].seq).upper()
    assert len(bg_seq) == SEQ_LEN, f"Expected {SEQ_LEN}, got {len(bg_seq)}"
    print(f"Loaded background sequence: {records[0].id}, length={len(bg_seq)}")
    return bg_seq


# ── Load CTCF sites ───────────────────────────────────────────────────────────
def load_ctcf_sites():
    """
    Load CTCF sites, sort by insertion_SCD descending (strongest first).
    Columns: chrom, end, start, strand, disruption_SCD, insertion_SCD
    """
    df = pd.read_csv(CTCF_TSV, sep="\t")
    print(f"Loaded {len(df)} CTCF sites")
    print(f"Columns: {list(df.columns)}")

    df = df.sort_values("insertion_SCD", ascending=False)

    # df_top = df.head(TOP_N_CTCF).reset_index(drop=True)
    df_top = df.tail(TOP_N_CTCF).reset_index(drop=True)
    print(f"Using top {len(df_top)} CTCFs (by insertion_SCD)")
    return df_top


# ── Load SINE B2 + CTCF elements ─────────────────────────────────────────────
def load_sineb2_sites():
    """
    Load SINE B2 + CTCF sites, sort by swScore descending, take top N.
    Columns: rmsk_index, bin, swScore, milliDiv, milliDel, milliIns,
             genoName, genoStart, genoEnd, genoLeft, strand, repName,
             repClass, repFamily, repStart, repEnd, repLeft, id
    """
    df = pd.read_csv(SINEB2_WITH_CTCF, sep="\t")
    print(f"Loaded {len(df)} SINE B2 + CTCF sites")
    print(f"Columns: {list(df.columns)}")

    df = df.sort_values("swScore", ascending=False)

    df_top = df.head(TOP_N_SINEB2).reset_index(drop=True)
    print(f"Using top {len(df_top)} SINE B2 + CTCF elements (by swScore)")
    return df_top


# ── Extract element sequences from genome ─────────────────────────────────────
def get_ctcf_seq(row, fasta_path=GENOME_FASTA, force_strand="+"):
    """Extract CTCF sequence, forced to '+' strand."""
    chrom = row["chrom"]
    start = int(row["start"])
    end = int(row["end"])
    strand = force_strand if force_strand is not None else row["strand"]
    return fetch_sequence(fasta_path, chrom, start, end, strand)


def get_sineb2_seq(row, fasta_path=GENOME_FASTA, force_strand="+"):
    """Extract SINE B2 + CTCF sequence, forced to '+' strand."""
    chrom = row["genoName"]
    start = int(row["genoStart"])
    end = int(row["genoEnd"])
    strand = force_strand if force_strand is not None else row["strand"]
    return fetch_sequence(fasta_path, chrom, start, end, strand)


# ── Extract all SINE B2 sequences once ────────────────────────────────────────
def extract_sineb2_seqs(sineb2_sites):
    """Extract top SINE B2 + CTCF sequences (always '+' strand)."""
    sineb2_seqs = []
    for i in range(min(TOP_N_SINEB2, len(sineb2_sites))):
        row = sineb2_sites.iloc[i]
        seq = get_sineb2_seq(row, force_strand="+")
        sineb2_seqs.append(seq)
        print(f"  SINE B2 #{i}: length={len(seq)} bp, "
              f"original strand={row['strand']}, forced to: + (>)")
    return sineb2_seqs


# ── Insert sequence into background ──────────────────────────────────────────
def insert_element(bg_seq, element_seq, position):
    """
    Insert element_seq into bg_seq centered at position.
    Replaces the corresponding region (does not shift).
    """
    elem_len = len(element_seq)
    start = position - elem_len // 2
    end = start + elem_len
    if start < 0 or end > len(bg_seq):
        print(f"Warning: element at pos {position} (len {elem_len}) "
              f"goes out of bounds [{start}, {end}]. Clamping.")
        start = max(0, start)
        end = min(len(bg_seq), end)
        element_seq = element_seq[:end - start]
    return bg_seq[:start] + element_seq + bg_seq[end:]


# ── Build sequence with CTCF + N SINE B2s ────────────────────────────────────
def build_sequence_with_sineb2s(bg_seq, ctcf_row, sineb2_seqs, n_sineb2_total):
    """
    Insert CTCF at center, then place n_sineb2_total SINE B2 elements
    symmetrically around it (n_sineb2_total/2 per side) with SPACING_BP
    between edges. Top 5 SINE B2 sequences are used cyclically.

    All elements in forward ('+' / '>') orientation.
    """
    center = SEQ_LEN // 2

    # Insert CTCF at center
    ctcf_seq = get_ctcf_seq(ctcf_row, force_strand="+")
    ctcf_len = len(ctcf_seq)
    seq = insert_element(bg_seq, ctcf_seq, center)

    if n_sineb2_total == 0:
        return seq

    # Place SINE B2s symmetrically
    n_per_side = n_sineb2_total // 2
    half_ctcf = ctcf_len // 2

    for side in ['left', 'right']:
        prev_offset = None
        for i in range(n_per_side):
            sineb2_idx = i % len(sineb2_seqs)
            sineb2_len = len(sineb2_seqs[sineb2_idx])

            if i == 0:
                # First element: edge starts at CTCF edge + spacing
                offset = half_ctcf + SPACING_BP + sineb2_len // 2
            else:
                prev_sineb2_idx = (i - 1) % len(sineb2_seqs)
                prev_sineb2_len = len(sineb2_seqs[prev_sineb2_idx])
                offset = (prev_offset
                          + prev_sineb2_len // 2
                          + SPACING_BP
                          + sineb2_len // 2)

            if side == 'left':
                pos = center - offset
            else:
                pos = center + offset

            seq = insert_element(seq, sineb2_seqs[sineb2_idx], pos)
            prev_offset = offset

    return seq


# ── Model prediction ─────────────────────────────────────────────────────────
def predict_map(model, seq_str, device='cuda'):
    """
    Run Akita model on a single sequence and return the 512x512 contact map.
    """
    x = one_hot_encode(seq_str)                      # (L, 4)
    x = torch.from_numpy(x).unsqueeze(0)             # (1, L, 4)
    x = x.permute(0, 2, 1)                           # (1, 4, L) — channels first
    x = x.to(device)

    with torch.no_grad():
        pred = model(x)

    maps = from_upper_triu_batch(pred)  # (1, 512, 512)
    return maps[0]  # (512, 512)


def upper_right_quarter_mean(contact_map):
    """
    Compute the mean of the upper-right quarter of the contact map.
    Uses rows 0:250 and cols 260:512, matching existing analysis code.
    """
    return np.nanmean(contact_map[0:250, 260:512])


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("=" * 70)
    print("CTCF + varying SINE B2 count Insertion Experiment")
    print(f"Spacing: {SPACING_BP} bp")
    print(f"SINE B2 counts tested: {SINEB2_COUNTS}")
    print("=" * 70)

    # Load data
    print("\n── Loading data ──")
    bg_seq = load_background()
    ctcf_sites = load_ctcf_sites()
    sineb2_sites = load_sineb2_sites()

    # Extract SINE B2 sequences once
    print("\n── Extracting SINE B2 + CTCF sequences ──")
    sineb2_seqs = extract_sineb2_seqs(sineb2_sites)

    # Load model
    print("\n── Loading model ──")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    model = SeqNN()
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()
    print("Model loaded successfully")

    # ── Predict background (once) ──
    print("\n── Predicting background ──")
    bg_map = predict_map(model, bg_seq, device)
    bg_urq = upper_right_quarter_mean(bg_map)
    print(f"  Background URQ mean: {bg_urq:.6f}")

    # ── Loop over all top CTCFs ──
    print(f"\n── Looping over {len(ctcf_sites)} CTCFs ──")
    rows = []

    # One background row
    rows.append({
        "ctcf_index": -1,
        "ctcf_chrom": "NA",
        "ctcf_start": "NA",
        "ctcf_end": "NA",
        "condition": "background",
        "n_sineb2": 0,
        "urq_mean": bg_urq,
    })

    for ci in range(len(ctcf_sites)):
        ctcf_row = ctcf_sites.iloc[ci]
        ctcf_label = (f"{ctcf_row['chrom']}:"
                      f"{ctcf_row['start']}-{ctcf_row['end']}")
        print(f"\n  CTCF {ci}/{len(ctcf_sites)}: {ctcf_label}")

        row_base = {
            "ctcf_index": ci,
            "ctcf_chrom": ctcf_row["chrom"],
            "ctcf_start": ctcf_row["start"],
            "ctcf_end": ctcf_row["end"],
        }

        # ── CTCF only (0 SINE B2s) ──
        seq_ctcf = build_sequence_with_sineb2s(
            bg_seq, ctcf_row, sineb2_seqs, n_sineb2_total=0
        )
        assert len(seq_ctcf) == SEQ_LEN
        map_ctcf = predict_map(model, seq_ctcf, device)
        urq_ctcf = upper_right_quarter_mean(map_ctcf)
        print(f"    CTCF only           URQ mean: {urq_ctcf:.6f}")
        rows.append({**row_base, "condition": "CTCF",
                     "n_sineb2": 0, "urq_mean": urq_ctcf})

        # ── CTCF + varying SINE B2 counts ──
        for n_sb in SINEB2_COUNTS:
            seq = build_sequence_with_sineb2s(
                bg_seq, ctcf_row, sineb2_seqs, n_sineb2_total=n_sb
            )
            assert len(seq) == SEQ_LEN
            cmap = predict_map(model, seq, device)
            urq = upper_right_quarter_mean(cmap)
            print(f"    CTCF + {n_sb:2d} SINE B2s  URQ mean: {urq:.6f}")
            rows.append({**row_base, "condition": f"CTCF + {n_sb} SINE_B2",
                         "n_sineb2": n_sb, "urq_mean": urq})

    # ── Save all results ──
    results_df = pd.DataFrame(rows)
    # output_dir = os.path.dirname(os.path.abspath(__file__))
    out_tsv = os.path.join("urq_results_sineb2_counts.tsv")
    results_df.to_csv(out_tsv, sep="\t", index=False)

    n_conditions = 1 + len(SINEB2_COUNTS)  # CTCF only + each SINE B2 count
    print(f"\nResults saved to: {out_tsv}")
    print(f"Total rows: {len(results_df)} "
          f"(1 background + {len(ctcf_sites)} CTCFs x {n_conditions} conditions)")

    # ── Summary stats ──
    print("\n" + "=" * 70)
    print("SUMMARY (mean +/- std of URQ means across CTCFs)")
    print("=" * 70)
    print(f"  {'background':35s} | {bg_urq:.6f}")
    for n_sb in [0] + SINEB2_COUNTS:
        if n_sb == 0:
            cond = "CTCF"
        else:
            cond = f"CTCF + {n_sb} SINE_B2"
        vals = results_df.loc[results_df["condition"] == cond, "urq_mean"]
        print(f"  {cond:35s} | {vals.mean():.6f} +/- {vals.std():.6f}")


if __name__ == "__main__":
    main()