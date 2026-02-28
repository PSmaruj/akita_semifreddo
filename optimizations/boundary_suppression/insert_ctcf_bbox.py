#!/usr/bin/env python
"""
Insert a strong CTCF in the middle of a background sequence,
then flank it with varying numbers of B-box motifs (GAGTTCAATT)
and measure Akita predictions.

Conditions (for each of the top 100 CTCFs):
1. Background only (once)
2. Background + central CTCF
3. Background + central CTCF + 2 B-boxes  (1 per side)
4. Background + central CTCF + 4 B-boxes  (2 per side)
5. Background + central CTCF + 6 B-boxes  (3 per side)
6. Background + central CTCF + 8 B-boxes  (4 per side)
7. Background + central CTCF + 10 B-boxes (5 per side)

B-boxes are placed symmetrically around the CTCF with 10 bp spacing.

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
BBOX_SEQ = "GAGTTCAATT"   # B-box motif (10 bp)
BBOX_COUNTS = [2, 4, 6, 8, 10]  # total B-boxes to test (split evenly per side)
SPACING_BP = 100           # spacing between elements


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


# ── Extract CTCF sequence from genome ─────────────────────────────────────────
def get_ctcf_seq(row, fasta_path=GENOME_FASTA, force_strand="+"):
    """
    Extract CTCF sequence from genome.
    force_strand: always return sequence on this strand ('+' = forward / '>').
    """
    chrom = row["chrom"]
    start = int(row["start"])
    end = int(row["end"])
    strand = force_strand if force_strand is not None else row["strand"]
    return fetch_sequence(fasta_path, chrom, start, end, strand)


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


# ── Build sequence with CTCF + N B-boxes ─────────────────────────────────────
def build_sequence_with_bboxes(bg_seq, ctcf_row, n_bboxes_total):
    """
    Insert CTCF at center, then place n_bboxes_total B-boxes symmetrically
    around it (n_bboxes_total/2 per side) with SPACING_BP between edges.

    All elements in forward ('+' / '>') orientation.
    """
    center = SEQ_LEN // 2

    # Insert CTCF at center
    ctcf_seq = get_ctcf_seq(ctcf_row, force_strand="+")
    ctcf_len = len(ctcf_seq)
    seq = insert_element(bg_seq, ctcf_seq, center)

    if n_bboxes_total == 0:
        return seq

    # Place B-boxes symmetrically
    n_per_side = n_bboxes_total // 2
    half_ctcf = ctcf_len // 2
    bbox_len = len(BBOX_SEQ)

    for side in ['left', 'right']:
        for i in range(n_per_side):
            if i == 0:
                # First B-box: edge starts at CTCF edge + spacing
                offset = half_ctcf + SPACING_BP + bbox_len // 2
            else:
                # Subsequent: previous center + half bbox + spacing + half bbox
                offset = prev_offset + bbox_len // 2 + SPACING_BP + bbox_len // 2

            if side == 'left':
                pos = center - offset
            else:
                pos = center + offset

            seq = insert_element(seq, BBOX_SEQ, pos)
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
    print("CTCF + B-box Insertion Experiment")
    print(f"B-box motif: {BBOX_SEQ} ({len(BBOX_SEQ)} bp)")
    print(f"Spacing: {SPACING_BP} bp")
    print(f"B-box counts tested: {BBOX_COUNTS}")
    print("=" * 70)

    # Load data
    print("\n── Loading data ──")
    bg_seq = load_background()
    ctcf_sites = load_ctcf_sites()

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
        "n_bboxes": 0,
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

        # ── CTCF only (0 B-boxes) ──
        seq_ctcf = build_sequence_with_bboxes(bg_seq, ctcf_row, n_bboxes_total=0)
        assert len(seq_ctcf) == SEQ_LEN
        map_ctcf = predict_map(model, seq_ctcf, device)
        urq_ctcf = upper_right_quarter_mean(map_ctcf)
        print(f"    CTCF only          URQ mean: {urq_ctcf:.6f}")
        rows.append({**row_base, "condition": "CTCF",
                     "n_bboxes": 0, "urq_mean": urq_ctcf})

        # ── CTCF + varying B-box counts ──
        for n_bb in BBOX_COUNTS:
            seq = build_sequence_with_bboxes(bg_seq, ctcf_row, n_bboxes_total=n_bb)
            assert len(seq) == SEQ_LEN
            cmap = predict_map(model, seq, device)
            urq = upper_right_quarter_mean(cmap)
            print(f"    CTCF + {n_bb:2d} B-boxes  URQ mean: {urq:.6f}")
            rows.append({**row_base, "condition": f"CTCF + {n_bb} B-boxes",
                         "n_bboxes": n_bb, "urq_mean": urq})

    # ── Save all results ──
    results_df = pd.DataFrame(rows)
    out_tsv = os.path.join("urq_results_bbox.tsv")
    results_df.to_csv(out_tsv, sep="\t", index=False)

    n_conditions = 1 + len(BBOX_COUNTS)  # CTCF only + each bbox count
    # print(f"\nResults saved to: {out_tsv}")
    print(f"Total rows: {len(results_df)} "
          f"(1 background + {len(ctcf_sites)} CTCFs x {n_conditions} conditions)")

    # ── Summary stats ──
    print("\n" + "=" * 70)
    print("SUMMARY (mean +/- std of URQ means across CTCFs)")
    print("=" * 70)
    print(f"  {'background':35s} | {bg_urq:.6f}")
    for n_bb in [0] + BBOX_COUNTS:
        if n_bb == 0:
            cond = "CTCF"
        else:
            cond = f"CTCF + {n_bb} B-boxes"
        vals = results_df.loc[results_df["condition"] == cond, "urq_mean"]
        print(f"  {cond:35s} | {vals.mean():.6f} +/- {vals.std():.6f}")


if __name__ == "__main__":
    main()