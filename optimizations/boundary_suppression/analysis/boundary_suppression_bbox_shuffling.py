"""
boundary_suppression_bbox_shuffling.py

For each optimized (suppressed) boundary sequence, detect B-box-like motifs
within the edited central bin and shuffle them. Re-run Akita to check whether
B-box disruption affects the suppression prediction.

Usage:
    python boundary_suppression_bbox_shuffling.py \
        --fold 7 \
        --run_name results
"""

import argparse
import os
import random
import sys

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

sys.path.append(os.path.abspath("/home1/smaruj/pytorch_akita/"))
sys.path.insert(0, os.path.abspath("/home1/smaruj/ledidi_akita/"))

from utils.data_utils import from_upper_triu_batch, gc_content
from utils.model_utils import load_model
from utils.scores_utils import compute_insulation_scores

# ── Fixed paths ───────────────────────────────────────────────────────────────
_PROJ = "/project2/fudenber_735/smaruj/sequence_design/ledidi_semifreddo_akita"

MODEL_CKPT       = (
    "/home1/smaruj/pytorch_akita/models/finetuned/mouse/Hsieh2019_mESC"
    "/checkpoints/Akita_v2_mouse_Hsieh2019_mESC_model0_finetuned.pth"
)
SEQ_BASE_DIR     = f"{_PROJ}/optimizations/boundary_suppression/initial_sequences"
RESULTS_BASE_DIR = f"{_PROJ}/optimizations/boundary_suppression"

# ── Architecture constants ────────────────────────────────────────────────────
CENTER_BIN_MAP = 256
CROPPING       = 64
BIN_SIZE       = 2048
EDIT_START     = (CENTER_BIN_MAP + CROPPING) * BIN_SIZE
EDIT_END       = EDIT_START + BIN_SIZE

# ── URQ slice ─────────────────────────────────────────────────────────────────
URQ_ROW_SLICE = slice(0, 250)
URQ_COL_SLICE = slice(260, 512)

# ── B-box consensus: RGTTCRANTCC ──────────────────────────────────────────────
BBOX_CONSENSUS = [
    {"A", "G"},              # R
    {"G"},                   # G
    {"T"},                   # T
    {"T"},                   # T
    {"C"},                   # C
    {"A", "G"},              # R
    {"A", "C", "G", "T"},   # N
    {"A", "G"},              # R
    {"T"},                   # T
    {"C"},                   # C
    {"C"},                   # C
]
BBOX_LEN = len(BBOX_CONSENSUS)  # 11

NT_IDX = {0: "A", 1: "C", 2: "G", 3: "T"}
IDX_NT = {"A": 0, "C": 1, "G": 2, "T": 3}


# ── B-box helpers ─────────────────────────────────────────────────────────────

def count_mismatches(seq, consensus=BBOX_CONSENSUS):
    """Count mismatches between a sequence string and the B-box consensus."""
    mismatches = 0
    for i, allowed in enumerate(consensus):
        if seq[i].upper() not in allowed:
            mismatches += 1
    return mismatches


def find_bbox_hits(seq_str, max_mismatches=3):
    """
    Scan a DNA string for B-box-like motifs (fuzzy matching).

    Returns:
        List of merged (start, end) tuples (0-indexed, end exclusive).
    """
    hits = []
    for i in range(len(seq_str) - BBOX_LEN + 1):
        if count_mismatches(seq_str[i:i + BBOX_LEN]) <= max_mismatches:
            hits.append((i, i + BBOX_LEN))
    if not hits:
        return hits
    merged = [hits[0]]
    for start, end in hits[1:]:
        if start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def shuffle_regions(seq_str, regions):
    """Randomly shuffle nucleotides within specified regions of a DNA string."""
    seq_list = list(seq_str)
    for start, end in regions:
        subseq = seq_list[start:end]
        random.shuffle(subseq)
        seq_list[start:end] = subseq
    return "".join(seq_list)


def onehot_to_seq(tensor_2d):
    """Convert a (4, L) one-hot tensor to a DNA string."""
    indices = tensor_2d.argmax(dim=0).cpu().numpy()
    return "".join(NT_IDX[i] for i in indices)


def seq_to_onehot(seq_str, device):
    """Convert a DNA string to a (4, L) one-hot float tensor."""
    t = torch.zeros(4, len(seq_str), dtype=torch.float32, device=device)
    for j, nt in enumerate(seq_str):
        t[IDX_NT[nt], j] = 1.0
    return t


# ── Dataset ───────────────────────────────────────────────────────────────────

class BBoxShuffledSuppressionDataset(Dataset):
    """
    Loads optimized (suppressed) boundary sequences and shuffles any B-box
    motifs found within the edited central bin before returning to the model.

    Args:
        coord_df:        DataFrame with chrom, centered_start, centered_end, fold columns.
        seq_path:        Directory containing original full-context sequence tensors.
        edited_seq_path: Directory containing per-window edited central-bin tensors
                         (as produced by the suppression optimiser).
        edit_start:      Start bp of the editable region in the full sequence.
        edit_end:        End bp of the editable region in the full sequence.
        max_mismatches:  Maximum mismatches to B-box consensus (default 2).
        seed:            Base random seed (per-sample offset added for reproducibility).
    """

    def __init__(self, coord_df, seq_path, edited_seq_path,
                 edit_start, edit_end,
                 max_mismatches=2, seed=None):
        self.coords          = coord_df
        self.seq_path        = seq_path
        self.edited_seq_path = edited_seq_path
        self.edit_start      = edit_start
        self.edit_end        = edit_end
        self.max_mismatches  = max_mismatches
        self.seed            = seed

    def __len__(self):
        return len(self.coords)

    def __getitem__(self, idx):
        row   = self.coords.iloc[idx]
        chrom = row["chrom"]
        start = row["centered_start"]
        end   = row["centered_end"]
        fold  = row["fold"]

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Full-context background sequence
        X = torch.load(
            f"{self.seq_path}fold{fold}/{chrom}_{start}_{end}_X.pt",
            weights_only=True, map_location=device,
        )

        # Optimized central-bin slice (the suppressed sequence)
        edited_slice = torch.load(
            f"{self.edited_seq_path}/{chrom}_{start}_{end}_gen_seq.pt",
            weights_only=True, map_location=device,
        )

        # Normalise to (4, L)
        if edited_slice.dim() == 3:
            edited_slice = edited_slice.squeeze(0)

        # ── B-box detection & shuffling ───────────────────────────────────
        seq_str   = onehot_to_seq(edited_slice)
        bbox_hits = find_bbox_hits(seq_str, max_mismatches=self.max_mismatches)

        if bbox_hits:
            if self.seed is not None:
                random.seed(self.seed + idx)
            seq_str      = shuffle_regions(seq_str, bbox_hits)
            edited_slice = seq_to_onehot(seq_str, device=device)

        # ── Insert shuffled slice into full-context sequence ──────────────
        editedX = X.clone()
        editedX[:, :, self.edit_start:self.edit_end] = edited_slice
        editedX = editedX.squeeze(0)

        return editedX, len(bbox_hits)   # also return hit count for bookkeeping


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="B-box shuffling analysis for boundary suppression sequences",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--fold",             type=int, required=True)
    p.add_argument("--run_name",         type=str, required=True,
                   help="Results subdirectory, e.g. 'results'")
    p.add_argument("--max_mismatches",   type=int, default=3)
    p.add_argument("--seed",             type=int, default=42)
    p.add_argument("--batch_size",       type=int, default=4)
    p.add_argument("--results_base_dir", default=RESULTS_BASE_DIR)
    return p.parse_args()


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args   = parse_args()
    fold   = args.fold
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    run_dir  = os.path.join(args.results_base_dir, args.run_name)
    fold_dir = os.path.join(run_dir, f"fold{fold}")

    print(f"Device          : {device}")
    print(f"Run dir         : {run_dir}")
    print(f"Fold dir        : {fold_dir}")
    print(f"Max mismatches  : {args.max_mismatches}")
    print(f"Seed            : {args.seed}")

    # ── Load suppression results table ────────────────────────────────────────
    df = pd.read_csv(
        os.path.join(run_dir, f"fold{fold}_suppression_results.tsv"),
        sep="\t",
    )
    print(f"Loaded {len(df)} windows")

    # ── Model ─────────────────────────────────────────────────────────────────
    model = load_model(MODEL_CKPT, device)

    # ── Dataset & loader ──────────────────────────────────────────────────────
    seq_path = f"{SEQ_BASE_DIR}/"

    dataset = BBoxShuffledSuppressionDataset(
        coord_df        = df,
        seq_path        = seq_path,
        edited_seq_path = fold_dir,
        edit_start      = EDIT_START,
        edit_end        = EDIT_END,
        max_mismatches  = args.max_mismatches,
        seed            = args.seed,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    # ── Prediction loop ───────────────────────────────────────────────────────
    urq_bbox_shuffled = []
    bbox_hit_counts   = []

    with torch.no_grad():
        for seqs, hit_counts in loader:
            seqs = seqs.to(device)
            maps = from_upper_triu_batch(model(seqs).cpu())
            urq_bbox_shuffled.extend(compute_insulation_scores(maps, URQ_ROW_SLICE, URQ_COL_SLICE))
            bbox_hit_counts.extend(hit_counts.tolist())

    # ── Assemble & save results ───────────────────────────────────────────────
    out_df = df[["chrom", "centered_start", "centered_end", "fold",
                 "insul_score_orig", "insul_score_edited"]].copy()
    out_df["insul_score_bbox_shuffled"] = urq_bbox_shuffled
    out_df["bbox_hit_regions"]          = bbox_hit_counts
    out_df["bbox_found"]                = out_df["bbox_hit_regions"] > 0

    n_with_bbox = out_df["bbox_found"].sum()
    print(f"\nWindows with ≥1 B-box hit : {n_with_bbox} / {len(out_df)} "
          f"({100 * n_with_bbox / len(out_df):.1f}%)")

    out_path = os.path.join(run_dir, f"fold{fold}_bbox_shuffled_results.tsv")
    out_df.to_csv(out_path, sep="\t", index=False)
    print(f"Saved → {out_path}")


if __name__ == "__main__":
    main()