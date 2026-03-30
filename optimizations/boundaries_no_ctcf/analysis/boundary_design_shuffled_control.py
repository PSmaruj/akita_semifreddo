"""
boundary_design_shuffled_control.py

Compute insulation scores for dinucleotide-shuffled versions of the optimised
central bin, as a null control for the boundary optimisation results.

For each successfully optimised sequence, the optimised 2,048 bp central bin
is replaced by a dinucleotide-preserving shuffle (via seqpro), and the full
Akita v2 model is run to obtain a predicted insulation score. All other parts
of the sequence are kept identical to the optimised sequence.

Usage:
    python boundary_design_shuffled_control.py \
        --fold 7 \
        --run_name results \
        --boundary_strength -0.2
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
import seqpro as sp
import torch
from torch.utils.data import DataLoader, Dataset

sys.path.append(os.path.abspath("/home1/smaruj/pytorch_akita/"))
sys.path.insert(0, os.path.abspath("/home1/smaruj/ledidi_akita/"))

from utils.data_utils import from_upper_triu_batch
from utils.model_utils import load_model
from utils.optimization_utils import strength_tag
from utils.scores_utils import insulation_score

# ── Fixed paths ───────────────────────────────────────────────────────────────
_PROJ = "/project2/fudenber_735/smaruj/sequence_design/ledidi_semifreddo_akita"

MODEL_CKPT        = (
    "/home1/smaruj/pytorch_akita/models/finetuned/mouse/Hsieh2019_mESC"
    "/checkpoints/Akita_v2_mouse_Hsieh2019_mESC_model0_finetuned.pth"
)
SEQ_BASE_DIR      = f"{_PROJ}/analysis/flat_regions"
RESULTS_BASE_DIR  = f"{_PROJ}/optimizations/boundaries_no_ctcf"

# ── Architecture constants ────────────────────────────────────────────────────
CENTER_BIN_MAP = 256
CROPPING       = 64
BIN_SIZE       = 2048
EDIT_START     = (CENTER_BIN_MAP + CROPPING) * BIN_SIZE
EDIT_END       = EDIT_START + BIN_SIZE

# ── URQ slice ─────────────────────────────────────────────────────────────────
URQ_ROW_SLICE = slice(0, 250)
URQ_COL_SLICE = slice(260, 512)

# ── Shuffle constants ─────────────────────────────────────────────────────────
K           = 2                                          # dinucleotide shuffle
BASES       = np.array(["A", "C", "G", "T"])
BASE_TO_IDX = {"A": 0, "C": 1, "G": 2, "T": 3}


# ── OHE ↔ sequence helpers ────────────────────────────────────────────────────

def ohe_to_bytes(ohe: np.ndarray) -> bytes:
    """Convert (4, L) OHE array to a byte string."""
    indices = np.argmax(ohe, axis=0)           # (L,)
    return "".join(BASES[indices]).encode()


def bytes_to_ohe(seq: bytes, length: int) -> np.ndarray:
    """Convert a byte string back to a (4, L) OHE float32 array."""
    ohe = np.zeros((4, length), dtype=np.float32)
    for i, base in enumerate(seq.decode()):
        ohe[BASE_TO_IDX[base], i] = 1.0
    return ohe


# ── Dataset ───────────────────────────────────────────────────────────────────

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
        shuffled_ohe   = bytes_to_ohe(shuffled_bytes, BIN_SIZE)  # (4, BIN_SIZE)
        shuffled = torch.from_numpy(shuffled_ohe).unsqueeze(0)   # (1, 4, BIN_SIZE)

        edited = X.clone()
        edited[:, :, self.edit_start:self.edit_end] = shuffled
        return edited.squeeze(0)


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Shuffled dinucleotide control for boundary optimisation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--fold",              type=int,   required=True)
    p.add_argument("--run_name",          type=str,   required=True)
    p.add_argument("--boundary_strength", type=float, required=True)
    p.add_argument("--batch_size",        type=int,   default=4)
    p.add_argument("--results_base_dir",  default=RESULTS_BASE_DIR)

    return p.parse_args()


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args   = parse_args()
    fold   = args.fold
    tag    = strength_tag(args.boundary_strength)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    run_dir  = os.path.join(args.results_base_dir, args.run_name)
    fold_dir = os.path.join(run_dir, f"fold{fold}")

    print(f"Device  : {device}")
    print(f"Run dir : {run_dir}")
    print(f"Fold    : {fold}")

    # ── Load results table ────────────────────────────────────────────────────
    results_tsv = os.path.join(
        run_dir,
        f"fold{fold}_selected_genomic_windows_centered_chrom_states_results.tsv",
    )
    df = pd.read_csv(results_tsv, sep="\t")
    print(f"Loaded {len(df)} windows")

    # ── Model ─────────────────────────────────────────────────────────────────
    model = load_model(MODEL_CKPT, device)

    # ── Dataset & loader ──────────────────────────────────────────────────────
    seq_path = f"{SEQ_BASE_DIR}/mouse_sequences/fold{fold}/"

    shuffled_dataset = ShuffledCentralInsertionDataset(
        coord_df   = df,
        seq_path   = seq_path,
        slice_path = fold_dir + "/",
        edit_start = EDIT_START,
        edit_end   = EDIT_END,
    )
    shuffled_loader = DataLoader(shuffled_dataset, batch_size=args.batch_size, shuffle=False)

    # ── Prediction loop ───────────────────────────────────────────────────────
    urq_shuffled = []

    with torch.no_grad():
        for batch in shuffled_loader:
            batch      = batch.to(device)
            maps       = from_upper_triu_batch(model(batch).cpu())
            urq_shuffled.extend(insulation_score(maps, URQ_ROW_SLICE, URQ_COL_SLICE))

    # ── Save ──────────────────────────────────────────────────────────────────
    df["insul_score_shuffled"] = urq_shuffled

    out_path = os.path.join(
        run_dir,
        f"fold{fold}_selected_genomic_windows_centered_chrom_states_shuffled_control.tsv",
    )
    df.to_csv(out_path, sep="\t", index=False)
    print(f"Saved → {out_path}")


if __name__ == "__main__":
    main()