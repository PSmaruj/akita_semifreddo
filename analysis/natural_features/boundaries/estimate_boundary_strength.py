"""
estimate_boundary_strength.py

For each natural boundary in the input table, centers it in the Akita prediction
window and estimates its insulation strength as the mean signal in the upper-right
quarter (URQ) of the predicted contact map.

Usage:
    python estimate_boundary_strength.py
"""

import os
import sys

import numpy as np
import pandas as pd
import torch
from pyfaidx import Fasta
from torch.utils.data import DataLoader, Dataset

sys.path.append(os.path.abspath("/home1/smaruj/pytorch_akita/"))
sys.path.insert(0, os.path.abspath("/home1/smaruj/ledidi_akita/"))

from akita.model import SeqNN
from utils.data_utils import from_upper_triu_batch, one_hot_encode_sequence

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

BOUNDARIES_TSV = (
    "/project2/fudenber_735/smaruj/sequence_design/ledidi_semifreddo_akita"
    "/analysis/natural_features/boundaries/mouse_natural_boundaries.tsv"
)
OUTPUT_TSV = (
    "/project2/fudenber_735/smaruj/sequence_design/ledidi_semifreddo_akita"
    "/analysis/natural_features/boundaries/mouse_natural_boundaries_strength.tsv"
)
FASTA_FILE = "/project2/fudenber_735/genomes/mm10/mm10.fa"
MODEL_CKPT = (
    "/home1/smaruj/pytorch_akita/models/finetuned/mouse/Hsieh2019_mESC"
    "/checkpoints/Akita_v2_mouse_Hsieh2019_mESC_model0_finetuned.pth"
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TARGET_LEN = 1_310_720
BATCH_SIZE = 16

# Upper-right quarter (URQ) slice of the 512-bin contact map.
# Rows 0–249  → upstream half  (excluding diagonal buffer)
# Cols 260–511 → downstream half (excluding diagonal buffer)
URQ_ROW_SLICE = slice(0, 250)
URQ_COL_SLICE = slice(260, 512)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class BoundaryCenteredDataset(Dataset):
    """
    Returns one-hot encoded sequences centered on each boundary window.

    Args:
        coord_df:     DataFrame with columns [chrom, window_start, window_end].
        genome_fasta: pyfaidx.Fasta object for the reference genome.
    """

    def __init__(self, coord_df: pd.DataFrame, genome_fasta: Fasta) -> None:
        self.coords = coord_df
        self.genome = genome_fasta

    def __len__(self) -> int:
        return len(self.coords)

    def __getitem__(self, idx: int) -> torch.Tensor:
        row = self.coords.iloc[idx]
        chrom = row["chrom"]
        start = row["window_start"]
        end   = row["window_end"]

        seq = self.genome[chrom][start:end].seq.upper()

        # Guard against off-chromosome windows: truncate or N-pad.
        if len(seq) != TARGET_LEN:
            seq = seq[:TARGET_LEN].ljust(TARGET_LEN, "N")

        one_hot = one_hot_encode_sequence(seq)  # (1, 4, TARGET_LEN) or (4, TARGET_LEN)
        one_hot = np.squeeze(one_hot)           # → (4, TARGET_LEN)
        return torch.from_numpy(one_hot.copy())


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------

def predict_urq_strengths(
    loader: DataLoader,
    model: torch.nn.Module,
    device: torch.device,
) -> list[float]:
    """
    Run inference over all batches and return per-sequence URQ mean values.

    The URQ (upper-right quarter) of the contact map captures the insulation
    signal across the boundary: high values indicate a strong boundary.
    """
    urq_means: list[float] = []

    with torch.no_grad():
        for batch in loader:
            preds = model(batch.to(device)).cpu()
            maps  = from_upper_triu_batch(preds)                          # (B, 512, 512)
            urq   = maps[:, URQ_ROW_SLICE, URQ_COL_SLICE]                 # (B, 250, 252)
            urq_means.extend(np.nanmean(urq, axis=(1, 2)).tolist())

    return urq_means


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    # -- Data ----------------------------------------------------------------
    boundaries_df = pd.read_csv(BOUNDARIES_TSV, sep="\t")
    genome        = Fasta(FASTA_FILE)

    dataset = BoundaryCenteredDataset(boundaries_df, genome)
    loader  = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    # -- Model ---------------------------------------------------------------
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = SeqNN()
    model.load_state_dict(torch.load(MODEL_CKPT, map_location=device))
    model.to(device).eval()

    # -- Inference -----------------------------------------------------------
    print(f"Running inference on {len(dataset)} boundaries...")
    urq_means = predict_urq_strengths(loader, model, device)

    # -- Save ----------------------------------------------------------------
    boundaries_df["URQ_mean"] = urq_means
    boundaries_df.to_csv(OUTPUT_TSV, sep="\t", index=False)
    print(f"Saved results to:\n  {OUTPUT_TSV}")


if __name__ == "__main__":
    main()