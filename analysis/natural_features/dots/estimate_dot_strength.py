"""
estimate_dot_strength.py

For each convergent dot in the input table, runs Akita inference and estimates
its strength as the mean signal in a (2*DOT_HALF+1) × (2*DOT_HALF+1) patch
centred on the dot's bin coordinates.

Usage:
    python estimate_dot_strength.py
"""

import os
import sys

import numpy as np
import pandas as pd
import torch
from pyfaidx import Fasta
from torch.utils.data import DataLoader

sys.path.append(os.path.abspath("/home1/smaruj/pytorch_akita/"))
sys.path.insert(0, os.path.abspath("/home1/smaruj/akita_semifreddo/"))

from akita.model import SeqNN
from utils.data_utils import from_upper_triu_batch
from utils.dataset_utils import FeatureDataset

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

DOT_TSV    = (
    "/project2/fudenber_735/smaruj/sequence_design/ledidi_semifreddo_akita"
    "/analysis/natural_features/dots/mouse_convergent_dots.tsv"
)
OUTPUT_TSV = (
    "/project2/fudenber_735/smaruj/sequence_design/ledidi_semifreddo_akita"
    "/analysis/natural_features/dots/mouse_convergent_dots_strength.tsv"
)
FASTA_FILE = "/project2/fudenber_735/genomes/mm10/mm10.fa"
MODEL_CKPT = (
    "/home1/smaruj/pytorch_akita/models/finetuned/mouse/Hsieh2019_mESC"
    "/checkpoints/Akita_v2_mouse_Hsieh2019_mESC_model0_finetuned.pth"
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TARGET_LEN  = 1_310_720
BATCH_SIZE  = 16
DOT_HALF    = 7     # bins on each side → patch is (2*DOT_HALF+1)² = 15×15
PATCH_SIZE  = 2 * DOT_HALF + 1


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def compute_dot_strengths(
    loader: DataLoader,
    dot_df: pd.DataFrame,
    model: torch.nn.Module,
    device: torch.device,
) -> list[float]:
    """
    Run inference over all batches and return per-dot mean patch values.

    The patch is centred on (anchor1_center_bin, anchor2_center_bin) and has
    side length 2*DOT_HALF+1 bins in each dimension.
    """
    strengths: list[float] = []
    batch_start = 0

    with torch.no_grad():
        for batch in loader:
            preds  = model(batch.to(device)).cpu()
            maps   = from_upper_triu_batch(preds)   # (B, N, N)

            for i in range(maps.shape[0]):
                abs_i = batch_start + i
                r = int(dot_df["anchor1_center_bin"].iloc[abs_i])
                c = int(dot_df["anchor2_center_bin"].iloc[abs_i])

                patch = maps[i,
                             r - DOT_HALF : r + DOT_HALF + 1,
                             c - DOT_HALF : c + DOT_HALF + 1]

                if patch.shape == (PATCH_SIZE, PATCH_SIZE):
                    strengths.append(float(np.nanmean(patch)))
                else:
                    strengths.append(float("nan"))
                    print(f"  Warning: patch {abs_i} has shape {patch.shape} — "
                          f"strength set to NaN (dot at r={r}, c={c})")

            batch_start += maps.shape[0]

    return strengths


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    # -- Data ----------------------------------------------------------------
    dot_df = pd.read_csv(DOT_TSV, sep="\t")
    genome = Fasta(FASTA_FILE)

    dataset = FeatureDataset(dot_df, genome, TARGET_LEN)
    loader  = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    # -- Model ---------------------------------------------------------------
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = SeqNN()
    model.load_state_dict(torch.load(MODEL_CKPT, map_location=device))
    model.to(device).eval()

    # -- Inference -----------------------------------------------------------
    print(f"Estimating dot strength for {len(dot_df):,} dots...")
    dot_df["dot_strength"] = compute_dot_strengths(loader, dot_df, model, device)

    # -- Save ----------------------------------------------------------------
    dot_df.to_csv(OUTPUT_TSV, sep="\t", index=False)
    print(f"Saved results to:\n  {OUTPUT_TSV}")


if __name__ == "__main__":
    main()