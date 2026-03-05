"""
extract_dot_patches.py

For each convergent dot in the input table, runs Akita inference and extracts
a (2*PATCH_HALF + 1) × (2*PATCH_HALF + 1) patch from the predicted contact map,
centred on the dot's bin coordinates. Patches are saved as individual .npy files
for downstream construction of a data-driven dot mask.

Usage:
    python extract_dot_patches.py
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
from utils.dataset_utils import DotsDataset

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

DOT_TSV   = (
    "/project2/fudenber_735/smaruj/sequence_design/ledidi_semifreddo_akita"
    "/analysis/natural_features/dots/mouse_convergent_dots.tsv"
)
FASTA_FILE = "/project2/fudenber_735/genomes/mm10/mm10.fa"
MODEL_CKPT = (
    "/home1/smaruj/pytorch_akita/models/finetuned/mouse/Hsieh2019_mESC"
    "/checkpoints/Akita_v2_mouse_Hsieh2019_mESC_model0_finetuned.pth"
)
SAVE_DIR   = (
    "/project2/fudenber_735/smaruj/sequence_design/ledidi_semifreddo_akita"
    "/analysis/natural_features/dots/dot_patches"
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TARGET_LEN  = 1_310_720
BATCH_SIZE  = 16
PATCH_HALF  = 7          # bins on each side → patch is (2*PATCH_HALF+1)² = 15×15


# ---------------------------------------------------------------------------
# Patch extraction
# ---------------------------------------------------------------------------

def extract_patches(
    maps: np.ndarray,
    dot_df: pd.DataFrame,
    batch_start: int,
    patch_half: int,
    save_dir: str,
) -> int:
    """
    Extract and save square patches centred on each dot's bin coordinates.

    Args:
        maps:        Contact maps for the current batch, shape (B, N, N).
        dot_df:      Full dot DataFrame (indexed globally).
        batch_start: Global index of the first element in this batch.
        patch_half:  Half-width of the patch in bins.
        save_dir:    Directory to write .npy patch files.

    Returns:
        Number of patches skipped due to boundary issues.
    """
    patch_size = 2 * patch_half + 1
    n_skipped  = 0

    for i in range(maps.shape[0]):
        abs_i = batch_start + i
        row   = dot_df.iloc[abs_i]

        r, c = int(row["anchor1_center_bin"]), int(row["anchor2_center_bin"])
        patch = maps[i, r - patch_half : r + patch_half + 1,
                        c - patch_half : c + patch_half + 1]

        if patch.shape == (patch_size, patch_size):
            np.save(os.path.join(save_dir, f"patch_{abs_i}.npy"), patch)
        else:
            print(f"  Warning: patch {abs_i} has shape {patch.shape} — skipped "
                  f"(dot at r={r}, c={c} is too close to map edge)")
            n_skipped += 1

    return n_skipped


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    os.makedirs(SAVE_DIR, exist_ok=True)

    # -- Data ----------------------------------------------------------------
    dot_df = pd.read_csv(DOT_TSV, sep="\t")
    genome = Fasta(FASTA_FILE)

    dataset = DotsDataset(dot_df, genome, TARGET_LEN)
    loader  = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    # -- Model ---------------------------------------------------------------
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = SeqNN()
    model.load_state_dict(torch.load(MODEL_CKPT, map_location=device))
    model.to(device).eval()

    # -- Inference & patch extraction ----------------------------------------
    print(f"Extracting {2*PATCH_HALF+1}×{2*PATCH_HALF+1} patches for {len(dot_df):,} dots...")
    batch_start = 0
    total_skipped = 0

    with torch.no_grad():
        for batch in loader:
            preds = model(batch.to(device)).cpu()
            maps  = from_upper_triu_batch(preds)   # (B, N, N)

            total_skipped += extract_patches(maps, dot_df, batch_start, PATCH_HALF, SAVE_DIR)
            batch_start   += maps.shape[0]

    n_saved = len(dot_df) - total_skipped
    print(f"Done. Saved {n_saved:,} patches to:\n  {SAVE_DIR}")
    if total_skipped:
        print(f"  ({total_skipped} patches skipped due to map-edge proximity)")


if __name__ == "__main__":
    main()