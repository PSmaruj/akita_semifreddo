"""
estimate_flame_strength.py

Estimate the predicted strength of chromatin stripes ("flames") by running
sequences through the Akita model and summarising contact enrichment within
each stripe's predicted region of the contact map.

Stripe positions must already have been annotated with:
  - triangular_half  ("upper" / "lower")
  - x_bins / y_bins  (stripe dimensions in Akita bins)
  - window_start / window_end  (1-Mb Akita input window, in bp)

See select_stripes_stripenn.py for how to generate that input file.

python estimate_flame_strength.py \
    --stripes /project2/fudenber_735/smaruj/sequence_design/ledidi_semifreddo_akita/analysis/natural_features/flames/mouse_selected_stripes.tsv \
    --out /project2/fudenber_735/smaruj/sequence_design/ledidi_semifreddo_akita/analysis/natural_features/flames/mouse_selected_stripes_strength.tsv \
    --fasta /project2/fudenber_735/genomes/mm10/mm10.fa \
    --model-weights /home1/smaruj/akita_pytorch/models/finetuned/mouse/Hsieh2019_mESC/checkpoints/Akita_v2_mouse_Hsieh2019_mESC_model0_finetuned.pth \
    --batch-size 16 --device cuda:0
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch
from pyfaidx import Fasta
from torch.utils.data import DataLoader

# ---------------------------------------------------------------------------
# Project-local imports – adjust paths via CLI or environment if needed.
# ---------------------------------------------------------------------------
sys.path.append(os.path.abspath("/home1/smaruj/akita_pytorch/"))
sys.path.insert(0, os.path.abspath("/home1/smaruj/akita_semifreddo/"))

from akita.model import SeqNN
from utils.data_utils import from_upper_triu_batch
from utils.dataset_utils import FeatureDataset

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
AKITA_RES_BP    = 2048   # Akita prediction resolution (bp per bin)
MAP_SIZE        = 512    # Predicted contact map is MAP_SIZE × MAP_SIZE
MAP_MIDBIN      = MAP_SIZE // 2  # Centre bin index (256)
TARGET_LEN = 1_310_720

# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------

def extract_stripe_region(
    contact_map: np.ndarray,
    x_bins: int,
    y_bins: int,
    triangular_half: str,
) -> np.ndarray:
    """Return the sub-array of a contact map that falls within a stripe's bounds.

    Stripes are anchored at the map centre (MAP_MIDBIN). The stripe extends
    MAP_MIDBIN → MAP_MIDBIN + x_bins along the x-axis (columns).  Its extent
    along the y-axis depends on which triangular half of the map it occupies:

      "upper"  → stripe runs from (MAP_MIDBIN - y_bins) up to MAP_MIDBIN
                  (i.e. above the diagonal)
      "lower"  → stripe runs from MAP_MIDBIN down to (MAP_MIDBIN + y_bins)
                  (i.e. below the diagonal)

    Parameters
    ----------
    contact_map : np.ndarray
        2-D predicted contact map for a single sample, shape (MAP_SIZE, MAP_SIZE).
    x_bins : int
        Stripe width in Akita bins.
    y_bins : int
        Stripe length in Akita bins.
    triangular_half : {"upper", "lower"}
        Which triangular half of the map the stripe occupies.

    Returns
    -------
    np.ndarray
        Extracted sub-array, or an empty array if the stripe falls outside
        the valid map region.
    """
    x_end = int(np.clip(MAP_MIDBIN + x_bins, 0, MAP_SIZE - 1))

    if triangular_half == "upper":
        y_start = int(np.clip(MAP_MIDBIN - y_bins, 0, MAP_SIZE - 1))
        if x_end > MAP_MIDBIN > y_start:
            return contact_map[y_start:MAP_MIDBIN, MAP_MIDBIN:x_end]
    else:  # "lower"
        y_start = int(np.clip(MAP_MIDBIN + y_bins, 0, MAP_SIZE - 1))
        if x_end > MAP_MIDBIN and y_start < MAP_SIZE:
            return contact_map[MAP_MIDBIN:x_end, MAP_MIDBIN:y_start]

    return np.array([])


def compute_stripe_strengths(
    flame_df: pd.DataFrame,
    genome: Fasta,
    model: torch.nn.Module,
    device: torch.device,
    batch_size: int = 16,
) -> pd.DataFrame:
    """Run Akita on every stripe window and record per-stripe contact enrichment.

    Three summary statistics are computed for each stripe region:
      - median contact value
      - 75th-percentile contact value
      - mean contact value

    Parameters
    ----------
    flame_df : pd.DataFrame
        Stripe table with geometry columns (triangular_half, x_bins, y_bins).
    genome : Fasta
        Genome FASTA handle (pyfaidx).
    model : torch.nn.Module
        Loaded, eval-mode Akita model.
    device : torch.device
        Device on which to run inference.
    batch_size : int
        Number of sequences per forward pass.

    Returns
    -------
    pd.DataFrame
        Copy of ``flame_df`` with three new columns:
        flame_strength_median, flame_strength_q3, flame_strength_mean.
    """
    dataset = FeatureDataset(flame_df, genome, TARGET_LEN)
    loader  = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    medians, q3s, means = [], [], []
    sample_idx = 0

    model.eval()
    with torch.no_grad():
        for batch in loader:
            preds    = model(batch.to(device)).cpu()
            maps     = from_upper_triu_batch(preds)           # (B, MAP_SIZE, MAP_SIZE)
            B        = maps.shape[0]

            for i in range(B):
                row = flame_df.iloc[sample_idx + i]
                region = extract_stripe_region(
                    contact_map=maps[i],
                    x_bins=int(row["x_bins"]),
                    y_bins=int(row["y_bins"]),
                    triangular_half=row["triangular_half"],
                )

                if region.size > 0:
                    medians.append(float(np.nanmedian(region)))
                    q3s.append(float(np.nanpercentile(region, 75)))
                    means.append(float(np.nanmean(region)))
                else:
                    medians.append(np.nan)
                    q3s.append(np.nan)
                    means.append(np.nan)

            sample_idx += B

    out = flame_df.copy()
    out["flame_strength_median"] = medians
    out["flame_strength_q3"]     = q3s
    out["flame_strength_mean"]   = means
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Estimate predicted stripe strength using the Akita model."
    )
    p.add_argument("--stripes", required=True,
                   help="Input TSV of selected stripes (from select_stripes_stripenn.py).")
    p.add_argument("--out", required=True,
                   help="Output TSV path.")
    p.add_argument("--fasta", required=True,
                   help="Path to genome FASTA (e.g. mm10.fa).")
    p.add_argument("--model-weights", required=True,
                   help="Path to Akita model weights (.pth).")
    p.add_argument("--batch-size", type=int, default=16,
                   help="Batch size for inference (default: 16).")
    p.add_argument("--device", default=None,
                   help="PyTorch device string, e.g. 'cuda:0' or 'cpu'. "
                        "Auto-detected if not set.")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    print(f"Loading stripes: {args.stripes}")
    flame_df = pd.read_csv(args.stripes, sep="\t")
    flame_df = flame_df.rename(columns={"chr": "chrom"})
    flame_df["y_bins"] = flame_df["length"] // AKITA_RES_BP
    flame_df["x_bins"] = flame_df["width"]  // AKITA_RES_BP
    print(f"  {len(flame_df):,} stripes.")

    print(f"Loading genome: {args.fasta}")
    genome = Fasta(args.fasta)

    device = torch.device(
        args.device if args.device else ("cuda:0" if torch.cuda.is_available() else "cpu")
    )
    print(f"Using device: {device}")

    print(f"Loading model weights: {args.model_weights}")
    model = SeqNN()
    model.load_state_dict(torch.load(args.model_weights, map_location=device))
    model.to(device)

    print("Running inference...")
    result_df = compute_stripe_strengths(
        flame_df=flame_df,
        genome=genome,
        model=model,
        device=device,
        batch_size=args.batch_size,
    )

    result_df.to_csv(args.out, sep="\t", index=False)
    print(f"Saved to: {args.out}")


if __name__ == "__main__":
    main()