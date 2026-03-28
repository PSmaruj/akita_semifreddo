"""
cross_celltype_boundary_design_analysis.py

Analyse cross-cell-type boundary optimisation results: predicted insulation
scores (URQ) for original and optimised sequences across H1hESC and HFF models.

For each cell type, insulation scores are computed using:
  - original sequences : average over models 0–3 (4 models)
  - optimised sequences: models 0,1 (used in optimisation) and models 2,3 (held-out)

Results are saved as a TSV enriched with per-condition insulation score columns.

Usage:
    python cross_celltype_boundary_design_analysis.py \
        --fold 7 \
        --run_name results/H1hESC_strong_neg0p5_HFF_weak_neg0p2 \
        --strong_cell_type H1hESC \
        --strong_strength -0.5 \
        --weak_strength -0.2
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

sys.path.append(os.path.abspath("/home1/smaruj/pytorch_akita/"))
sys.path.insert(0, os.path.abspath("/home1/smaruj/ledidi_akita/"))

from akita.model import SeqNN
from utils.dataset_utils import CentralInsertionDataset, SequenceDataset
from utils.data_utils import from_upper_triu_batch
from utils.optimization_utils import strength_tag
from utils.scores_utils import insulation_score

# ── Fixed paths ───────────────────────────────────────────────────────────────
_PROJ = "/project2/fudenber_735/smaruj/sequence_design/ledidi_semifreddo_akita"

MODEL_PATH_PATTERN = (
    "/home1/smaruj/pytorch_akita/models/finetuned/human/"
    "Krietenstein2019_{cell_type}/checkpoints/"
    "Akita_v2_human_Krietenstein2019_{cell_type}_model{model_idx}_finetuned.pth"
)
SEQ_BASE_DIR      = f"{_PROJ}/analysis/flat_regions"
RESULTS_BASE_DIR  = f"{_PROJ}/optimizations/cross_celltype_boundaries"
FLAT_REGIONS_BASE = f"{_PROJ}/analysis/flat_regions/human_flat_regions_tsv"

# ── Architecture constants ────────────────────────────────────────────────────
CENTER_BIN_MAP = 256
CROPPING       = 64
BIN_SIZE       = 2048
EDIT_START     = (CENTER_BIN_MAP + CROPPING) * BIN_SIZE
EDIT_END       = EDIT_START + BIN_SIZE

# ── Model indices ─────────────────────────────────────────────────────────────
OPT_MODEL_INDICES  = [0, 1]    # used in optimisation
HELD_MODEL_INDICES = [2, 3]    # held-out validation
ALL_MODEL_INDICES  = OPT_MODEL_INDICES + HELD_MODEL_INDICES

CELL_TYPES = ["H1hESC", "HFF"]

# ── URQ slice ─────────────────────────────────────────────────────────────────
URQ_ROW_SLICE = slice(0, 250)
URQ_COL_SLICE = slice(260, 512)


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Cross-cell-type boundary optimisation analysis",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--fold",             type=int,   required=True)
    p.add_argument("--run_name",         type=str,   required=True)
    p.add_argument("--strong_cell_type", choices=CELL_TYPES, required=True)
    p.add_argument("--strong_strength",  type=float, default=-0.5)
    p.add_argument("--weak_strength",    type=float, default=-0.2)
    p.add_argument("--batch_size",       type=int,   default=4)
    p.add_argument("--results_base_dir", default=RESULTS_BASE_DIR)
    p.add_argument("--flat_regions_base", default=FLAT_REGIONS_BASE)
    return p.parse_args()


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_models(cell_type: str, model_indices: list,
                device: torch.device) -> list:
    """Load a list of SeqNN models for a given cell type."""
    models = []
    for idx in model_indices:
        m = SeqNN()
        m.load_state_dict(torch.load(
            MODEL_PATH_PATTERN.format(cell_type=cell_type, model_idx=idx),
            map_location=device, weights_only=True,
        ))
        m.to(device).eval()
        models.append(m)
        print(f"  {cell_type} model {idx} loaded")
    return models


def predict_insulation(
    loader: DataLoader,
    models: list,
    device: torch.device,
    average: bool = False,
) -> list:
    """Run models on all batches and return per-sequence insulation scores.

    Parameters
    ----------
    loader  : DataLoader yielding sequence tensors (B, 4, L)
    models  : list of SeqNN instances
    device  : torch.device
    average : if True, average scores across all models;
              if False, return a list of per-model score lists

    Returns
    -------
    If average=True  : list of floats, length = n_sequences
    If average=False : list of lists, shape (n_models, n_sequences)
    """
    per_model = [[] for _ in models]

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            for i, m in enumerate(models):
                maps = from_upper_triu_batch(m(batch).cpu())
                per_model[i].extend(insulation_score(maps, URQ_ROW_SLICE, URQ_COL_SLICE))

    if average:
        return np.mean(per_model, axis=0).tolist()
    return per_model


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args   = parse_args()
    fold   = args.fold
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    weak_cell_type = [ct for ct in CELL_TYPES if ct != args.strong_cell_type][0]
    strong_tag     = strength_tag(args.strong_strength)
    weak_tag       = strength_tag(args.weak_strength)
    target_tag     = (
        f"{args.strong_cell_type}_strong_{strong_tag}_"
        f"{weak_cell_type}_weak_{weak_tag}"
    )

    run_dir  = os.path.join(args.results_base_dir, args.run_name)
    fold_dir = os.path.join(run_dir, f"fold{fold}")

    print(f"Device   : {device}")
    print(f"Run dir  : {run_dir}")
    print(f"Fold dir : {fold_dir}")
    print(f"Strong   : {args.strong_cell_type} ({args.strong_strength})")
    print(f"Weak     : {weak_cell_type} ({args.weak_strength})")

    # ── Load opt metadata table ───────────────────────────────────────────────
    tsv_path = os.path.join(
        run_dir,
        f"fold{fold}_selected_genomic_windows_centered_opt.tsv",
    )
    df = pd.read_csv(tsv_path, sep="\t")
    print(f"Loaded {len(df)} windows")

    # ── Datasets ──────────────────────────────────────────────────────────────
    seq_dir = f"{SEQ_BASE_DIR}/human_sequences/fold{fold}/"

    orig_dataset   = SequenceDataset(
        df, seq_dir, "chrom", "centered_start", "centered_end", "X"
    )
    edited_dataset = CentralInsertionDataset(
        df, seq_dir, fold_dir + "/", EDIT_START, EDIT_END
    )

    orig_loader   = DataLoader(orig_dataset,   batch_size=args.batch_size, shuffle=False)
    edited_loader = DataLoader(edited_dataset, batch_size=args.batch_size, shuffle=False)

    # ── Load models and compute insulation scores ─────────────────────────────
    for ct in CELL_TYPES:
        print(f"\nCell type: {ct}")

        # Original sequences — average over all 4 models
        print("  Loading models 0–3 for original sequence scoring...")
        all_models = load_models(ct, ALL_MODEL_INDICES, device)
        df[f"insul_orig_{ct}"] = predict_insulation(
            orig_loader, all_models, device, average=True
        )
        del all_models
        torch.cuda.empty_cache()

        # Optimised sequences — opt models (0, 1)
        print("  Loading opt models 0–1 for optimised sequence scoring...")
        opt_models = load_models(ct, OPT_MODEL_INDICES, device)
        opt_scores = predict_insulation(
            edited_loader, opt_models, device, average=False
        )
        for idx, scores in zip(OPT_MODEL_INDICES, opt_scores):
            df[f"insul_opt_{ct}_model{idx}"] = scores
        del opt_models
        torch.cuda.empty_cache()

        # Optimised sequences — held-out models (2, 3)
        print("  Loading held-out models 2–3 for optimised sequence scoring...")
        held_models = load_models(ct, HELD_MODEL_INDICES, device)
        held_scores = predict_insulation(
            edited_loader, held_models, device, average=False
        )
        for idx, scores in zip(HELD_MODEL_INDICES, held_scores):
            df[f"insul_heldout_{ct}_model{idx}"] = scores
        del held_models
        torch.cuda.empty_cache()

    # ── Save ──────────────────────────────────────────────────────────────────
    out_path = os.path.join(
        run_dir,
        f"fold{fold}_selected_genomic_windows_centered_results.tsv",
    )
    df.to_csv(out_path, sep="\t", index=False)
    print(f"\nSaved → {out_path}")


if __name__ == "__main__":
    main()