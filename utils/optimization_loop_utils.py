"""
optimization_loop_utils.py

Shared building blocks for sequence design optimisation scripts.
Imported by run_boundary_design.py, run_flame_design.py, and run_dot_design.py.
"""

import os
import logging

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from ledidi import Ledidi
from utils.optimization_utils import build_stem, last_accepted_step, count_edits

from semifreddo.semifreddo import CTCFAwareSemifreddoWrapper


log = logging.getLogger(__name__)

MAX_ITER       = 2000
EARLY_STOPPING = 2000


def run_one_design(
    row: pd.Series,
    fold: int,
    args,
    sf_wrapper,
    output_loss: nn.Module,
    X: torch.Tensor,
    target: torch.Tensor,
    device: torch.device,
    out_dir: str,
) -> dict:
    """Run Ledidi optimisation for a single genomic window.

    Accepts a pre-built Semifreddo wrapper and output loss, so it works for
    any single-bin feature type (boundary, flame). For dots, use run_one_design_dot.

    Parameters
    ----------
    row : pd.Series
        Row from the flat-regions table (used only for logging via stem).
    fold : int
        Fold index (used only for logging).
    args : argparse.Namespace
        Must carry .L, .tau, .eps.
    sf_wrapper :
        Any SemifreddoLedidiWrapper instance with .center_bp_start / .center_bp_end.
    output_loss : nn.Module
        Pre-built output loss (e.g. LocalL1Loss with the feature mask).
    X : torch.Tensor
        Full one-hot sequence, shape (1, 4, L), already on device.
    target : torch.Tensor
        Optimisation target, already on device.
    device : torch.device
    out_dir : str
        Directory where <stem>_gen_seq.pt is saved.

    Returns
    -------
    dict with keys 'n_edits' and 'last_accepted_step'.
    """
    chrom = row["chrom"]
    start = int(row["centered_start"])
    end   = int(row["centered_end"])
    stem  = build_stem(chrom, start, end)
    
    # ── If the loss needs sequence access, wrap sf_wrapper and wire it in ─────
    if hasattr(output_loss, "seq_wrapper"):
        model_for_ledidi = CTCFAwareSemifreddoWrapper(sf_wrapper)
        output_loss.seq_wrapper = model_for_ledidi
        log.info("    CTCFAwareSemifreddoWrapper active — CTCF penalty enabled")
    else:
        model_for_ledidi = sf_wrapper
    
    X_center = X[:, :, sf_wrapper.center_bp_start:sf_wrapper.center_bp_end]

    ledidi_optimizer = Ledidi(
        sf_wrapper,
        shape               = X_center.shape[1:],
        input_loss          = torch.nn.L1Loss(reduction="sum"),
        output_loss         = output_loss,
        batch_size          = 1,
        l                   = args.L,
        tau                 = args.tau,
        eps                 = args.eps,
        max_iter            = MAX_ITER,
        early_stopping_iter = EARLY_STOPPING,
        return_history      = True,
        verbose             = False,
    ).cuda()

    generated_seq, history = ledidi_optimizer.fit_transform(X_center, target)

    # Reconstruct full sequence for edit counting
    full_generated_seq = X.clone()
    full_generated_seq[
        :, :, sf_wrapper.center_bp_start:sf_wrapper.center_bp_end
    ] = generated_seq

    n_edits   = count_edits(X, full_generated_seq)
    last_step = last_accepted_step(history)
    log.info(f"    Edits: {n_edits}  |  Last accepted step: {last_step}")

    torch.save(generated_seq.cpu(), os.path.join(out_dir, f"{stem}_gen_seq.pt"))
    log.info(f"    Saved → {os.path.join(out_dir, f'{stem}_gen_seq.pt')}")

    return {"n_edits": n_edits, "last_accepted_step": last_step}


def run_one_design_dot(
    row: pd.Series,
    fold: int,
    args,
    sf_wrapper,
    output_loss: nn.Module,
    X: torch.Tensor,
    target: torch.Tensor,
    device: torch.device,
    out_dir: str,
    bin_size: int = 2048,
) -> dict:
    """Run Ledidi optimisation for a single genomic window with two anchor bins.

    Identical to run_one_design except generated_anchors is split into lo/hi
    bins for edit counting and only the concatenated anchor tensor is saved.

    Parameters
    ----------
    sf_wrapper :
        TwoAnchorSemifreddoLedidiWrapper with .center_bp_start/.center_bp_end
        and .bp_lo_start/.bp_lo_end/.bp_hi_start/.bp_hi_end.
    bin_size : int
        Size of each anchor bin in bp (default 2048).
    All other parameters are identical to run_one_design.

    Returns
    -------
    dict with keys 'n_edits' and 'last_accepted_step'.
    """
    chrom = row["chrom"]
    start = int(row["centered_start"])
    end   = int(row["centered_end"])
    stem  = build_stem(chrom, start, end)

    bp_lo_start = sf_wrapper.bp_lo_start
    bp_lo_end   = sf_wrapper.bp_lo_end
    bp_hi_start = sf_wrapper.bp_hi_start
    bp_hi_end   = sf_wrapper.bp_hi_end

    X_lo = X[:, :, bp_lo_start:bp_lo_end]
    X_hi = X[:, :, bp_hi_start:bp_hi_end]
    X_anchors = torch.cat([X_lo, X_hi], dim=2)   # (1, 4, 2*BIN_SIZE)

    ledidi_optimizer = Ledidi(
        sf_wrapper,
        shape               = X_anchors.shape[1:],
        input_loss          = torch.nn.L1Loss(reduction="sum"),
        output_loss         = output_loss,
        batch_size          = 1,
        l                   = args.L,
        tau                 = args.tau,
        eps                 = args.eps,
        max_iter            = MAX_ITER,
        early_stopping_iter = EARLY_STOPPING,
        return_history      = True,
        verbose             = False,
    ).cuda()

    generated_anchors, history = ledidi_optimizer.fit_transform(X_anchors, target)

    # Reconstruct full sequence for edit counting
    full_generated_seq = X.clone()
    full_generated_seq[:, :, sf_wrapper.bp_lo_start:sf_wrapper.bp_lo_end] = generated_anchors[:, :, :bin_size]
    full_generated_seq[:, :, sf_wrapper.bp_hi_start:sf_wrapper.bp_hi_end] = generated_anchors[:, :, bin_size:]

    n_edits   = count_edits(X, full_generated_seq)
    last_step = last_accepted_step(history)
    log.info(f"    Edits: {n_edits}  |  Last accepted step: {last_step}")
    last_step = last_accepted_step(history)
    log.info(f"    Edits: {n_edits}  |  Last accepted step: {last_step}")

    torch.save(generated_anchors.cpu(), os.path.join(out_dir, f"{stem}_gen_seq.pt"))
    log.info(f"    Saved → {os.path.join(out_dir, f'{stem}_gen_seq.pt')}")

    return {"n_edits": n_edits, "last_accepted_step": last_step}


def run_fold(
    fold: int,
    args,
    run_one_fn,
    flat_regions_base: str,
    results_base_dir: str,
) -> None:
    """Iterate over all windows in a fold's flat-regions table.

    Parameters
    ----------
    fold : int
    args : argparse.Namespace
        Passed through to run_one_fn.
    run_one_fn : callable
        Signature: (row, fold, args, out_dir) -> dict.
        All feature-specific setup (wrapper, loss, tensor loading) happens inside.
    flat_regions_base : str
    results_base_dir : str
    """
    log.info(f"{'=' * 60}")
    log.info(f"Fold {fold}")
    log.info(f"{'=' * 60}")

    out_dir = os.path.join(results_base_dir, args.run_name, f"fold{fold}")
    os.makedirs(out_dir, exist_ok=True)

    tsv_path = (
        f"{flat_regions_base}/"
        f"fold{fold}_selected_genomic_windows_centered_chrom_states.tsv"
    )
    df = pd.read_csv(tsv_path, sep="\t")
    log.info(f"Loaded {len(df)} windows from {tsv_path}")

    results = []
    for i, row in df.iterrows():
        log.info(f"  [{i + 1}/{len(df)}]")
        try:
            meta = run_one_fn(row, fold, args, out_dir)
        except Exception as e:
            log.error(f"  FAILED: {e}")
            meta = {"n_edits": np.nan, "last_accepted_step": np.nan}
        results.append(meta)

    df_out  = pd.concat([df, pd.DataFrame(results, index=df.index)], axis=1)
    out_tsv = os.path.join(
        os.path.dirname(out_dir),
        f"fold{fold}_selected_genomic_windows_centered_chrom_states_opt.tsv",
    )
    df_out.to_csv(out_tsv, sep="\t", index=False)
    log.info(f"Saved enriched table → {out_tsv}")