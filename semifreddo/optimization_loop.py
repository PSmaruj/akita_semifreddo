"""
semifreddo/optimization_loop.py

Shared building blocks for sequence design optimization scripts.
Imported by run_boundary_design.py, run_flame_design.py, and run_dot_design.py.

Each run_one_* function handles a single genomic window: it wraps a pre-built
Semifreddo wrapper and output loss in a Ledidi optimizer, runs the optimization,
saves the generated sequence, and returns edit statistics. run_fold iterates
run_one_* over all windows in a fold's flat-regions table.

Optimization loop
-----------------
run_one_design     : run Ledidi optimization for a single-bin feature (boundary, flame)
run_one_design_dot : run Ledidi optimization for a two-anchor-bin feature (dot)
run_fold           : iterate a run_one_* function over all windows in a fold

Utilities
---------
build_stem         : build a filesystem-safe stem string from genomic coordinates
strength_tag       : convert a float feature strength to a filesystem-safe string
last_accepted_step : return the last Ledidi iteration that introduced a new edit
count_edits        : count nucleotide positions that differ between two sequences
"""

import sys
import os
import logging
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from ledidi import Ledidi

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
    input_mask: torch.Tensor | None = None,
) -> dict:
    """Run Ledidi optimization for a single genomic window (single-bin feature).

    Wraps a pre-built Semifreddo wrapper and output loss in a Ledidi optimizer,
    runs optimization on the central editable bin, saves the generated sequence,
    and returns edit statistics. For dot optimization (two anchor bins), use
    run_one_design_dot instead.

    Parameters
    ----------
    row : pd.Series
        Row from the flat-regions table with columns [chrom, centered_start,
        centered_end]; used to build the output filename stem.
    fold : int
        Fold index; currently used for logging only.
    args : argparse.Namespace
        Must carry .L (Ledidi regularization weight), .tau (temperature),
        and .eps (step size).
    sf_wrapper : SemifreddoLedidiWrapper or CTCFAwareSemifreddoWrapper
        Pre-built wrapper exposing .center_bp_start and .center_bp_end.
    output_loss : nn.Module
        Pre-built output loss, e.g. LocalL1Loss or LocalL1LossWithCTCFPenalty.
    X : torch.Tensor
        Full one-hot sequence of shape (1, 4, L), already on device.
    target : torch.Tensor
        Optimization target of shape (1, 1, N_triu), already on device.
    device : torch.device
        Device X and target live on.
    out_dir : str
        Directory where {stem}_gen_seq.pt is saved.
    input_mask : torch.Tensor or None
        Shape (seq_len,); boolean mask passed to Ledidi where True positions
        are frozen (weight set to -inf). Used to freeze CTCF motif positions
        during suppression. If None, no positions are frozen (default).

    Returns
    -------
    dict
        Keys: 'n_edits' (int), 'last_accepted_step' (int).
    """
    chrom = row["chrom"]
    start = int(row["centered_start"])
    end   = int(row["centered_end"])
    stem  = build_stem(chrom, start, end)
    
    X_center = X[:, :, sf_wrapper.center_bp_start:sf_wrapper.center_bp_end]
    
    print(f"model type passed to Ledidi: {type(sf_wrapper)}")
    
    ledidi_optimizer = Ledidi(
        sf_wrapper,
        shape               = X_center.shape[1:],
        input_loss          = torch.nn.L1Loss(reduction="sum"),
        output_loss         = output_loss,
        input_mask          = input_mask,
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
    """Run Ledidi optimization for a single genomic window (two-anchor-bin dot feature).

    Identical to run_one_design except the editable region consists of two
    anchor bins (lo and hi) concatenated into a single (1, 4, 2*bin_size)
    input. The generated anchors are split back into lo/hi for edit counting,
    and saved as a single concatenated tensor.

    Parameters
    ----------
    row : pd.Series
        Row from the flat-regions table with columns [chrom, centered_start,
        centered_end]; used to build the output filename stem.
    fold : int
        Fold index; currently used for logging only.
    args : argparse.Namespace
        Must carry .L (Ledidi regularization weight), .tau (temperature),
        and .eps (step size).
    sf_wrapper : TwoAnchorSemifreddoLedidiWrapper
        Pre-built wrapper exposing .bp_lo_start, .bp_lo_end,
        .bp_hi_start, .bp_hi_end.
    output_loss : nn.Module
        Pre-built output loss, e.g. LocalL1Loss with a dot mask.
    X : torch.Tensor
        Full one-hot sequence of shape (1, 4, L), already on device.
    target : torch.Tensor
        Optimization target of shape (1, 1, N_triu), already on device.
    device : torch.device
        Device X and target live on.
    out_dir : str
        Directory where {stem}_gen_seq.pt is saved.
    bin_size : int
        Size of each anchor bin in bp (default 2048).

    Returns
    -------
    dict
        Keys: 'n_edits' (int), 'last_accepted_step' (int).
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

    torch.save(generated_anchors.cpu(), os.path.join(out_dir, f"{stem}_gen_seq.pt"))
    log.info(f"    Saved → {os.path.join(out_dir, f'{stem}_gen_seq.pt')}")

    return {"n_edits": n_edits, "last_accepted_step": last_step}


def run_fold(
    fold: int,
    args,
    run_one_fn,
    flat_regions_base: str,
    results_base_dir: str,
    tsv_suffix: str = "fold{fold}_selected_genomic_windows_centered_chrom_states.tsv",
) -> None:
    """Iterate a run_one_* function over all windows in a fold's flat-regions table.

    Loads the flat-regions TSV for the given fold, calls run_one_fn on each
    row, collects edit statistics, and writes an enriched TSV with results
    appended as new columns.

    Parameters
    ----------
    fold : int
        Fold index.
    args : argparse.Namespace
        Passed through to run_one_fn; must carry .run_name for output directory.
    run_one_fn : callable
        Signature: (row, fold, args, out_dir) -> dict. All feature-specific
        setup (wrapper, loss, tensor loading) is handled inside this function.
    flat_regions_base : str
        Directory containing the flat-regions TSV files.
    results_base_dir : str
        Root directory for output; results are written to
        results_base_dir/args.run_name/fold{fold}/.
    tsv_suffix : str
        Filename template for the flat-regions TSV; must contain '{fold}'.
        Defaults to the mouse chrom-states filename. For human flat regions,
        pass "fold{fold}_selected_genomic_windows_centered.tsv".
    """
    log.info(f"{'=' * 60}")
    log.info(f"Fold {fold}")
    log.info(f"{'=' * 60}")

    out_dir = os.path.join(results_base_dir, args.run_name, f"fold{fold}")
    os.makedirs(out_dir, exist_ok=True)

    tsv_path = os.path.join(flat_regions_base, tsv_suffix.format(fold=fold))
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
        tsv_suffix.format(fold=fold).replace(".tsv", "_opt.tsv"),
    )
    df_out.to_csv(out_tsv, sep="\t", index=False)
    log.info(f"Saved enriched table → {out_tsv}")


# =============================================================================
# Utilities
# =============================================================================


def strength_tag(strength: float) -> str:
    """Convert a float feature strength to a filesystem-safe string.

    Parameters
    ----------
    strength : float
        Feature strength value, e.g. -0.5, 1.0, 0.0.

    Returns
    -------
    str
        Filesystem-safe string representation, e.g. -0.5 → 'neg0p5',
        1.0 → 'pos1p0', 0.0 → '0p0'.
    """
    prefix = "neg" if strength < 0 else ("pos" if strength > 0 else "")
    val_str = f"{abs(strength):.1f}".replace(".", "p")
    
    return f"{prefix}{val_str}"


def build_stem(chrom: str, start: int, end: int) -> str:
    """Build a filesystem-safe stem string from genomic coordinates.

    Parameters
    ----------
    chrom : str
        Chromosome name, e.g. 'chr1'.
    start : int
        Window start coordinate.
    end : int
        Window end coordinate.

    Returns
    -------
    str
        Stem string of the form '{chrom}_{start}_{end}',
        e.g. 'chr1_1000000_2097152'.
    """
    return f"{chrom}_{start}_{end}"


def last_accepted_step(history: dict) -> int:
    """Return the index of the last iteration that introduced a new edited position.

    Each entry in history["edits"] is a tuple of three tensors:
        (batch_indices, nucleotide_indices, position_indices)
    We track the cumulative set of positions seen across steps 0..i-1 and
    return the last i where history["edits"][i][2] contains a position not
    previously seen.
    """
    edits = history["edits"]
    seen  = set()
    last  = 0
    for i, edit in enumerate(edits):
        positions = set(edit[2].cpu().tolist())
        if positions - seen:
            last = i
            seen |= positions
    return last


def count_edits(original_X: torch.Tensor, generated_full: torch.Tensor) -> int:
    """Count nucleotide positions that differ between the original and generated sequence.

    Parameters
    ----------
    original_X : torch.Tensor
        Original one-hot sequence of shape (1, 4, L).
    generated_full : torch.Tensor
        Generated one-hot sequence of the same shape.

    Returns
    -------
    int
        Number of positions where argmax differs between the two sequences.
    """
    return int(
        (torch.argmax(generated_full, dim=1) != torch.argmax(original_X, dim=1))
        .sum()
        .item()
    )