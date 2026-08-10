"""
benchmark.py

Self-contained benchmark for boundary optimisation (single-bin, 2048 bp editable).
Measures wall-clock time and peak GPU memory of the ENTIRE optimization
(model forward passes + backprop-to-input + accept/reject), for N windows.

Inlines the per-window logic that normally lives in
run_one_design / run_fold, so the timer wraps exactly the Ledidi optimization
and nothing else (tensor loads and the per-window save are excluded).

Usage:
    python benchmark_boundary_design.py \
        --folds 0 \
        --seeds 0 \
        --run_name benchmark/boundary \
        --boundary_strength -0.5 \
        --L 125.0 \
        --n_benchmark 100 --warmup 1
"""

import os
import sys
import time
import argparse
import logging
import statistics
import numpy as np
import pandas as pd
import torch


# --- path setup -----------------------------------------------------------
# Cloned ledidi repo takes priority over any pip-installed version
sys.path.insert(0, os.path.abspath("/home1/smaruj/ledidi/"))

# Akita model package — appended so its utils/ does NOT shadow ledidi_akita/utils/
sys.path.append(os.path.abspath("/home1/smaruj/akita_pytorch/"))

# Project root — gives access to utils/
sys.path.insert(0, os.path.abspath("/home1/smaruj/akita_semifreddo/"))
# --------------------------------------------------------------------

from ledidi import Ledidi
from semifreddo.semifreddo import SemifreddoLedidiWrapper
from semifreddo.losses import LocalL1Loss
from semifreddo.optimization_loop import strength_tag  # leaf helper only
from utils.model_utils import load_model
from semifreddo.optimization_loop import build_stem, last_accepted_step, count_edits

# ── Default paths ─────────────────────────────────────────────────────────────
_PROJ = os.environ["AKITA_SF_DIR"]

DEFAULT_MODEL_PATH = os.environ["MOUSE_MODEL_CKPT"]
DEFAULT_SEQ_BASE_DIR      = f"{_PROJ}/analysis/flat_regions"
DEFAULT_TARGET_BASE_DIR   = f"{_PROJ}/optimizations/boundaries/targets"
DEFAULT_MASK_PATH         = f"{_PROJ}/optimizations/feature_masks/boundary_mask.pt"
DEFAULT_RESULTS_BASE_DIR  = f"{_PROJ}/optimizations/boundaries"
DEFAULT_FLAT_REGIONS_BASE = f"{_PROJ}/analysis/flat_regions/mouse_flat_regions_chrom_states_tsv"

# ── Semifreddo / architecture constants ───────────────────────────────────────
CENTER_BIN_MAP   = 256
CONTEXT_BINS     = 5
BIN_SIZE         = 2048
CROPPING_APPLIED = 64
N_TRIU           = 130305

MAX_ITER       = 2000
EARLY_STOPPING = 2000

TSV_SUFFIX = "fold{fold}_selected_genomic_windows_centered_chrom_states.tsv"

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ── Benchmark utility ─────────────────────────────────────────────────────────

def summarize(values):
    """Mean / std / min / max for a list of floats (std=0 if <2 samples)."""
    n = len(values)
    if n == 0:
        return dict(n=0, mean=float("nan"), std=float("nan"),
                    min=float("nan"), max=float("nan"))
    return dict(
        n=n,
        mean=statistics.fmean(values),
        std=statistics.stdev(values) if n > 1 else 0.0,
        min=min(values),
        max=max(values),
    )


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Benchmark boundary optimisation (single-bin editable)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--folds",    type=int, nargs="+", required=True,
                   help="One or more fold indices, e.g. --folds 0")
    p.add_argument("--seeds",    type=int, nargs="+", default=[0],
                   help="One or more random seeds")
    p.add_argument("--run_name", type=str, required=True,
                   help="Results subdirectory name for benchmark outputs")
    p.add_argument("--boundary_strength", type=float, required=True,
                   help="Value applied to off-diagonal quadrants of the boundary mask")
    p.add_argument("--L",   type=float, default=0.01,  help="Input-loss regularisation weight")
    p.add_argument("--tau", type=float, default=1.0,   help="Ledidi tau parameter")
    p.add_argument("--eps", type=float, default=1e-4,  help="Ledidi eps parameter")
    p.add_argument("--model_path",        default=DEFAULT_MODEL_PATH)
    p.add_argument("--seq_base_dir",      default=DEFAULT_SEQ_BASE_DIR)
    p.add_argument("--target_base_dir",   default=DEFAULT_TARGET_BASE_DIR)
    p.add_argument("--mask_path",         default=DEFAULT_MASK_PATH)
    p.add_argument("--results_base_dir",  default=DEFAULT_RESULTS_BASE_DIR)
    p.add_argument("--flat_regions_base", default=DEFAULT_FLAT_REGIONS_BASE)
    # ── benchmark args ──
    p.add_argument("--n_benchmark",     type=int, default=100,
                   help="Number of windows to benchmark, then stop")
    p.add_argument("--warmup",          type=int, default=1,
                   help="Leading windows run but excluded from averages "
                        "(absorbs CUDA/cuDNN init cost)")
    p.add_argument("--benchmark_label", type=str, default="boundary_design",
                   help="Tag written into the benchmark output filenames")
    p.add_argument("--save_sequences",  action="store_true",
                   help="Also save each generated sequence (outside the timer). "
                        "Off by default for a benchmark run.")
    return p.parse_args()


# ── One benchmarked optimization ──────────────────────────────────────────────

def benchmark_one_window(row, fold, args, model, output_loss, tag, device, is_cuda, out_dir):
    """Inlined run_one_design: load inputs, run Ledidi (timed), post-process (untimed).

    Returns a dict with timing/memory + a couple of sanity-check fields.
    """
    stem  = build_stem(row["chrom"], int(row["centered_start"]), int(row["centered_end"]))
    log.info(f"  Window: {stem}")

    # ── input prep (OUTSIDE the timed region) ────────────────────────────────
    X      = torch.load(f"{args.seq_base_dir}/mouse_sequences/fold{fold}/{stem}_X.pt",
                        weights_only=True).to(device)
    tower  = torch.load(f"{args.seq_base_dir}/mouse_tower_outputs/fold{fold}/{stem}_tower_out.pt",
                        weights_only=True).to(device)
    target = torch.load(f"{args.target_base_dir}/boundary_{tag}/fold{fold}/{stem}_target.pt",
                        weights_only=True).to(device)

    sf_wrapper = SemifreddoLedidiWrapper(
        model=model, precomputed_full_output=tower, full_X=X,
        edited_bin=CENTER_BIN_MAP, context_bins=CONTEXT_BINS,
        cropping_applied=CROPPING_APPLIED,
    )
    # single-bin restriction: only the center bin is handed to Ledidi to edit
    X_center = X[:, :, sf_wrapper.center_bp_start:sf_wrapper.center_bp_end]

    # ── reset peak-memory counter and start the clock ────────────────────────
    if is_cuda:
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    t0 = time.perf_counter()

    # ── THE ENTIRE OPTIMIZATION (model work included) ────────────────────────
    ledidi_optimizer = Ledidi(
        sf_wrapper,
        shape               = X_center.shape[1:],
        input_loss          = torch.nn.L1Loss(reduction="sum"),
        output_loss         = output_loss,
        input_mask          = None,
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

    # ── stop the clock and read peak memory ──────────────────────────────────
    if is_cuda:
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - t0

    peak_alloc = torch.cuda.max_memory_allocated(device) if is_cuda else float("nan")
    peak_resv  = torch.cuda.max_memory_reserved(device)  if is_cuda else float("nan")

    # ── post-processing (OUTSIDE the timed region) ───────────────────────────
    full_generated_seq = X.clone()
    full_generated_seq[:, :, sf_wrapper.center_bp_start:sf_wrapper.center_bp_end] = generated_seq
    n_edits   = count_edits(X, full_generated_seq)
    last_step = last_accepted_step(history)

    if args.save_sequences:
        torch.save(generated_seq.cpu(), os.path.join(out_dir, f"{stem}_gen_seq.pt"))

    rec = dict(
        fold=fold, chrom=row["chrom"],
        start=int(row["centered_start"]), end=int(row["centered_end"]),
        time_s=elapsed,
        peak_alloc_bytes=peak_alloc,
        peak_reserved_bytes=peak_resv,
        n_edits=n_edits, last_accepted_step=last_step,
    )

    # free per-window tensors so measurements stay clean window-to-window
    del X, tower, target, sf_wrapper, X_center, generated_seq, history, full_generated_seq, ledidi_optimizer
    if is_cuda:
        torch.cuda.empty_cache()

    return rec


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args    = parse_args()
    tag     = strength_tag(args.boundary_strength)
    device  = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    is_cuda = device.type == "cuda"

    log.info(f"Device: {device}  |  Folds: {args.folds}  |  Seeds: {args.seeds}")
    log.info(f"Boundary strength: {args.boundary_strength} (tag: {tag})  |  "
             f"L={args.L}  tau={args.tau}  eps={args.eps}")
    log.info(f"Benchmarking {args.n_benchmark} optimizations "
             f"(first {args.warmup} excluded from averages as warm-up).")

    # ── Shared resources (loaded once) ────────────────────────────────────────
    model       = load_model(args.model_path, device)
    mask        = torch.load(args.mask_path, weights_only=True).to(device)
    output_loss = LocalL1Loss(mask, n_triu=N_TRIU, reduction="sum").to(device)

    records = []

    # ── Inlined run_fold: iterate windows, stop once we have enough ──────────
    stop = False
    for seed in args.seeds:
        if stop:
            break
        torch.manual_seed(seed)
        log.info(f"=== Seed {seed} ===")

        for fold in args.folds:
            if stop:
                break

            out_dir = os.path.join(args.results_base_dir, args.run_name, f"fold{fold}")
            os.makedirs(out_dir, exist_ok=True)

            tsv_path = os.path.join(args.flat_regions_base, TSV_SUFFIX.format(fold=fold))
            df = pd.read_csv(tsv_path, sep="\t")
            log.info(f"Fold {fold}: loaded {len(df)} windows from {tsv_path}")

            for _, row in df.iterrows():
                if len(records) >= args.n_benchmark:
                    stop = True
                    break

                log.info(f"  [{len(records) + 1}/{args.n_benchmark}]")
                try:
                    rec = benchmark_one_window(
                        row, fold, args, model, output_loss, tag, device, is_cuda, out_dir
                    )
                except Exception as e:
                    log.error(f"  FAILED ({e}); skipping this window (not counted).")
                    if is_cuda:
                        torch.cuda.empty_cache()
                    continue

                rec["warmup"] = len(records) < args.warmup
                flag = "  [warm-up, excluded]" if rec["warmup"] else ""
                msg = f"    time={rec['time_s']:.2f}s"
                if is_cuda:
                    msg += (f"  peak_alloc={rec['peak_alloc_bytes'] / 1024**3:.2f} GiB"
                            f"  peak_reserved={rec['peak_reserved_bytes'] / 1024**3:.2f} GiB")
                log.info(msg + f"  edits={rec['n_edits']}" + flag)

                records.append(rec)

    log.info(f"Benchmarking complete: {len(records)} optimizations recorded.")

    # ======================================================================
    # Benchmark output
    # ======================================================================
    if not records:
        log.warning("No optimizations were benchmarked; nothing to write.")
        return

    bench_df = pd.DataFrame(records)
    bench_df["peak_alloc_GiB"]    = bench_df["peak_alloc_bytes"]    / 1024**3
    bench_df["peak_reserved_GiB"] = bench_df["peak_reserved_bytes"] / 1024**3

    out_base = os.path.join(args.results_base_dir, args.run_name)
    os.makedirs(out_base, exist_ok=True)

    per_window_path = os.path.join(
        out_base, f"benchmark_{args.benchmark_label}_per_window.tsv"
    )
    bench_df.to_csv(per_window_path, sep="\t", index=False)

    # Averages exclude warm-up windows.
    kept = bench_df[~bench_df["warmup"]]
    t_stats     = summarize(kept["time_s"].tolist())
    alloc_stats = summarize(kept["peak_alloc_GiB"].tolist())
    resv_stats  = summarize(kept["peak_reserved_GiB"].tolist())

    summary_path = os.path.join(
        out_base, f"benchmark_{args.benchmark_label}_summary.tsv"
    )
    with open(summary_path, "w") as fh:
        fh.write("metric\tn\tmean\tstd\tmin\tmax\n")
        fh.write(f"time_s\t{t_stats['n']}\t{t_stats['mean']:.4f}\t"
                 f"{t_stats['std']:.4f}\t{t_stats['min']:.4f}\t{t_stats['max']:.4f}\n")
        fh.write(f"peak_alloc_GiB\t{alloc_stats['n']}\t{alloc_stats['mean']:.4f}\t"
                 f"{alloc_stats['std']:.4f}\t{alloc_stats['min']:.4f}\t{alloc_stats['max']:.4f}\n")
        fh.write(f"peak_reserved_GiB\t{resv_stats['n']}\t{resv_stats['mean']:.4f}\t"
                 f"{resv_stats['std']:.4f}\t{resv_stats['min']:.4f}\t{resv_stats['max']:.4f}\n")

    log.info(f"Benchmark ({args.benchmark_label}, n={t_stats['n']} after warm-up):")
    log.info(f"  avg time           = {t_stats['mean']:.2f} ± {t_stats['std']:.2f} s")
    log.info(f"  avg peak allocated = {alloc_stats['mean']:.2f} ± {alloc_stats['std']:.2f} GiB")
    log.info(f"  avg peak reserved  = {resv_stats['mean']:.2f} ± {resv_stats['std']:.2f} GiB")
    log.info(f"  per-window -> {per_window_path}")
    log.info(f"  summary    -> {summary_path}")


if __name__ == "__main__":
    main()