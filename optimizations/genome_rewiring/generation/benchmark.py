"""
benchmark.py

Runs Ledidi genome rewiring on all windows of a given fold.
Each source window is optimised so that Akita v2 predicts a contact
map resembling that of the next window in the fold (circular shift).

Now also benchmarks per-optimization wall-clock time and peak GPU
memory, and writes the averages (plus a per-window breakdown) to file.

Outputs
-------
- One .pt file per window containing the optimised OHE sequence
- A TSV summarising the optimization (last accepted step per window)
- A per-window benchmark TSV and a one-line averages summary
"""

import argparse
import os
import sys
import time
import statistics

import pandas as pd
import torch

from ledidi import ledidi
from utils.model_utils import load_model
from utils.df_utils import build_optimization_table

# ==========================================================================
# Helpers
# ==========================================================================

def load_tensor(path: str, device: torch.device) -> torch.Tensor:
    return torch.load(path, weights_only=True, map_location=device)


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


# ==========================================================================
# CLI
# ==========================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Ledidi genome rewiring on selected genomic loci."
    )
    parser.add_argument("--fold",               type=int, required=True)
    parser.add_argument("--model_path",         type=str, required=True)
    parser.add_argument("--input_dir",          type=str, required=True)
    parser.add_argument("--output_dir",         type=str, required=True)
    parser.add_argument("--max_iter",           type=int, default=2000)
    parser.add_argument("--early_stopping_iter",type=int, default=2000)
    parser.add_argument("--l",                  type=float, default=0.05,
                        help="Input/output loss mixing weight")
    parser.add_argument("--n_benchmark",        type=int, default=100,
                        help="Number of windows to benchmark (default: 100)")
    parser.add_argument("--warmup",             type=int, default=1,
                        help="Leading windows run but excluded from averages "
                             "(absorbs CUDA/cuDNN init cost; default: 1)")
    parser.add_argument("--benchmark_label",    type=str, default="genome_rewiring",
                        help="Tag written into the benchmark output filenames")
    return parser.parse_args()


# ==========================================================================
# Main
# ==========================================================================

def main() -> None:
    args = parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    is_cuda = device.type == "cuda"
    print(f"Using device: {device}")

    model = load_model(args.model_path, device)

    FOLD       = args.fold
    input_dir  = args.input_dir
    output_dir = args.output_dir

    results_dir = os.path.join(output_dir, f"results_fold{FOLD}")
    os.makedirs(results_dir, exist_ok=True)

    df = pd.read_csv(f"{input_dir}/df_select_fold{FOLD}.tsv", sep="\t")
    df = build_optimization_table(df)
    df.to_csv(
        f"{output_dir}/genomic_optimization_fold{FOLD}.tsv",
        sep="\t", index=False,
    )

    # Cap the number of optimizations we run for the benchmark.
    n_run = min(args.n_benchmark, len(df))
    df = df.head(n_run).reset_index(drop=True)
    print(f"Benchmarking {n_run} optimizations "
          f"(first {args.warmup} excluded from averages as warm-up).")

    bench_rows = []  # per-window timing + memory records

    for i, row in enumerate(df.itertuples(index=False)):
        chrom,  pred_start,   pred_end   = row.chrom,        row.start,        row.end
        tchrom, target_start, target_end = row.target_chrom, row.target_start, row.target_end

        print(f"\n[{i+1}/{len(df)}] {chrom}:{pred_start}-{pred_end}  →  {tchrom}:{target_start}-{target_end}")

        # --- load inputs (kept OUTSIDE the timed region) ---
        X = load_tensor(
            f"{input_dir}/ohe_X_fold{FOLD}/{chrom}_{pred_start}_{pred_end}_X.pt",
            device,
        )  # (1, 4, seq_len)

        y_bar = load_tensor(
            f"{input_dir}/genomic_targets_fold{FOLD}/{tchrom}_{target_start}_{target_end}_target.pt",
            device,
        )  # (1, 1, 130305)

        # --- reset peak-memory counter and start the clock ---
        if is_cuda:
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
        t0 = time.perf_counter()

        generated_seq, history = ledidi(
            model, X, y_bar,
            batch_size          = 1,
            l                   = args.l,
            max_iter            = args.max_iter,
            early_stopping_iter = args.early_stopping_iter,
            input_loss          = torch.nn.L1Loss(reduction="sum"),
            output_loss         = torch.nn.L1Loss(reduction="sum"),
            return_history      = True,
            verbose             = True,
            device              = device,
        )

        # --- stop the clock and read peak memory ---
        if is_cuda:
            torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - t0

        peak_alloc = torch.cuda.max_memory_allocated(device) if is_cuda else float("nan")
        peak_resv  = torch.cuda.max_memory_reserved(device)  if is_cuda else float("nan")

        is_warmup = i < args.warmup
        print(f"    time={elapsed:.2f}s"
              + (f"  peak_alloc={peak_alloc/1024**3:.2f} GiB"
                 f"  peak_reserved={peak_resv/1024**3:.2f} GiB" if is_cuda else "")
              + ("  [warm-up, excluded]" if is_warmup else ""))

        bench_rows.append(dict(
            chrom=chrom, start=pred_start, end=pred_end,
            time_s=elapsed,
            peak_alloc_bytes=peak_alloc,
            peak_reserved_bytes=peak_resv,
            warmup=is_warmup,
        ))

        # last_accepted_step: index of the final improvement before early stopping
        total_losses = history["total_loss"]
        last_accepted = int(min(range(len(total_losses)), key=lambda k: total_losses[k]))
        df.at[i, "last_accepted_step"] = last_accepted

        torch.save(
            generated_seq.cpu(),
            os.path.join(results_dir, f"{chrom}_{pred_start}_{pred_end}_seq.pt"),
        )

        del generated_seq, history, X, y_bar
        if is_cuda:
            torch.cuda.empty_cache()

    df.to_csv(
        f"{output_dir}/genomic_optimization_TIME.tsv",
        sep="\t", index=False,
    )

    # ======================================================================
    # Benchmark output
    # ======================================================================
    bench_df = pd.DataFrame(bench_rows)
    bench_df["peak_alloc_GiB"]    = bench_df["peak_alloc_bytes"]    / 1024**3
    bench_df["peak_reserved_GiB"] = bench_df["peak_reserved_bytes"] / 1024**3

    per_window_path = os.path.join(
        output_dir, f"benchmark_{args.benchmark_label}_fold{FOLD}_per_window.tsv"
    )
    bench_df.to_csv(per_window_path, sep="\t", index=False)

    # Averages exclude warm-up windows.
    kept = bench_df[~bench_df["warmup"]]
    t_stats     = summarize(kept["time_s"].tolist())
    alloc_stats = summarize(kept["peak_alloc_GiB"].tolist())
    resv_stats  = summarize(kept["peak_reserved_GiB"].tolist())

    summary_path = os.path.join(
        output_dir, f"benchmark_{args.benchmark_label}_fold{FOLD}_summary.tsv"
    )
    with open(summary_path, "w") as fh:
        fh.write("metric\tn\tmean\tstd\tmin\tmax\n")
        fh.write(f"time_s\t{t_stats['n']}\t{t_stats['mean']:.4f}\t"
                 f"{t_stats['std']:.4f}\t{t_stats['min']:.4f}\t{t_stats['max']:.4f}\n")
        fh.write(f"peak_alloc_GiB\t{alloc_stats['n']}\t{alloc_stats['mean']:.4f}\t"
                 f"{alloc_stats['std']:.4f}\t{alloc_stats['min']:.4f}\t{alloc_stats['max']:.4f}\n")
        fh.write(f"peak_reserved_GiB\t{resv_stats['n']}\t{resv_stats['mean']:.4f}\t"
                 f"{resv_stats['std']:.4f}\t{resv_stats['min']:.4f}\t{resv_stats['max']:.4f}\n")

    print(f"\nBenchmark ({args.benchmark_label}, fold {FOLD}, "
          f"n={t_stats['n']} after warm-up):")
    print(f"  avg time           = {t_stats['mean']:.2f} ± {t_stats['std']:.2f} s")
    print(f"  avg peak allocated = {alloc_stats['mean']:.2f} ± {alloc_stats['std']:.2f} GiB")
    print(f"  avg peak reserved  = {resv_stats['mean']:.2f} ± {resv_stats['std']:.2f} GiB")
    print(f"  per-window -> {per_window_path}")
    print(f"  summary    -> {summary_path}")
    print("\nDone.")


if __name__ == "__main__":
    main()