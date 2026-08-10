#!/usr/bin/env python
"""
run_cassette_design.py

Repeated Ledidi + Semifreddo boundary-removal design at the Chakraborty et al.
C1-C4 cassette.

Starts from the C1-C4flx sequence and optimises toward a target map in which
the cross-boundary block carries the measured Cre-minus-flx difference. Edits
are confined to the 672 bp cassette. Each run uses an independent seed; the
script records where the accepted edits fall so the positional distribution
can be compared against the CTCF core annotations afterwards.

Outputs (in --out_dir)
---------------------
  run_records.json        per-run edit positions, counts, and boundary scores
  edit_positions.npz      pooled per-position tallies + motif annotation
  run_summary.tsv         one row per run, for quick inspection

Usage
-----
python run_cassette_design.py --n_runs 100 --lam 10.0 \
    --out_dir /path/to/cassette_boundary_lost
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import pearsonr

from memelite import fimo                                          # noqa: E402
from akita.model import SeqNN                                      # noqa: E402
from ledidi import Ledidi                                          # noqa: E402
from semifreddo.semifreddo import RegionSemifreddoLedidiWrapper    # noqa: E402
from semifreddo.losses import LocalL1Loss                          # noqa: E402
from utils.data_utils import (                                     # noqa: E402
    from_upper_triu,
    fragment_indices_in_upper_triangular,
)
from utils.fimo_utils import read_meme_pwm                         # noqa: E402

# =============================================================================
# Constants
# =============================================================================

MODEL_PATH = (
    "/home1/smaruj/akita_pytorch/models/finetuned/mouse/Hsieh2019_mESC"
    "/checkpoints/Akita_v2_mouse_Hsieh2019_mESC_model0_finetuned.pth"
)

CHROM        = "chr7"
LEFT         = 144_105_472
DEL_START    = 144_775_729
CASSETTE_LEN = 672

BP_START = DEL_START - LEFT                  # 670,257
BP_END   = BP_START + CASSETTE_LEN           # 670,929

MAP_SIZE   = 512
BIN_SIZE   = 2048
PADDING    = 64
NUM_DIAGS  = 2
N_TRIU     = 130_305

B_CASSETTE     = (DEL_START - LEFT) // BIN_SIZE - PADDING     # 263
ROW_LO, ROW_HI = 165, B_CASSETTE                              # 165–262
COL_LO, COL_HI = B_CASSETTE, 323                              # 263–322

LOXP        = "ATAACTTCGTATAGCATACATTATACGAAGTTAT"
IDX_TO_BASE = np.array(list("ACGT"))
MOTIF_ORDER = ["C1", "C2", "C3", "C4"]


# =============================================================================
# Helpers
# =============================================================================

def onehot_to_str(t):
    return "".join(IDX_TO_BASE[torch.argmax(t, dim=1).squeeze(0).cpu().numpy()])


def boundary_score(y):
    """Mean predicted signal in the cross-boundary rectangle.
    Higher = more contact across the boundary = weaker boundary."""
    if torch.is_tensor(y):
        y = y[0, 0, :].detach().cpu().numpy()
    mat = from_upper_triu(y, matrix_len=MAP_SIZE, num_diags=NUM_DIAGS)
    return float(np.nanmean(mat[ROW_LO:ROW_HI, COL_LO:COL_HI]))


def edit_positions(X_ref, X_opt):
    """Cassette-relative indices (0 .. CASSETTE_LEN-1) where the base changed.

    Both tensors cover the bin-aligned block, so BP_START - edit_bp_start is
    subtracted to express positions relative to the cassette start.
    """
    a = torch.argmax(X_ref, dim=1).squeeze(0)
    b = torch.argmax(X_opt, dim=1).squeeze(0)
    return torch.nonzero(a != b).squeeze(-1).cpu().numpy()


def locate_cores(cassette, pwm_path, flank):
    """FIMO-scan the cassette and return {label: (start, end, strand)} in
    cassette-relative coordinates, core +/- flank, clamped to the cassette."""
    pwm  = read_meme_pwm(pwm_path)
    hits = fimo(motifs={"CTCF": pwm},
                sequences=cassette.cpu().detach().numpy(),
                threshold=1e-4, reverse_complement=True)[0]
    hits = hits.sort_values("start").reset_index(drop=True)

    if len(hits) != len(MOTIF_ORDER):
        raise RuntimeError(
            f"expected {len(MOTIF_ORDER)} CTCF hits, got {len(hits)}"
        )

    cores = {}
    for label, row in zip(MOTIF_ORDER, hits.itertuples()):
        i = max(int(row.start) - flank, 0)
        j = min(int(row.end)   + flank, CASSETTE_LEN)
        cores[label] = (i, j, row.strand)

    if cores["C1"][2] != "-":
        raise RuntimeError(f"C1 is {cores['C1'][2]} strand, expected minus")
    if any(cores[k][2] != "+" for k in ("C2", "C3", "C4")):
        raise RuntimeError("C2-C4 should all be plus strand")

    spans = sorted((i, j) for i, j, _ in cores.values())
    for (_, j1), (i2, _) in zip(spans, spans[1:]):
        if j1 > i2:
            raise RuntimeError(f"motif windows overlap at {j1}/{i2}; reduce flank")

    return cores


def breakdown(pos, cores, motif_bp, background_bp):
    """Counts and per-kb rates for one run."""
    out = {"total": int(len(pos))}
    in_motif = 0
    for label, (i, j, _) in cores.items():
        n = int(((pos >= i) & (pos < j)).sum())
        out[label] = n
        in_motif += n
    out["background"] = out["total"] - in_motif
    out["C2-C4"]      = out["C2"] + out["C3"] + out["C4"]

    c234_bp = sum(motif_bp[k] for k in ("C2", "C3", "C4"))
    out["rate_C1"]     = out["C1"] / motif_bp["C1"] * 1000
    out["rate_C2-C4"]  = out["C2-C4"] / c234_bp * 1000
    out["rate_bg"]     = out["background"] / background_bp * 1000
    out["C1_vs_C2-C4"] = (out["rate_C1"] / out["rate_C2-C4"]
                          if out["C2-C4"] else float("inf"))
    out["C1_vs_bg"]    = (out["rate_C1"] / out["rate_bg"]
                          if out["background"] else float("inf"))
    return out


# =============================================================================
# Argument parsing
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="Repeated boundary-removal design at the C1-C4 cassette."
    )
    p.add_argument("--base_path", type=str, default=(
        "/project2/fudenber_735/smaruj/sequence_design/"
        "ledidi_semifreddo_akita/optimizations/validation"
    ))
    p.add_argument("--out_dir", type=str, default=None,
                   help="Default: <base_path>/cassette_boundary_lost")
    p.add_argument("--pwm_path", type=str, default="./../data/pwm/MA0139.1.meme")
    p.add_argument("--n_runs", type=int, default=100)
    p.add_argument("--seed0", type=int, default=0,
                   help="First seed; runs use seed0 .. seed0+n_runs-1.")
    p.add_argument("--lam", type=float, default=10.0)
    p.add_argument("--max_iter", type=int, default=2000)
    p.add_argument("--early_stopping", type=int, default=2000)
    p.add_argument("--flank", type=int, default=20)
    p.add_argument("--save_every", type=int, default=10,
                   help="Checkpoint results every N runs.")
    return p.parse_args()


# =============================================================================
# Main
# =============================================================================

def main():
    args    = parse_args()
    out_dir = args.out_dir or os.path.join(args.base_path,
                                           "cassette_boundary_lost")
    os.makedirs(out_dir, exist_ok=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Output: {out_dir}\n")

    # ── Model and inputs ─────────────────────────────────────────────────────
    model = SeqNN()
    model.load_state_dict(
        torch.load(MODEL_PATH, map_location=device, weights_only=True)
    )
    model.to(device).eval()

    X      = torch.load(f"{args.base_path}/cassette_seq.pt",
                        weights_only=True).to(device)
    tower  = torch.load(f"{args.base_path}/cassette_seq_convt_output.pt",
                        weights_only=True).to(device)
    target = torch.load(f"{args.base_path}/target_cassette_boundary_lost.pt",
                        weights_only=True).to(device)
    while target.ndim < 3:
        target = target.unsqueeze(0)

    print(f"X {tuple(X.shape)}  tower {tuple(tower.shape)}  "
          f"target {tuple(target.shape)}")
    assert 0 < BP_START < BP_END < X.shape[-1]

    # ── CTCF cores ───────────────────────────────────────────────────────────
    cassette     = X[:, :, BP_START:BP_END]
    cassette_str = onehot_to_str(cassette)
    assert cassette_str.startswith(LOXP) and cassette_str.endswith(LOXP), \
        "cassette does not start and end with loxP"

    cores = locate_cores(cassette, args.pwm_path, args.flank)
    motif_bp      = {k: j - i for k, (i, j, _) in cores.items()}
    background_bp = CASSETTE_LEN - sum(motif_bp.values())

    print("\nCTCF cores (cassette-relative, +/- flank):")
    for label, (i, j, strand) in cores.items():
        print(f"  {label} ({strand})  bp {i}–{j-1}  ({j - i} bp)")
    print(f"  motifs {sum(motif_bp.values())} bp, background {background_bp} bp, "
          f"chance rate {sum(motif_bp.values()) / CASSETTE_LEN:.1%}\n")

    # ── Wrapper ──────────────────────────────────────────────────────────────
    sf_wrapper = RegionSemifreddoLedidiWrapper(
        model                   = model,
        precomputed_full_output = tower,
        full_X                  = X,
        bp_start                = BP_START,
        bp_end                  = BP_END,
        context_bins            = 5,
        splice_buffer           = 2,
        cropping_applied        = PADDING,
    ).to(device)

    X_edit0 = X[:, :, sf_wrapper.edit_bp_start:sf_wrapper.edit_bp_end].clone()
    cass_offset = BP_START - sf_wrapper.edit_bp_start   # cassette start within block

    with torch.no_grad():
        y_full = model(X)
        y_sf   = sf_wrapper(X_edit0)
    r, _ = pearsonr(y_full.cpu().flatten().numpy(), y_sf.cpu().flatten().numpy())
    print(f"Sanity check — Pearson R (full vs Semifreddo): {r:.6f}")
    assert r > 0.9999, "Semifreddo does not reproduce the full model"

    # ── Loss ─────────────────────────────────────────────────────────────────
    fragment_bool = np.zeros((MAP_SIZE, MAP_SIZE), dtype=bool)
    fragment_bool[ROW_LO:ROW_HI, COL_LO:COL_HI] = True
    fragment_bool[COL_LO:COL_HI, ROW_LO:ROW_HI] = True
    boundary_mask = torch.tensor(
        fragment_indices_in_upper_triangular(matrix_size=MAP_SIZE,
                                             fragment_mask=fragment_bool)
    ).to(device)

    output_loss = LocalL1Loss(boundary_mask, n_triu=N_TRIU,
                              reduction="sum").to(device)
    input_loss  = nn.L1Loss(reduction="sum").to(device)

    S_START, S_TARGET = boundary_score(y_full), boundary_score(target)
    print(f"\nboundary score  start {S_START:+.5f}  target {S_TARGET:+.5f}  "
          f"gap {S_TARGET - S_START:+.5f}")
    assert S_TARGET > S_START, "target does not raise cross-boundary signal"
    print(f"mask {boundary_mask.shape[0]:,} entries, "
          f"scale {N_TRIU / boundary_mask.shape[0]:.1f}x\n")

    # ── Runs ─────────────────────────────────────────────────────────────────
    records   = []
    tally     = np.zeros(CASSETTE_LEN, dtype=np.int64)
    t0        = time.time()

    for k in range(args.n_runs):
        seed = args.seed0 + k
        torch.manual_seed(seed)
        np.random.seed(seed)

        opt = Ledidi(
            sf_wrapper,
            shape               = X_edit0.shape[1:],
            input_loss          = input_loss,
            output_loss         = output_loss,
            batch_size          = 1,
            l                   = args.lam,
            max_iter            = args.max_iter,
            early_stopping_iter = args.early_stopping,
            return_history      = False,
            verbose             = False,
        ).to(device)

        X_gen = opt.fit_transform(X_edit0, target)
        if isinstance(X_gen, (tuple, list)):
            X_gen = X_gen[0]
        X_gen = sf_wrapper.freeze(X_gen.to(device))

        pos_block = edit_positions(X_edit0, X_gen)
        pos       = pos_block - cass_offset                  # cassette-relative
        in_range  = (pos >= 0) & (pos < CASSETTE_LEN)
        assert in_range.all(), (
            f"seed {seed}: {int((~in_range).sum())} edits outside the cassette"
        )
        tally[pos] += 1

        full_gen = X.clone()
        full_gen[:, :, sf_wrapper.edit_bp_start:sf_wrapper.edit_bp_end] = X_gen
        with torch.no_grad():
            pred = model(full_gen)

        s_opt = boundary_score(pred)
        brk   = breakdown(pos, cores, motif_bp, background_bp)

        records.append({
            "seed": seed,
            "lambda": args.lam,
            "boundary_score": s_opt,
            "gap_closed": (s_opt - S_START) / (S_TARGET - S_START),
            "l1_to_target": float((pred - target).abs().sum()),
            "edits": brk,
            "edit_pos": pos.tolist(),
        })

        print(f"[{k+1:3d}/{args.n_runs}] seed {seed:3d}: "
              f"{brk['total']:3d} edits | C1 {brk['C1']:2d} C2 {brk['C2']:2d} "
              f"C3 {brk['C3']:2d} C4 {brk['C4']:2d} bg {brk['background']:3d} | "
              f"score {s_opt:+.5f} "
              f"({(s_opt - S_START) / (S_TARGET - S_START):5.1%}) | "
              f"{(time.time() - t0) / (k + 1):.1f}s/run", flush=True)

        if (k + 1) % args.save_every == 0 or k == args.n_runs - 1:
            _write(out_dir, records, tally, cores, motif_bp, background_bp,
                   args, S_START, S_TARGET)

    # ── Summary ──────────────────────────────────────────────────────────────
    tot = {k: sum(r["edits"][k] for r in records)
           for k in ("total", "C1", "C2", "C3", "C4", "background")}
    chance = {**{k: motif_bp[k] / CASSETTE_LEN for k in cores},
              "background": background_bp / CASSETTE_LEN}

    print(f"\n{'='*68}\npooled over {len(records)} runs "
          f"({tot['total']:,} edits total)\n")
    print(f"{'region':<12}{'n':>7}{'obs/exp':>10}")
    for k in ("C1", "C2", "C3", "C4", "background"):
        oe = tot[k] / (tot["total"] * chance[k]) if tot["total"] else np.nan
        print(f"{k:<12}{tot[k]:>7}{oe:>10.2f}")

    gaps = np.array([r["gap_closed"] for r in records])
    eds  = np.array([r["edits"]["total"] for r in records])
    print(f"\nedits per run : {eds.mean():.1f} +/- {eds.std():.1f} "
          f"(min {eds.min()}, max {eds.max()})")
    print(f"gap closed    : {gaps.mean():.1%} +/- {gaps.std():.1%}")
    print(f"total time    : {(time.time() - t0) / 60:.1f} min")


def _write(out_dir, records, tally, cores, motif_bp, background_bp,
           args, s_start, s_target):
    """Checkpoint all outputs."""
    with open(os.path.join(out_dir, "run_records.json"), "w") as fh:
        json.dump({
            "config": vars(args),
            "s_start": s_start, "s_target": s_target,
            "cassette_len": CASSETTE_LEN,
            "cores": {k: [int(i), int(j), str(s)] for k, (i, j, s) in cores.items()},
            "motif_bp": motif_bp, "background_bp": background_bp,
            "runs": records,
        }, fh, indent=2)

    np.savez(
        os.path.join(out_dir, "edit_positions.npz"),
        tally=tally,
        n_runs=len(records),
        cassette_len=CASSETTE_LEN,
        core_labels=np.array(list(cores)),
        core_starts=np.array([cores[k][0] for k in cores]),
        core_ends=np.array([cores[k][1] for k in cores]),
        core_strands=np.array([cores[k][2] for k in cores]),
        flank=args.flank, lam=args.lam,
        s_start=s_start, s_target=s_target,
        bp_start=BP_START, bp_end=BP_END, left=LEFT, del_start=DEL_START,
    )

    pd.DataFrame([{
        "seed": r["seed"], "n_edits": r["edits"]["total"],
        "C1": r["edits"]["C1"], "C2": r["edits"]["C2"],
        "C3": r["edits"]["C3"], "C4": r["edits"]["C4"],
        "background": r["edits"]["background"],
        "C1_vs_C2-C4": r["edits"]["C1_vs_C2-C4"],
        "boundary_score": r["boundary_score"],
        "gap_closed": r["gap_closed"],
    } for r in records]).to_csv(
        os.path.join(out_dir, "run_summary.tsv"), sep="\t", index=False
    )


if __name__ == "__main__":
    main()