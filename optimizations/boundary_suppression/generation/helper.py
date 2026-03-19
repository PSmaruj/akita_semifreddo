import torch

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from matplotlib.patches import FancyArrow
import matplotlib.colors as mcolors
import seaborn as sns


def make_ctcf_exclusion_mask(
    motif_positions: list[tuple[int, int]],
    flank: int = 15,
    seq_len: int = 2048,
) -> torch.Tensor:
    """
    Build an input_mask for Ledidi that freezes positions overlapping CTCF motifs.

    Parameters
    ----------
    motif_positions : list of (start, end) tuples in relative coordinates
                      within the editable bin (0-indexed, end exclusive).
    flank           : number of bp to freeze on each side of each motif.
    seq_len         : length of the editable sequence passed to Ledidi.

    Returns
    -------
    mask : boolean tensor of shape (seq_len,), True = position is frozen.
    """
    mask = torch.zeros(seq_len, dtype=torch.bool)
    for start, end in motif_positions:
        frozen_start = max(0, start - flank)
        frozen_end   = min(seq_len, end + flank)
        mask[frozen_start:frozen_end] = True
    return mask


def plot_history(
    history: dict,
    ctcf_positions: list[tuple[int, int]],
    ctcf_hits: pd.DataFrame,
    ctcf_flank: int = 15,
    seq_len: int = 2048,
    figsize: tuple = (14, 5),
    title: str = "Edit history: edits over optimisation",
) -> tuple:
    """
    Visualise which sequence positions were edited at each Ledidi iteration,
    with CTCF motif positions shown below.

    Parameters
    ----------
    history         : dict returned by Ledidi with return_history=True.
    ctcf_positions  : list of (start, end) tuples (relative, within editable bin,
                      end-exclusive) — used to shade frozen regions.
    ctcf_hits       : DataFrame with columns ["start", "end", "strand"] for
                      CTCF arrows in the lower panel.
    ctcf_flank      : bp frozen on each side of each motif (default: 15).
    seq_len         : Length of the editable sequence (default: 2048).
    figsize         : Figure size passed to plt.figure.
    title           : Plot title.

    Returns
    -------
    fig, (ax_main, ax_ctcf)
    """
    sns.set_theme(style="white")

    batch_size = history["batch_size"]
    n_iter     = len(history["edits"])

    # ── Build (n_iter, seq_len) edit-frequency matrix ─────────────────────────
    freq = np.zeros((n_iter, seq_len), dtype=float)
    for t, where_tuple in enumerate(history["edits"]):
        positions = where_tuple[2].cpu().numpy()
        for pos in positions:
            if pos < seq_len:
                freq[t, pos] += 1
    freq /= batch_size

    iters, pos_list, alphas = [], [], []
    for t in range(n_iter):
        edited = np.nonzero(freq[t])[0]
        for pos in edited:
            iters.append(t)
            pos_list.append(pos)
            alphas.append(min(freq[t, pos], 1.0))

    # ── Layout ────────────────────────────────────────────────────────────────
    main_h = max(4, n_iter * 0.25)
    fig = plt.figure(figsize=figsize)
    gs  = GridSpec(
        2, 1,
        height_ratios=[main_h, 50],
        hspace=0.05,
        figure=fig,
    )
    ax_main = fig.add_subplot(gs[0])
    ax_ctcf = fig.add_subplot(gs[1], sharex=ax_main)

    # ── Frozen region shading ─────────────────────────────────────────────────
    COLOR_FLANK = "#42C74D"   # light green
    COLOR_CORE  = "#23782B"   # deep green  (used for arrows too)
    COLOR_EDIT  = "#585958"   # grey

    for start, end in ctcf_positions:
        flank_start = max(0, start - ctcf_flank)
        flank_end   = min(seq_len, end + ctcf_flank)
        # flanks
        ax_main.axvspan(flank_start, start, color=COLOR_FLANK, linewidth=0, zorder=0)
        ax_main.axvspan(end,         flank_end, color=COLOR_FLANK, linewidth=0, zorder=0)
        # core motif
        ax_main.axvspan(start, end, color=COLOR_CORE,  linewidth=0, zorder=0)

    # ── Edit scatter ──────────────────────────────────────────────────────────
    if iters:
        # ax_main.scatter(
        #     pos_list, iters,
        #     c=COLOR_EDIT,
        #     alpha=0.5,
        #     s=4,
        #     linewidths=0,
        #     rasterized=True,
        # )
        
        # 1. Convert your hex grey to an RGB tuple (0.0 to 1.0)
        base_rgb = mcolors.to_rgb(COLOR_EDIT)
        
        # 2. Create a list of RGBA tuples using your pre-calculated alphas
        # This makes rare edits faint and frequent edits dark
        alpha_scaling = 0.05
        rgba_colors = [(*base_rgb, min(a * alpha_scaling, 1.0)) for a in alphas]

        ax_main.scatter(
            pos_list, 
            iters,
            c=rgba_colors,   # Use the individual RGBA values
            s=4,
            linewidths=0,
            rasterized=True,
        )

    ax_main.set_ylabel("Iteration", fontsize=11)
    ax_main.set_title(title, fontsize=13)
    ax_main.set_xlim(-10, seq_len + 10)
    ax_main.set_ylim(-0.5, n_iter - 0.5)
    ax_main.invert_yaxis()
    ax_main.tick_params(labelbottom=False)

    legend_handles = [
        mpatches.Patch(color=COLOR_EDIT,  label="Edited"),
        mpatches.Patch(color=COLOR_CORE,  label="CTCF core (frozen)"),
        mpatches.Patch(color=COLOR_FLANK, label=f"CTCF ±{ctcf_flank} bp flank (frozen)"),
    ]
    ax_main.legend(handles=legend_handles, loc="upper right", fontsize=9)

    # ── CTCF strip ────────────────────────────────────────────────────────────
    COLOR_POS = "blue"
    COLOR_NEG = "red"
    Y_ARROW   = 0.5

    ax_ctcf.axhline(Y_ARROW, color="lightgrey", linewidth=0.8, linestyle="--", alpha=0.8)

    for _, row in ctcf_hits.iterrows():
        start  = int(row["start"])
        end    = int(row["end"])
        strand = row["strand"]
        color  = COLOR_POS if strand == "+" else COLOR_NEG
        width  = end - start
        dx     = width if strand == "+" else -width

        arrow = FancyArrow(
            start if strand == "+" else end,
            Y_ARROW,
            dx, 0,
            width=0.15,
            head_width=0.30,
            head_length=min(25, abs(width) * 0.4),
            length_includes_head=True,
            color=color,
            zorder=3,
        )
        ax_ctcf.add_patch(arrow)

    ax_ctcf.set_xlim(-10, seq_len + 10)
    ax_ctcf.set_ylim(0, 1)
    ax_ctcf.set_yticks([Y_ARROW])
    ax_ctcf.set_yticklabels(["CTCF"], fontsize=9)
    ax_ctcf.set_xlabel("Position within editable bin (bp)", fontsize=11)
    ax_ctcf.tick_params(axis="y", length=0)

    sns.despine(ax=ax_main, bottom=True, left=True)
    sns.despine(ax=ax_ctcf, left=True)

    plt.tight_layout()
    return fig, (ax_main, ax_ctcf)