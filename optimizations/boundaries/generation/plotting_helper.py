import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from matplotlib.patches import FancyArrow

# ── Parse edit history ────────────────────────────────────────────────────────

def parse_edit_positions(edit_entry):
    """Return the set of edited positions from one history entry (3-tuple of tensors)."""
    pos_tensor = edit_entry[2]  # third element: positions within 2048 bp
    if pos_tensor.numel() == 0:
        return set()
    return set(pos_tensor.cpu().numpy().tolist())

def build_edit_events(history_edits):
    """
    Walk through accepted-edit snapshots and label each position at each step as:
      - 'new'      : newly accepted at this step
      - 'retained' : accepted in a prior step, still present
      - (absent)   : not edited at this step
    
    Returns
    -------
    steps         : list of step indices (one per accepted snapshot)
    new_pos       : list of arrays — newly accepted positions at each step
    retained_pos  : list of arrays — previously accepted positions still present
    """
    steps, new_pos, retained_pos = [], [], []
    prev_positions = set()

    for i, entry in enumerate(history_edits):
        curr_positions = parse_edit_positions(entry)
        newly_accepted  = sorted(curr_positions - prev_positions)
        still_retained  = sorted(curr_positions & prev_positions)

        steps.append(i)
        new_pos.append(np.array(newly_accepted,  dtype=int))
        retained_pos.append(np.array(still_retained, dtype=int))

        prev_positions = curr_positions

    return steps, new_pos, retained_pos

# ── Plot proposed and accepted ────────────────────────────────────────────────────────

def plot_edit_history(
    new_pos, 
    retained_pos, 
    edited_hits, 
    x_lim=(-10, 2058), 
    title="Edit history: accepted edits over optimisation",
    figsize=(14, 5)
):
    """
    Plots the history of proposed/accepted edits along with CTCF strand markers.
    """
    n_steps = len(new_pos)
    main_h = max(4, n_steps * 0.25)
    
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(
        2, 1,
        height_ratios=[main_h, 50], # Adjusted ratio for better CTCF visibility
        hspace=0.05,
        figure=fig,
    )
    
    ax_main = fig.add_subplot(gs[0])
    ax_ctcf = fig.add_subplot(gs[1], sharex=ax_main)

    # --- Main Scatter ---
    COLOR_PROP = "#95a5a6"
    COLOR_ACC  = "forestgreen"

    for step_idx, (new, ret) in enumerate(zip(new_pos, retained_pos)):
        y = step_idx
        if hasattr(new, 'size') and new.size:
            ax_main.scatter(new, np.full(new.size, y), color=COLOR_PROP,
                            s=6, linewidths=0, alpha=0.9, rasterized=True)
        if hasattr(ret, 'size') and ret.size:
            ax_main.scatter(ret, np.full(ret.size, y), color=COLOR_ACC,
                            s=4, linewidths=0, alpha=0.6, rasterized=True)

    ax_main.set_ylabel("Iteration (Snapshot)", fontsize=11)
    ax_main.set_title(title, fontsize=13)
    ax_main.set_xlim(*x_lim)
    ax_main.set_ylim(-0.5, n_steps - 0.5)
    ax_main.invert_yaxis()
    ax_main.tick_params(labelbottom=False)

    legend_handles = [
        mpatches.Patch(color=COLOR_PROP, label="Proposed"),
        mpatches.Patch(color=COLOR_ACC,  label="Accepted"),
    ]
    ax_main.legend(handles=legend_handles, loc="upper right", fontsize=9)

    # --- CTCF Strip ---
    COLOR_POS = "blue"
    COLOR_NEG = "red"
    Y_ARROW   = 0.5

    ax_ctcf.axhline(Y_ARROW, color="lightgrey", linewidth=0.8, linestyle="--", alpha=0.8)

    for _, row in edited_hits.iterrows():
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

    ax_ctcf.set_xlim(*x_lim)
    ax_ctcf.set_ylim(0, 1)
    ax_ctcf.set_yticks([Y_ARROW])
    ax_ctcf.set_yticklabels(["CTCF"], fontsize=9)
    ax_ctcf.set_xlabel("Position within sequence (bp)", fontsize=11)
    ax_ctcf.tick_params(axis="y", length=0)

    plt.tight_layout()
    return fig, (ax_main, ax_ctcf)