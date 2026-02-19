"""
Volcano-style plot for conv_tower_block3 filters.

X-axis: mean activation change (after - before optimization)
Y-axis: -log10(p-value) from paired t-test or Wilcoxon test

Highlights filters 68, 81, 22 (top B2+CTCF-specific) to see if they
are among the most enriched after optimization.

Usage:
    from plot_filter_volcano import plot_filter_volcano
    plot_filter_volcano(acts_before, acts_after,
                        highlight_filters=[68, 81, 22])
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import os

OUTPUT_DIR = "b2_filter_tracking"


def maxpool(acts):
    if acts.ndim == 3:
        return acts.max(axis=2)
    return acts


def plot_filter_volcano(acts_before, acts_after,
                         highlight_filters=None,
                         highlight_labels=None,
                         save_dir=OUTPUT_DIR):
    """
    Volcano plot: all 128 filters, x = delta activation, y = significance.

    Args:
        acts_before: (N, 128, spatial) or (N, 128)
        acts_after:  (N, 128, spatial) or (N, 128)
        highlight_filters: list of filter indices to label (e.g., [68, 81, 22])
        highlight_labels: optional dict {filter_idx: "label"}
    """
    os.makedirs(save_dir, exist_ok=True)

    mp_before = maxpool(acts_before)
    mp_after = maxpool(acts_after)
    n_filters = mp_before.shape[1]

    if highlight_filters is None:
        highlight_filters = [68, 81, 22]

    if highlight_labels is None:
        highlight_labels = {f: f"F{f}" for f in highlight_filters}

    # Compute delta and p-value for each filter
    deltas = []
    pvals = []
    for fi in range(n_filters):
        before_vals = mp_before[:, fi]
        after_vals = mp_after[:, fi]
        delta = after_vals.mean() - before_vals.mean()
        # Wilcoxon signed-rank test (paired, non-parametric)
        try:
            stat, pval = stats.wilcoxon(after_vals, before_vals, alternative="two-sided")
        except ValueError:
            pval = 1.0
        deltas.append(delta)
        pvals.append(pval)

    deltas = np.array(deltas)
    pvals = np.array(pvals)
    # Avoid log(0)
    pvals = np.clip(pvals, 1e-300, 1.0)
    neg_log_p = -np.log10(pvals)

    # ── Plot ─────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 6))

    # Classify points
    is_highlight = np.array([fi in highlight_filters for fi in range(n_filters)])
    sig_threshold = -np.log10(0.05 / n_filters)  # Bonferroni

    # Background points
    bg = ~is_highlight
    ax.scatter(deltas[bg], neg_log_p[bg],
              s=30, c="#B0B0B0", alpha=0.5, edgecolors="none", zorder=2)

    # Highlighted filters
    for fi in highlight_filters:
        color = "#E63946"
        ax.scatter(deltas[fi], neg_log_p[fi],
                  s=120, c=color, edgecolors="k", linewidths=1.2,
                  zorder=4, marker="D")
        ax.annotate(
            highlight_labels.get(fi, f"F{fi}"),
            (deltas[fi], neg_log_p[fi]),
            textcoords="offset points",
            xytext=(8, 8),
            fontsize=10, fontweight="bold", color=color,
            arrowprops=dict(arrowstyle="-", color=color, lw=0.8),
        )

    # Reference lines
    ax.axhline(sig_threshold, color="gray", linestyle="--", lw=0.8, alpha=0.6)
    ax.axvline(0, color="gray", linestyle="-", lw=0.5, alpha=0.5)
    ax.text(ax.get_xlim()[1] * 0.95, sig_threshold + 0.5,
            f"Bonferroni p < 0.05", ha="right", fontsize=7,
            fontstyle="italic", color="gray")

    # Label other notable filters (top 5 by |delta| that aren't highlighted)
    top_by_delta = np.argsort(np.abs(deltas))[::-1]
    n_labeled = 0
    for fi in top_by_delta:
        if fi in highlight_filters:
            continue
        if n_labeled >= 5:
            break
        ax.annotate(
            f"F{fi}",
            (deltas[fi], neg_log_p[fi]),
            textcoords="offset points",
            xytext=(6, -6),
            fontsize=7, color="#555555", alpha=0.8,
        )
        n_labeled += 1

    ax.set_xlabel("Δ mean activation (after − before optimization)", fontsize=11)
    ax.set_ylabel("−log₁₀(p-value)", fontsize=11)
    ax.set_title("Filter activation changes after boundary suppression\n"
                 "(conv_tower_block3, all 128 filters)",
                 fontsize=12, fontweight="bold")
    ax.grid(alpha=0.2)

    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker="D", color="w", markerfacecolor="#E63946",
               markersize=10, markeredgecolor="k", label="B2+CTCF-specific filters"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#B0B0B0",
               markersize=8, label="Other filters"),
    ]
    ax.legend(handles=legend_elements, loc="upper left", fontsize=9)

    plt.tight_layout()
    # path = f"{save_dir}/volcano_filters.png"
    # fig.savefig(path, dpi=200, bbox_inches="tight")
    path = f"{save_dir}/volcano_filters.svg"
    plt.savefig(path, format="svg")
    
    plt.close(fig)
    print(f"Saved: {path}")

    # ── Print summary ────────────────────────────────────────────────────
    print(f"\nHighlighted filters:")
    for fi in highlight_filters:
        print(f"  Filter {fi}: Δ={deltas[fi]:+.2f}, p={pvals[fi]:.2e}, "
              f"-log10(p)={neg_log_p[fi]:.1f}")

    print(f"\nTop 10 filters by |Δ activation|:")
    for fi in np.argsort(np.abs(deltas))[::-1][:10]:
        tag = " ★" if fi in highlight_filters else ""
        print(f"  Filter {fi:3d}: Δ={deltas[fi]:+.2f}, p={pvals[fi]:.2e}{tag}")

    # Where do highlighted filters rank?
    rank_by_delta = np.argsort(deltas)[::-1]
    print(f"\nRank of highlighted filters (by Δ, descending):")
    for fi in highlight_filters:
        rank = np.where(rank_by_delta == fi)[0][0] + 1
        print(f"  Filter {fi}: rank {rank}/128")

    return deltas, pvals, neg_log_p