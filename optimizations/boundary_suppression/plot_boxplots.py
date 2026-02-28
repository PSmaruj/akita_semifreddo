"""
Two separate boxplot figures:
  Plot 1: Before vs After optimization (2048bp)
  Plot 2: CTCFs vs B2+CTCF vs B2 noCTCF (220bp references)

Usage:
    from plot_filter_boxplots_separate import plot_separate_boxplots
    plot_separate_boxplots(acts_ctcf, acts_b2ctcf, acts_b2no,
                           acts_before, acts_after,
                           top_filters=[68, 56, 117])
"""

import numpy as np
import matplotlib.pyplot as plt
import os

OUTPUT_DIR = "b2_filter_tracking"


def maxpool(acts):
    if acts.ndim == 3:
        return acts.max(axis=2)
    return acts


def plot_separate_boxplots(acts_ctcf, acts_b2ctcf, acts_b2no,
                            acts_before, acts_after,
                            top_filters=None, n_top=3,
                            save_dir=OUTPUT_DIR):
    os.makedirs(save_dir, exist_ok=True)

    mp_ctcf = maxpool(acts_ctcf)
    mp_b2ctcf = maxpool(acts_b2ctcf)
    mp_b2no = maxpool(acts_b2no)
    mp_before = maxpool(acts_before)
    mp_after = maxpool(acts_after)

    if top_filters is None:
        deltas = mp_after.mean(0) - mp_before.mean(0)
        top_filters = np.argsort(deltas)[::-1][:n_top].tolist()

    n = len(top_filters)

    # ══════════════════════════════════════════════════════════════════════
    # PLOT 1: Before vs After optimization (2048bp)
    # ══════════════════════════════════════════════════════════════════════
    fig, axes = plt.subplots(1, n, figsize=(3.5 * n, 5), sharey=False)
    if n == 1:
        axes = [axes]

    for i, fi in enumerate(top_filters):
        ax = axes[i]
        data = [mp_before[:, fi], mp_after[:, fi]]
        labels = ["Before\noptimization", "After\noptimization"]
        colors = ["#A8DADC", "#E76F51"]

        bp = ax.boxplot(data, labels=labels, patch_artist=True,
                        widths=0.55, showfliers=False)
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.75)
            patch.set_edgecolor("k")
            patch.set_linewidth(1)
        for median in bp["medians"]:
            median.set_color("k")
            median.set_linewidth(1.5)

        # for j, (d, color) in enumerate(zip(data, colors)):
        #     jitter = np.random.normal(0, 0.05, len(d))
        #     ax.scatter(np.ones(len(d)) * (j + 1) + jitter, d,
        #               s=8, alpha=0.25, c=color, edgecolors="none", zorder=3)

        delta = mp_after[:, fi].mean() - mp_before[:, fi].mean()
        pct = (delta / mp_before[:, fi].mean()) * 100 if mp_before[:, fi].mean() != 0 else 0
        ax.set_title(f"Filter {fi}\nΔ = {delta:+.1f} ({pct:+.0f}%)", fontsize=11)
        ax.grid(alpha=0.3, axis="y")
        ax.set_ylabel("Max-pooled activation" if i == 0 else "")

    plt.suptitle("Before vs After boundary suppression optimization\n(conv_tower_block3, 2048bp)",
                 fontsize=13, fontweight="bold", y=1.04)
    plt.tight_layout()
    path1 = f"{save_dir}/boxplot_before_after_top{n}.svg"
    plt.savefig(path1, format="svg")
    plt.close(fig)
    print(f"Saved: {path1}")

    # ══════════════════════════════════════════════════════════════════════
    # PLOT 2: CTCFs vs B2+CTCF vs B2 noCTCF (220bp)
    # ══════════════════════════════════════════════════════════════════════
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 5), sharey=False)
    if n == 1:
        axes = [axes]

    for i, fi in enumerate(top_filters):
        ax = axes[i]
        data = [mp_ctcf[:, fi], mp_b2ctcf[:, fi], mp_b2no[:, fi]]
        labels = ["CTCFs", "SINE B2\n+ CTCF", "SINE B2\nno CTCF"]
        colors = ["#E63946", "#F4A261", "#457B9D"]

        bp = ax.boxplot(data, labels=labels, patch_artist=True,
                        widths=0.55, showfliers=False)
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.75)
            patch.set_edgecolor("k")
            patch.set_linewidth(1)
        for median in bp["medians"]:
            median.set_color("k")
            median.set_linewidth(1.5)

        for j, (d, color) in enumerate(zip(data, colors)):
            jitter = np.random.normal(0, 0.05, len(d))
            ax.scatter(np.ones(len(d)) * (j + 1) + jitter, d,
                      s=8, alpha=0.25, c=color, edgecolors="none", zorder=3)

        ax.set_title(f"Filter {fi}", fontsize=11, fontweight="bold")
        ax.grid(alpha=0.3, axis="y")
        ax.set_ylabel("Max-pooled activation" if i == 0 else "")

    plt.suptitle("Reference group activations\n(conv_tower_block3, 220bp sequences)",
                 fontsize=13, fontweight="bold", y=1.04)
    plt.tight_layout()
    path2 = f"{save_dir}/boxplot_references_top{n}.svg"
    plt.savefig(path2, format="svg")
    plt.close(fig)
    print(f"Saved: {path2}")

    return top_filters