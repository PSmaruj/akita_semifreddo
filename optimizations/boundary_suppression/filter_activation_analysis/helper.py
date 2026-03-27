import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from matplotlib.lines import Line2D

TOP_K = 20
BASES = ["A", "C", "G", "T"]


def build_filter_df(a1, a2, a3, top_k=TOP_K):
    """Mean max-pooled activation per filter, union of top-k across groups."""
    m1, m2, m3 = a1.mean(0), a2.mean(0), a3.mean(0)
    df = pd.DataFrame({
        "filter_idx": np.arange(len(m1)),
        "CTCF":       m1,
        "B2+CTCF":    m2,
        "B2 noCTCF":  m3,
    })
    top_union = (
        set(df.nlargest(top_k, "CTCF")["filter_idx"])
        | set(df.nlargest(top_k, "B2+CTCF")["filter_idx"])
        | set(df.nlargest(top_k, "B2 noCTCF")["filter_idx"])
    )
    return df, top_union


def plot_filter_heatmap(a1, a2, a3, top_k=TOP_K, save_path=None):
    df, top_union = build_filter_df(a1, a2, a3, top_k)
    sub = (
        df[df["filter_idx"].isin(top_union)]
        .set_index("filter_idx")[["CTCF", "B2+CTCF", "B2 noCTCF"]]
    )
    # Normalise each filter's row to its own maximum
    sub = sub.div(sub.max(axis=1), axis=0).sort_values("CTCF", ascending=False)

    fig, ax = plt.subplots(figsize=(4, max(6, len(sub) * 0.22)))
    sns.heatmap(
        sub, cmap="YlOrRd", ax=ax,
        cbar_kws={"label": "Relative activation", "shrink": 0.6},
        linewidths=0.3, linecolor="white",
    )
    ax.set_ylabel("Filter index")
    ax.set_title(f"Top {top_k} filters per group (union)", fontsize=10)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def one_hot_to_seq(oh: np.ndarray) -> str:
    """oh : (4, L) → DNA string; ambiguous positions shown as N."""
    seq = []
    for pos in range(oh.shape[1]):
        col = oh[:, pos]
        if col.max() == 0.25:
            seq.append("N")
        else:
            seq.append(BASES[col.argmax()])
    return "".join(seq)


def rc(seq: str) -> str:
    comp = str.maketrans("ACGT", "TGCA")
    return seq.translate(comp)[::-1]


def top_activating_seqs(filter_idx: int, a2_spatial: np.ndarray,
                         seqs_g2: np.ndarray, kernel_size: int,
                         n_top: int = 3, use_rc: bool = False):
    """
    For *filter_idx*, return the n_top B2+CTCF sequences with the highest
    max activation, together with the 15 bp substring at the peak position.
    """
    # Max activation per sequence for this filter
    max_acts = a2_spatial[:, filter_idx, :].max(axis=1)  # (n_seqs,)
    top_idxs = np.argsort(max_acts)[::-1][:n_top]

    results = []
    for seq_i in top_idxs:
        peak_pos  = a2_spatial[seq_i, filter_idx, :].argmax()
        # conv_block_1 has pool=2, so input position ≈ peak_pos * 2
        inp_start = peak_pos * 2
        inp_end   = inp_start + kernel_size
        subseq    = one_hot_to_seq(seqs_g2[seq_i, :, inp_start:inp_end])
        if use_rc:
            subseq = rc(subseq)
        results.append((seq_i, float(max_acts[seq_i]), subseq))
    return results


def kernel_to_consensus(w: np.ndarray, use_rc: bool = False) -> str:
    """w : (4, kernel_size) → consensus string via argmax over bases."""
    consensus = "".join(BASES[i] for i in w.argmax(axis=0))
    return rc(consensus) if use_rc else consensus


def top_activating_seqs(filter_idx: int, a2_spatial: np.ndarray,
                         seqs_g2: np.ndarray, kernel_size: int,
                         n_top: int = 3, use_rc: bool = False):
    """
    For *filter_idx*, return the n_top B2+CTCF sequences with the highest
    max activation, together with the kernel_size bp substring at the peak position.
    """
    max_acts = a2_spatial[:, filter_idx, :].max(axis=1)  # (n_seqs,)
    top_idxs = np.argsort(max_acts)[::-1][:n_top]
    results  = []
    for seq_i in top_idxs:
        peak_pos  = a2_spatial[seq_i, filter_idx, :].argmax()
        inp_start = peak_pos * 2           # conv_block_1 pool stride = 2
        inp_end   = inp_start + kernel_size
        subseq    = one_hot_to_seq(seqs_g2[seq_i, :, inp_start:inp_end])
        if use_rc:
            subseq = rc(subseq)
        results.append((seq_i, float(max_acts[seq_i]), subseq))
    return results


def maxpool(acts):
    return acts.max(axis=2) if acts.ndim == 3 else acts


def plot_boxplots_references(l4_g1, l4_g2, l4_g3, filters, save_path=None):
    """One boxplot per filter: CTCFs / B2+CTCF / B2 noCTCF (220bp groups)."""
    mp1, mp2, mp3 = maxpool(l4_g1), maxpool(l4_g2), maxpool(l4_g3)
    n = len(filters)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 5), sharey=False)
    if n == 1:
        axes = [axes]

    for ax, fi in zip(axes, filters):
        data   = [mp1[:, fi], mp2[:, fi], mp3[:, fi]]
        labels = ["CTCFs", "SINE B2\n+ CTCF", "SINE B2\nno CTCF"]
        colors = ["#E63946", "#F4A261", "#457B9D"]

        bp = ax.boxplot(data, labels=labels, patch_artist=True,
                        widths=0.55, showfliers=False)
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color); patch.set_alpha(0.75)
            patch.set_edgecolor("k");   patch.set_linewidth(1)
        for med in bp["medians"]:
            med.set_color("k"); med.set_linewidth(1.5)
        # for j, (d, color) in enumerate(zip(data, colors)):
        #     jitter = np.random.normal(0, 0.05, len(d))
        #     ax.scatter(np.ones(len(d)) * (j + 1) + jitter, d,
        #                s=8, alpha=0.25, c=color, edgecolors="none", zorder=3)

        ax.set_title(f"Filter {fi}", fontsize=11, fontweight="bold")
        ax.grid(alpha=0.3, axis="y")
        ax.set_ylabel("Max-pooled activation" if ax is axes[0] else "")

    plt.suptitle("Reference group activations\n(conv_tower_block3, 220bp)",
                 fontsize=13, fontweight="bold", y=1.04)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, format="svg", bbox_inches="tight")
    plt.show()


def plot_boxplots_before_after(l4_before, l4_after, filters, save_path=None):
    """One boxplot per filter: before / after boundary suppression (2048bp)."""
    mp_before, mp_after = maxpool(l4_before), maxpool(l4_after)
    n = len(filters)
    fig, axes = plt.subplots(1, n, figsize=(3.5 * n, 5), sharey=False)
    if n == 1:
        axes = [axes]

    for ax, fi in zip(axes, filters):
        data   = [mp_before[:, fi], mp_after[:, fi]]
        labels = ["Before\noptimization", "After\noptimization"]
        colors = ["#A8DADC", "#E76F51"]

        bp = ax.boxplot(data, labels=labels, patch_artist=True,
                        widths=0.55, showfliers=False)
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color); patch.set_alpha(0.75)
            patch.set_edgecolor("k");   patch.set_linewidth(1)
        for med in bp["medians"]:
            med.set_color("k"); med.set_linewidth(1.5)

        delta = mp_after[:, fi].mean() - mp_before[:, fi].mean()
        pct   = (delta / mp_before[:, fi].mean()) * 100 if mp_before[:, fi].mean() != 0 else 0
        ax.set_title(f"Filter {fi}\nΔ = {delta:+.1f} ({pct:+.0f}%)", fontsize=11)
        ax.grid(alpha=0.3, axis="y")
        ax.set_ylabel("Max-pooled activation" if ax is axes[0] else "")

    plt.suptitle("Before vs after boundary suppression optimization\n(conv_tower_block3, 2048bp)",
                 fontsize=13, fontweight="bold", y=1.04)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, format="svg", bbox_inches="tight")
    plt.show()


def plot_volcano(l4_before, l4_after, highlight_filters,
                 save_path=None):
    """
    X: mean activation change (after − before), per filter.
    Y: −log10(p-value) from paired Wilcoxon signed-rank test.
    Highlighted filters shown as red diamonds with labels.
    """
    mp_before = maxpool(l4_before)
    mp_after  = maxpool(l4_after)
    n_filters = mp_before.shape[1]

    deltas, pvals = [], []
    for fi in range(n_filters):
        delta = mp_after[:, fi].mean() - mp_before[:, fi].mean()
        try:
            _, pval = stats.wilcoxon(mp_after[:, fi], mp_before[:, fi],
                                     alternative="two-sided")
        except ValueError:
            pval = 1.0
        deltas.append(delta)
        pvals.append(pval)

    deltas      = np.array(deltas)
    pvals       = np.clip(pvals, 1e-300, 1.0)
    neg_log_p   = -np.log10(pvals)
    is_highlight = np.array([fi in highlight_filters for fi in range(n_filters)])

    fig, ax = plt.subplots(figsize=(8, 6))

    # Background filters
    ax.scatter(deltas[~is_highlight], neg_log_p[~is_highlight],
               s=30, c="#B0B0B0", alpha=0.5, edgecolors="none", zorder=2)

    # Highlighted filters
    for fi in highlight_filters:
        ax.scatter(deltas[fi], neg_log_p[fi], s=120, c="#E63946",
                   edgecolors="k", linewidths=1.2, zorder=4, marker="D")
        ax.annotate(f"F{fi}", (deltas[fi], neg_log_p[fi]),
                    textcoords="offset points", xytext=(8, 8),
                    fontsize=10, fontweight="bold", color="#E63946",
                    arrowprops=dict(arrowstyle="-", color="#E63946", lw=0.8))

    # Bonferroni threshold line
    bonf = -np.log10(0.05 / n_filters)
    ax.axhline(bonf, color="gray", linestyle="--", lw=0.8, alpha=0.6)
    ax.text(ax.get_xlim()[1] * 0.98, bonf + 0.3, "Bonferroni p < 0.05",
            ha="right", fontsize=7, fontstyle="italic", color="gray")
    ax.axvline(0, color="gray", lw=0.5, alpha=0.5)

    ax.set_xlabel("Δ mean activation (after − before optimization)", fontsize=11)
    ax.set_ylabel("−log₁₀(p-value)", fontsize=11)
    ax.set_title("Filter activation changes after boundary suppression\n"
                 "(conv_tower_block3, all filters)", fontsize=12, fontweight="bold")
    ax.grid(alpha=0.2)
    ax.legend(handles=[
        Line2D([0], [0], marker="D", color="w", markerfacecolor="#E63946",
               markersize=10, markeredgecolor="k", label="B2+CTCF-specific filters"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#B0B0B0",
               markersize=8, label="Other filters"),
    ], loc="upper left", fontsize=9)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, format="svg", bbox_inches="tight")
    plt.show()

    # Summary printout
    print(f"\nHighlighted filters:")
    for fi in highlight_filters:
        print(f"  Filter {fi}: Δ={deltas[fi]:+.2f}, p={pvals[fi]:.2e}, "
              f"-log10(p)={neg_log_p[fi]:.1f}")
    rank_order = np.argsort(deltas)[::-1]
    print(f"\nRank of highlighted filters (by Δ, descending):")
    for fi in highlight_filters:
        rank = int(np.where(rank_order == fi)[0][0]) + 1
        print(f"  Filter {fi}: rank {rank}/{n_filters}")