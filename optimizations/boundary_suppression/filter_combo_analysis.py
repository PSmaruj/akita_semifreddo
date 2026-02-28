"""
Filter combination analysis for AkitaV2 early trunk layers.

Three complementary approaches:
  1) Co-activation analysis: which first-layer filter PAIRS fire together
     differentially across the 3 groups?
  2) Deeper-layer top filters: what input subsequences (longer receptive
     fields) maximally activate deeper filters, per group?
  3) First-layer → deeper-layer attribution: which first-layer filters
     contribute most to each top deeper-layer filter?

Groups:
  1) Strong CTCFs (padded to 220bp)
  2) SINE B2 + CTCF (padded to 220bp)
  3) SINE B2 no CTCF (padded to 220bp)

Prerequisites:
  - model, g1, g2, g3 loaded (from filter_activation_analysis.py)
  - trunk_layers extracted via get_trunk_layers(model)
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from collections import OrderedDict
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr
from itertools import combinations
import warnings
warnings.filterwarnings("ignore")

OUTPUT_DIR = "filter_combo_analysis"
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 64

GROUP_NAMES = ["CTCF", "B2+CTCF", "B2_noCTCF"]
GROUP_COLORS = ["#E63946", "#F4A261", "#457B9D"]


# ═══════════════════════════════════════════════════════════════════════════════
# PART 1: FIRST-LAYER CO-ACTIVATION ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════

def compute_coactivation_matrices(sequences, model, layer):
    """
    For each sequence, get first-layer activations and compute the
    filter-filter correlation matrix (which filters co-activate).

    Args:
        sequences: list of one-hot arrays (4, 220)
        model: loaded model
        layer: the first conv layer module

    Returns:
        per_position_acts: (n_seqs, n_filters, spatial_len) raw activations
        corr_matrix: (n_filters, n_filters) mean pairwise Pearson correlation
    """
    all_acts = []

    for i in range(0, len(sequences), BATCH_SIZE):
        batch = np.stack(sequences[i:i + BATCH_SIZE])
        t = torch.tensor(batch, dtype=torch.float32).to(DEVICE)
        with torch.no_grad():
            out = layer(t)  # (B, filters, spatial)
        all_acts.append(out.cpu().numpy())

    acts = np.concatenate(all_acts, axis=0)  # (N, filters, spatial)

    # Correlation: for each sequence, compute spatial correlation between
    # filter pairs, then average across sequences
    n_seqs, n_filters, spatial = acts.shape

    # Max-pool per sequence to get (N, filters) — simple co-activation
    maxpool = acts.max(axis=2)  # (N, filters)

    # Pearson correlation across sequences
    corr = np.corrcoef(maxpool.T)  # (filters, filters)

    return acts, maxpool, corr


def differential_coactivation(corr1, corr2, corr3, top_k=20):
    """
    Find filter pairs with the biggest difference in co-activation
    between groups.

    Returns DataFrame of top differential pairs.
    """
    n = corr1.shape[0]
    records = []

    for i, j in combinations(range(n), 2):
        c1, c2, c3 = corr1[i, j], corr2[i, j], corr3[i, j]
        records.append({
            "filter_i": i,
            "filter_j": j,
            "corr_CTCF": c1,
            "corr_B2CTCF": c2,
            "corr_B2noCTCF": c3,
            # Key contrasts
            "diff_B2CTCF_vs_CTCF": c2 - c1,
            "diff_B2CTCF_vs_B2no": c2 - c3,
            "diff_CTCF_vs_B2no": c1 - c3,
        })

    df = pd.DataFrame(records)
    return df


def plot_coactivation_diff(corr1, corr2, corr3, save_dir=OUTPUT_DIR):
    """Plot differential co-activation heatmaps."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    pairs = [
        (corr2 - corr1, "B2+CTCF minus CTCF"),
        (corr2 - corr3, "B2+CTCF minus B2 noCTCF"),
        (corr1 - corr3, "CTCF minus B2 noCTCF"),
    ]

    for ax, (diff, title) in zip(axes, pairs):
        vmax = np.abs(diff).max() * 0.8
        sns.heatmap(diff, cmap="RdBu_r", center=0, vmin=-vmax, vmax=vmax,
                    ax=ax, cbar_kws={"shrink": 0.6})
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Filter")
        ax.set_ylabel("Filter")

    plt.tight_layout()
    fig.savefig(f"{save_dir}/coactivation_diff_heatmaps.png", dpi=150,
                bbox_inches="tight")
    plt.close(fig)


def plot_top_differential_pairs(diff_df, contrast_col, title, n_top=15,
                                 save_path=None):
    """Bar chart of top differential filter pairs."""
    top = diff_df.nlargest(n_top, contrast_col)
    labels = [f"{r['filter_i']}-{r['filter_j']}" for _, r in top.iterrows()]

    fig, ax = plt.subplots(figsize=(8, 5))
    colors_map = {
        "diff_B2CTCF_vs_CTCF": "#F4A261",
        "diff_B2CTCF_vs_B2no": "#2A9D8F",
        "diff_CTCF_vs_B2no": "#E63946",
    }
    ax.barh(labels[::-1], top[contrast_col].values[::-1],
            color=colors_map.get(contrast_col, "#888"))
    ax.set_xlabel("Δ correlation")
    ax.set_title(title, fontsize=10)
    ax.axvline(0, color="k", lw=0.5)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
# PART 2: DEEPER-LAYER MAXIMALLY ACTIVATING SUBSEQUENCES
# ═══════════════════════════════════════════════════════════════════════════════

def get_receptive_field_size(layer_idx):
    """
    Approximate receptive field size at each layer.
    conv_block_1: kernel=15, pool=2 → RF=15, output spatial /= 2
    conv_tower blocks: kernel=5, pool=2 each

    After conv_block_1: RF ≈ 15
    After tower block 0: RF ≈ 15 + (5-1)*2 = 23, spatial /= 2
    After tower block 1: RF ≈ 23 + (5-1)*4 = 39
    After tower block 2: RF ≈ 39 + (5-1)*8 = 71
    After tower block 3: RF ≈ 71 + (5-1)*16 = 135
    After tower block 4: RF ≈ 135 + (5-1)*32 = 263 (> 220, full sequence)
    """
    rfs = [15, 23, 39, 71, 135, 220]
    if layer_idx < len(rfs):
        return rfs[layer_idx]
    return 220


def get_deeper_layer_activations(sequences, trunk_layers, target_layer_idx):
    """
    Run sequences through trunk up to target layer, returning per-position
    activations at that layer.

    Returns:
        acts: (n_seqs, n_filters, spatial_len)
    """
    layer_names = list(trunk_layers.keys())
    all_acts = []

    for i in range(0, len(sequences), BATCH_SIZE):
        batch = np.stack(sequences[i:i + BATCH_SIZE])
        x = torch.tensor(batch, dtype=torch.float32).to(DEVICE)

        with torch.no_grad():
            for li, (name, layer) in enumerate(trunk_layers.items()):
                try:
                    x = layer(x)
                except RuntimeError:
                    break
                if li == target_layer_idx:
                    all_acts.append(x.cpu().numpy())
                    break

    return np.concatenate(all_acts, axis=0)


def map_deeper_position_to_input(pos, layer_idx):
    """
    Map a position in a deeper layer back to approximate input coordinates.
    Each layer halves spatial (pool=2), so position p at layer L
    corresponds to input region [p * 2^L, (p+1) * 2^L + kernel_overlap].
    """
    # Total downsampling factor up to this layer
    # conv_block_1 has pool=2, each tower block has pool=2
    ds = 2 ** (layer_idx + 1)  # +1 for conv_block_1
    rf = get_receptive_field_size(layer_idx)

    center = pos * ds + ds // 2
    start = max(0, center - rf // 2)
    end = min(220, center + rf // 2)
    return start, end


def find_top_activating_regions(sequences, acts, filter_idx, layer_idx,
                                 n_top=30):
    """
    For a given deeper-layer filter, find the input regions that
    maximally activate it.

    Returns list of (seq_idx, input_start, input_end, activation, subseq_str)
    """
    n_seqs, n_filters, spatial = acts.shape
    results = []

    INDEX_TO_BASE = {0: "A", 1: "C", 2: "G", 3: "T"}

    for seq_idx in range(n_seqs):
        for pos in range(spatial):
            act_val = acts[seq_idx, filter_idx, pos]
            inp_start, inp_end = map_deeper_position_to_input(pos, layer_idx)
            # Get input subsequence
            oh = sequences[seq_idx]  # (4, 220)
            bases = "".join(INDEX_TO_BASE[oh[:, p].argmax()] for p in range(inp_start, inp_end))
            results.append((seq_idx, inp_start, inp_end, act_val, bases))

    results.sort(key=lambda x: x[3], reverse=True)
    return results[:n_top]


def analyze_deeper_layer(sequences_list, trunk_layers, target_layer_idx,
                          top_k_filters=10, n_top_seqs=30):
    """
    For a deeper layer, find group-differential filters and their
    maximally activating input regions.
    """
    layer_names = list(trunk_layers.keys())
    target_name = layer_names[target_layer_idx]
    print(f"\nAnalyzing layer: {target_name} (idx={target_layer_idx})")
    print(f"  Approx receptive field: {get_receptive_field_size(target_layer_idx)}bp")

    # Get activations for each group
    acts_list = []
    for gi, seqs in enumerate(sequences_list):
        acts = get_deeper_layer_activations(seqs, trunk_layers, target_layer_idx)
        acts_list.append(acts)
        print(f"  Group {gi+1} ({GROUP_NAMES[gi]}): acts shape {acts.shape}")

    # Max-pool to (n_seqs, n_filters)
    maxpooled = [a.max(axis=2) for a in acts_list]
    means = [m.mean(axis=0) for m in maxpooled]

    n_filters = means[0].shape[0]
    df = pd.DataFrame({
        "filter_idx": range(n_filters),
        "CTCF": means[0],
        "B2CTCF": means[1],
        "B2noCTCF": means[2],
    })

    # Find filters differentially activated by B2+CTCF
    df["B2CTCF_vs_CTCF"] = df["B2CTCF"] - df["CTCF"]
    df["B2CTCF_vs_B2no"] = df["B2CTCF"] - df["B2noCTCF"]

    # Top filters for B2+CTCF
    top_b2ctcf = df.nlargest(top_k_filters, "B2CTCF")

    print(f"\n  Top {top_k_filters} filters for B2+CTCF:")
    print(top_b2ctcf[["filter_idx", "CTCF", "B2CTCF", "B2noCTCF",
                        "B2CTCF_vs_CTCF"]].to_string(index=False))

    # For top differential filters, extract maximally activating regions
    results = {}
    for _, row in top_b2ctcf.head(5).iterrows():
        filt = int(row["filter_idx"])
        regions = find_top_activating_regions(
            sequences_list[1], acts_list[1], filt,
            target_layer_idx, n_top=n_top_seqs
        )
        results[filt] = regions
        if regions:
            # Show top 3 subsequences
            print(f"\n  Filter {filt} — top activating regions from B2+CTCF:")
            for si, start, end, act, seq in regions[:3]:
                print(f"    seq={si:3d}  [{start:3d}:{end:3d}]  act={act:.3f}  {seq}")

    return df, acts_list, results


# ═══════════════════════════════════════════════════════════════════════════════
# PART 3: FIRST-LAYER → DEEPER-LAYER ATTRIBUTION
# ═══════════════════════════════════════════════════════════════════════════════

def compute_layer1_to_deeper_attribution(sequences, trunk_layers,
                                          target_layer_idx, target_filter,
                                          n_seqs=50):
    """
    For a specific deeper-layer filter, compute which first-layer filters
    contribute most to its activation using input ablation.

    Method: for each first-layer filter, zero it out and measure the
    change in the target deeper-layer filter's max activation.

    Returns:
        attribution: (n_first_layer_filters,) importance scores
    """
    layer_list = list(trunk_layers.values())
    layer_names = list(trunk_layers.keys())

    # Baseline: run normally
    subset = sequences[:n_seqs]
    batch = np.stack(subset)
    t = torch.tensor(batch, dtype=torch.float32).to(DEVICE)

    with torch.no_grad():
        x = t
        for li, layer in enumerate(layer_list):
            try:
                x = layer(x)
            except RuntimeError:
                break
            if li == target_layer_idx:
                break
        baseline_act = x[:, target_filter, :].max(dim=1).values.mean().item()

    # Ablate each first-layer filter
    first_layer = layer_list[0]
    # Get first layer output
    with torch.no_grad():
        first_out = first_layer(t)  # (B, 128, spatial)

    n_filters_l1 = first_out.shape[1]
    attributions = np.zeros(n_filters_l1)

    for fi in range(n_filters_l1):
        with torch.no_grad():
            ablated = first_out.clone()
            ablated[:, fi, :] = 0  # zero out filter fi

            x = ablated
            for li in range(1, target_layer_idx + 1):
                try:
                    x = layer_list[li](x)
                except RuntimeError:
                    break

            ablated_act = x[:, target_filter, :].max(dim=1).values.mean().item()
            attributions[fi] = baseline_act - ablated_act  # positive = filter was important

    return attributions


def plot_attribution(attributions, target_layer, target_filter,
                      highlight_filters=None, save_path=None):
    """Bar chart of first-layer filter importance for a deeper filter."""
    fig, ax = plt.subplots(figsize=(14, 4))
    x = np.arange(len(attributions))
    colors = ["#E63946" if highlight_filters and i in highlight_filters
              else "#A8DADC" for i in x]
    ax.bar(x, attributions, color=colors, edgecolor="none", width=1.0)
    ax.set_xlabel("First-layer filter index")
    ax.set_ylabel("Δ activation (importance)")
    ax.set_title(f"First-layer attribution → {target_layer} filter {target_filter}")
    ax.axhline(0, color="k", lw=0.5)

    if highlight_filters:
        for fi in highlight_filters:
            if fi < len(attributions):
                ax.annotate(str(fi), (fi, attributions[fi]),
                           ha="center", va="bottom", fontsize=7, color="#E63946")

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def run_combo_analysis(model, g1, g2, g3, trunk_layers,
                        known_filters=None):
    """
    Full combo analysis pipeline.

    Args:
        model: loaded AkitaV2
        g1, g2, g3: sequence lists for the 3 groups
        trunk_layers: OrderedDict from get_trunk_layers()
        known_filters: dict of known filter indices, e.g.
            {"B2_conserved": 36, "BoxB": 122, "B2_CTCF": 104, "filter21": 21}
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if known_filters is None:
        known_filters = {"B2_conserved": 36, "BoxB": 122,
                          "B2_CTCF": 104, "filter21": 21}

    layer_names = list(trunk_layers.keys())
    first_layer = list(trunk_layers.values())[0]

    # ══════════════════════════════════════════════════════════════════════
    # PART 1: Co-activation at first layer
    # ══════════════════════════════════════════════════════════════════════
    print("=" * 65)
    print("PART 1 — First-layer co-activation")
    print("=" * 65)

    _, mp1, corr1 = compute_coactivation_matrices(g1, model, first_layer)
    _, mp2, corr2 = compute_coactivation_matrices(g2, model, first_layer)
    _, mp3, corr3 = compute_coactivation_matrices(g3, model, first_layer)

    plot_coactivation_diff(corr1, corr2, corr3)

    diff_df = differential_coactivation(corr1, corr2, corr3)
    diff_df.to_csv(f"{OUTPUT_DIR}/coactivation_pairs.csv", index=False)

    # Show co-activation between known filters
    print("\nCo-activation between known filters:")
    kf = known_filters
    for (n1, f1), (n2, f2) in combinations(kf.items(), 2):
        print(f"  {n1}({f1}) × {n2}({f2}):  "
              f"CTCF={corr1[f1,f2]:.3f}  "
              f"B2+CTCF={corr2[f1,f2]:.3f}  "
              f"B2no={corr3[f1,f2]:.3f}")

    # Top differential pairs
    for col, title in [
        ("diff_B2CTCF_vs_CTCF", "Pairs more co-activated in B2+CTCF than CTCF"),
        ("diff_B2CTCF_vs_B2no", "Pairs more co-activated in B2+CTCF than B2 noCTCF"),
    ]:
        top = diff_df.nlargest(10, col)
        print(f"\n{title}:")
        for _, r in top.iterrows():
            print(f"  filters {int(r['filter_i']):3d}-{int(r['filter_j']):3d}  "
                  f"Δ={r[col]:.3f}  "
                  f"(CTCF={r['corr_CTCF']:.3f}, B2+CTCF={r['corr_B2CTCF']:.3f}, "
                  f"B2no={r['corr_B2noCTCF']:.3f})")

        plot_top_differential_pairs(
            diff_df, col, title,
            save_path=f"{OUTPUT_DIR}/{col}_top_pairs.png"
        )

    # ══════════════════════════════════════════════════════════════════════
    # PART 2: Deeper-layer analysis
    # ══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 65)
    print("PART 2 — Deeper-layer top filters & activating regions")
    print("=" * 65)

    deeper_results = {}
    # Analyze layers 2-4 (conv_tower_block1 through block3)
    for target_idx in [2, 3, 4]:
        if target_idx >= len(layer_names):
            break
        df, acts, regions = analyze_deeper_layer(
            [g1, g2, g3], trunk_layers, target_idx
        )
        df.to_csv(f"{OUTPUT_DIR}/deeper_layer{target_idx}_filters.csv", index=False)
        deeper_results[target_idx] = (df, acts, regions)

    # ══════════════════════════════════════════════════════════════════════
    # PART 3: Attribution from first layer to deeper layers
    # ══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 65)
    print("PART 3 — First-layer → deeper-layer attribution")
    print("=" * 65)

    highlight = set(known_filters.values())

    for target_idx, (df, _, _) in deeper_results.items():
        # Top 3 B2+CTCF-differential filters at this depth
        top_filters = df.nlargest(3, "B2CTCF_vs_CTCF")["filter_idx"].tolist()

        for tf in top_filters:
            tf = int(tf)
            lname = layer_names[target_idx]
            print(f"\n  Attribution: {lname} filter {tf}")

            attr = compute_layer1_to_deeper_attribution(
                g2, trunk_layers, target_idx, tf, n_seqs=50
            )

            # Top contributing first-layer filters
            top_l1 = np.argsort(attr)[::-1][:10]
            print(f"    Top L1 contributors: {list(top_l1)}")
            print(f"    Importance scores:   {[f'{attr[i]:.4f}' for i in top_l1]}")

            # Check if known filters are important
            for name, fi in known_filters.items():
                rank = np.where(np.argsort(attr)[::-1] == fi)[0]
                if len(rank):
                    print(f"    {name} (filter {fi}): rank={rank[0]+1}, "
                          f"importance={attr[fi]:.4f}")

            plot_attribution(
                attr, lname, tf, highlight_filters=highlight,
                save_path=f"{OUTPUT_DIR}/attribution_{lname}_f{tf}.png"
            )

    print(f"\n{'='*65}")
    print(f"Done! Outputs in: {OUTPUT_DIR}/")
    print(f"{'='*65}")


# ═══════════════════════════════════════════════════════════════════════════════
# Run from notebook:
#   from filter_combo_analysis import run_combo_analysis
#   from filter_activation_analysis import get_trunk_layers
#   trunk_layers = get_trunk_layers(model)
#   run_combo_analysis(model, g1, g2, g3, trunk_layers)
# ═══════════════════════════════════════════════════════════════════════════════