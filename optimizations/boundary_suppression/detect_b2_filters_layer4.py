"""
Identify B2-detecting filters at conv_tower_block3 (layer 4), then compare
their activations across:
  1) 300 SINE B2 + CTCF sequences (220bp, padded)
  2) "Before" optimization slices (2048bp) — init sequences at the central bin
  3) "After" optimization slices (2048bp) — optimized/suppressed sequences

The before/after slices are loaded directly from slice .pt files (2048bp).

Usage from notebook:
    from detect_b2_filters_layer4 import run_b2_filter_tracking
    run_b2_filter_tracking(
        model, g2, trunk_layers,
        coord_df=coord_df,
        init_seq_path="/path/to/init/",
        slice_path="/path/to/slices/",
    )
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from collections import OrderedDict
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

OUTPUT_DIR = "b2_filter_tracking"
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
BATCH_SIZE_220 = 64
BATCH_SIZE_2048 = 16

INDEX_TO_BASE = {0: "A", 1: "C", 2: "G", 3: "T"}


# ═══════════════════════════════════════════════════════════════════════════════
# PARTIAL FORWARD PASS (works for any input length)
# ═══════════════════════════════════════════════════════════════════════════════

def partial_forward_to_layer(sequences, trunk_layers, target_layer_idx,
                              batch_size=BATCH_SIZE_220, device=DEVICE):
    """
    Run sequences through trunk up to target_layer_idx.
    sequences: list of numpy arrays (4, L) — can be any length.

    Returns:
        acts: np.ndarray (n_seqs, n_filters, spatial_len)
    """
    layer_list = list(trunk_layers.values())
    all_acts = []

    for i in range(0, len(sequences), batch_size):
        batch = np.stack(sequences[i:i + batch_size])
        x = torch.tensor(batch, dtype=torch.float32).to(device)

        with torch.no_grad():
            for li in range(target_layer_idx + 1):
                try:
                    x = layer_list[li](x)
                except RuntimeError as e:
                    print(f"  Error at layer {li}: {e}")
                    break

        all_acts.append(x.cpu().numpy())

    return np.concatenate(all_acts, axis=0)


def partial_forward_tensors(tensors, trunk_layers, target_layer_idx,
                             batch_size=BATCH_SIZE_2048, device=DEVICE):
    """
    Same as above but input is a list of torch tensors (4, L) or (1, 4, L).
    """
    layer_list = list(trunk_layers.values())
    all_acts = []

    for i in range(0, len(tensors), batch_size):
        batch = tensors[i:i + batch_size]
        # Stack into (B, 4, L)
        stacked = []
        for t in batch:
            if t.dim() == 3:
                t = t.squeeze(0)
            stacked.append(t)
        x = torch.stack(stacked).to(device)

        with torch.no_grad():
            for li in range(target_layer_idx + 1):
                try:
                    x = layer_list[li](x)
                except RuntimeError as e:
                    print(f"  Error at layer {li}: {e}")
                    break

        all_acts.append(x.cpu().numpy())

    return np.concatenate(all_acts, axis=0)


# ═══════════════════════════════════════════════════════════════════════════════
# LOAD BEFORE/AFTER SLICES
# ═══════════════════════════════════════════════════════════════════════════════

def load_slices(coord_df, path_template, has_fold=True):
    """
    Load slice tensors (2048bp, the "after" optimization sequences).

    Args:
        coord_df: DataFrame with chrom, centered_start, centered_end, [fold]
        path_template: base path to slice files

    Returns:
        list of tensors (4, 2048), list of row indices
    """
    slices = []
    indices = []

    for idx, row in coord_df.iterrows():
        chrom = row["chrom"]
        start = row["centered_start"]
        end = row["centered_end"]

        if has_fold and "fold" in row.index:
            fold = row["fold"]
            fpath = f"{path_template}/fold{fold}/{chrom}_{start}_{end}_slice.pt"
        else:
            fpath = f"{path_template}/{chrom}_{start}_{end}_slice.pt"

        try:
            t = torch.load(fpath, weights_only=True, map_location="cpu")
            if t.dim() == 3:
                t = t.squeeze(0)  # (4, L)
            slices.append(t)
            indices.append(idx)
        except FileNotFoundError:
            continue

    print(f"  Loaded {len(slices)} slices (of {len(coord_df)} requested)")
    if slices:
        print(f"  Shape: {slices[0].shape}")
    return slices, indices


def load_init_bins(coord_df, init_seq_path, slice_offset=256, cropping=64,
                    bin_size=2048, has_fold=True):
    """
    Load the central 2048bp bin from init (full 524kb) sequences.
    This is the "before" optimization sequence at the insertion site.

    Path pattern matches: {init_seq_path}/fold{fold}/{chrom}_{start}_{end}_X.pt
    """
    bins = []
    indices = []

    edit_start = (slice_offset + cropping) * bin_size
    edit_end = edit_start + bin_size

    for idx, row in coord_df.iterrows():
        chrom = row["chrom"]
        start = row["centered_start"]
        end = row["centered_end"]

        if has_fold and "fold" in row.index:
            fold = row["fold"]
            fpath = f"{init_seq_path}/fold{fold}/{chrom}_{start}_{end}_X.pt"
        else:
            fpath = f"{init_seq_path}/{chrom}_{start}_{end}_X.pt"

        try:
            X = torch.load(fpath, weights_only=True, map_location="cpu")
            if X.dim() == 3:
                X = X.squeeze(0)  # (4, 524288)
            central_bin = X[:, edit_start:edit_end]  # (4, 2048)
            bins.append(central_bin)
            indices.append(idx)
        except FileNotFoundError:
            continue

    print(f"  Loaded {len(bins)} init bins (of {len(coord_df)} requested)")
    if bins:
        print(f"  Shape: {bins[0].shape}")
    return bins, indices


# ═══════════════════════════════════════════════════════════════════════════════
# IDENTIFY B2-SPECIFIC FILTERS
# ═══════════════════════════════════════════════════════════════════════════════

def identify_b2_filters(acts_b2ctcf, acts_ctcf, acts_b2no, top_k=10):
    """
    Find filters at the target layer that are most specific to B2+CTCF.

    Returns DataFrame with filter stats and list of top filter indices.
    """
    # Max-pool over spatial → (n_seqs, n_filters)
    mp1 = acts_ctcf.max(axis=2)
    mp2 = acts_b2ctcf.max(axis=2)
    mp3 = acts_b2no.max(axis=2)

    m1, m2, m3 = mp1.mean(0), mp2.mean(0), mp3.mean(0)
    n_filters = len(m1)

    df = pd.DataFrame({
        "filter_idx": range(n_filters),
        "CTCF": m1,
        "B2CTCF": m2,
        "B2noCTCF": m3,
        "B2CTCF_vs_CTCF": m2 - m1,
        "B2CTCF_vs_B2no": m2 - m3,
        # Specificity: high in B2+CTCF, low in both others
        "B2CTCF_specificity": m2 - (m1 + m3) / 2,
    })

    # Top B2+CTCF overall
    top_b2 = df.nlargest(top_k, "B2CTCF")["filter_idx"].tolist()

    # Top B2+CTCF-specific (higher than both other groups)
    top_specific = df.nlargest(top_k, "B2CTCF_specificity")["filter_idx"].tolist()

    print(f"\n  Top {top_k} filters by B2+CTCF activation:")
    print(df.nlargest(top_k, "B2CTCF")[
        ["filter_idx", "CTCF", "B2CTCF", "B2noCTCF", "B2CTCF_specificity"]
    ].to_string(index=False))

    print(f"\n  Top {top_k} filters by B2+CTCF specificity:")
    print(df.nlargest(top_k, "B2CTCF_specificity")[
        ["filter_idx", "CTCF", "B2CTCF", "B2noCTCF", "B2CTCF_specificity"]
    ].to_string(index=False))

    # Union of both sets
    selected = sorted(set(top_b2 + top_specific))
    print(f"\n  Selected filters (union): {selected}")

    return df, selected


# ═══════════════════════════════════════════════════════════════════════════════
# COMPARE ACTIVATIONS: B2+CTCF vs BEFORE vs AFTER
# ═══════════════════════════════════════════════════════════════════════════════

def compare_activations(acts_b2, acts_before, acts_after, selected_filters,
                         save_dir=OUTPUT_DIR):
    """
    Compare activation distributions for selected filters across:
      - 300 B2+CTCF (220bp reference)
      - Before optimization (2048bp init bins)
      - After optimization (2048bp slices)
    """
    # Max-pool over spatial
    mp_b2 = acts_b2.max(axis=2)         # (300, filters)
    mp_before = acts_before.max(axis=2)  # (N, filters)
    mp_after = acts_after.max(axis=2)    # (N, filters)

    records = []
    for fi in selected_filters:
        records.append({
            "filter": fi,
            "B2CTCF_mean": mp_b2[:, fi].mean(),
            "B2CTCF_std": mp_b2[:, fi].std(),
            "before_mean": mp_before[:, fi].mean(),
            "before_std": mp_before[:, fi].std(),
            "after_mean": mp_after[:, fi].mean(),
            "after_std": mp_after[:, fi].std(),
            "delta_optimization": mp_after[:, fi].mean() - mp_before[:, fi].mean(),
        })

    summary = pd.DataFrame(records)
    summary.to_csv(f"{save_dir}/filter_activation_comparison.csv", index=False)

    print("\n  Activation comparison:")
    print(summary.to_string(index=False))

    # ── Plot: grouped bar chart ──────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(max(8, len(selected_filters) * 1.2), 5))
    x = np.arange(len(selected_filters))
    w = 0.25

    ax.bar(x - w, summary["B2CTCF_mean"], w, label="B2+CTCF (220bp ref)",
           color="#F4A261", edgecolor="k", lw=0.5)
    ax.bar(x, summary["before_mean"], w, label="Before optimization",
           color="#A8DADC", edgecolor="k", lw=0.5)
    ax.bar(x + w, summary["after_mean"], w, label="After optimization",
           color="#E63946", edgecolor="k", lw=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels([f"F{f}" for f in selected_filters], fontsize=8)
    ax.set_ylabel("Max-pooled activation")
    ax.set_title("B2-detecting filter activations: reference vs before/after optimization")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    fig.savefig(f"{save_dir}/activation_comparison_bars.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ── Plot: per-filter violin/box ──────────────────────────────────────
    n_filters = len(selected_filters)
    n_cols = min(4, n_filters)
    n_rows = (n_filters + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3.5 * n_rows))
    if n_filters == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for i, fi in enumerate(selected_filters):
        ax = axes[i]
        data = [mp_b2[:, fi], mp_before[:, fi], mp_after[:, fi]]
        labels = ["B2+CTCF\n(ref)", "Before\nopt", "After\nopt"]
        colors = ["#F4A261", "#A8DADC", "#E63946"]

        bp = ax.boxplot(data, labels=labels, patch_artist=True, widths=0.6)
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        ax.set_title(f"Filter {fi}", fontsize=10)
        ax.grid(alpha=0.3, axis="y")

    # Hide unused axes
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle("Per-filter activation distributions", fontsize=12, y=1.02)
    plt.tight_layout()
    fig.savefig(f"{save_dir}/activation_comparison_boxes.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ── Plot: before vs after scatter per sample ─────────────────────────
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
    if n_filters == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for i, fi in enumerate(selected_filters):
        ax = axes[i]
        n = min(len(mp_before), len(mp_after))
        ax.scatter(mp_before[:n, fi], mp_after[:n, fi], s=10, alpha=0.5, c="#457B9D")
        lims = [min(mp_before[:n, fi].min(), mp_after[:n, fi].min()) - 0.5,
                max(mp_before[:n, fi].max(), mp_after[:n, fi].max()) + 0.5]
        ax.plot(lims, lims, "k--", lw=0.5, alpha=0.5)
        ax.set_xlabel("Before")
        ax.set_ylabel("After")
        ax.set_title(f"Filter {fi}", fontsize=10)
        ax.grid(alpha=0.3)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle("Before vs After optimization (per sample)", fontsize=12, y=1.02)
    plt.tight_layout()
    fig.savefig(f"{save_dir}/before_vs_after_scatter.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    return summary


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def run_b2_filter_tracking(model, g1, g2, g3, trunk_layers,
                            coord_df, init_seq_path, slice_path,
                            target_layer_idx=4, top_k=10,
                            has_fold=True):
    """
    Full pipeline:
    1. Identify B2-detecting filters at target layer using 220bp groups
    2. Load before/after slices
    3. Run partial forward on all three sets
    4. Compare activations

    Args:
        model: loaded AkitaV2
        g1, g2, g3: 220bp sequence groups (lists of numpy arrays)
        trunk_layers: from get_trunk_layers(model)
        coord_df: DataFrame for before/after sequences
        init_seq_path: path to init X tensors
        slice_path: path to optimized slice tensors
        target_layer_idx: which layer to analyze (default 4 = conv_tower_block3)
        has_fold: whether paths include fold subdirectory
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    layer_names = list(trunk_layers.keys())
    target_name = layer_names[target_layer_idx]
    print(f"Target layer: {target_name} (idx={target_layer_idx})")

    # ── Step 1: Activations for 220bp groups ─────────────────────────────
    print("\n" + "=" * 65)
    print("Step 1 — 220bp reference group activations")
    print("=" * 65)

    print("  Computing activations for CTCF group...")
    acts_ctcf = partial_forward_to_layer(g1, trunk_layers, target_layer_idx,
                                          batch_size=BATCH_SIZE_220)
    print(f"    Shape: {acts_ctcf.shape}")

    print("  Computing activations for B2+CTCF group...")
    acts_b2ctcf = partial_forward_to_layer(g2, trunk_layers, target_layer_idx,
                                            batch_size=BATCH_SIZE_220)
    print(f"    Shape: {acts_b2ctcf.shape}")

    print("  Computing activations for B2 noCTCF group...")
    acts_b2no = partial_forward_to_layer(g3, trunk_layers, target_layer_idx,
                                          batch_size=BATCH_SIZE_220)
    print(f"    Shape: {acts_b2no.shape}")

    # ── Step 2: Identify B2-specific filters ─────────────────────────────
    print("\n" + "=" * 65)
    print("Step 2 — Identify B2-detecting filters")
    print("=" * 65)

    filter_df, selected = identify_b2_filters(
        acts_b2ctcf, acts_ctcf, acts_b2no, top_k=top_k
    )
    filter_df.to_csv(f"{OUTPUT_DIR}/layer4_filter_stats.csv", index=False)

    # Save B2+CTCF activations for selected filters
    mp_b2 = acts_b2ctcf.max(axis=2)  # (300, 128)
    np.save(f"{OUTPUT_DIR}/b2ctcf_activations_layer4.npy", mp_b2)

    # ── Step 3: Load before/after slices ─────────────────────────────────
    print("\n" + "=" * 65)
    print("Step 3 — Load before/after optimization sequences")
    print("=" * 65)

    print("  Loading 'before' (init central bins)...")
    before_tensors, before_idx = load_init_bins(
        coord_df, init_seq_path, has_fold=has_fold
    )

    print("  Loading 'after' (optimized slices)...")
    after_tensors, after_idx = load_slices(
        coord_df, slice_path, has_fold=has_fold
    )

    # Use only samples present in both
    common_idx = sorted(set(before_idx) & set(after_idx))
    before_map = {idx: t for idx, t in zip(before_idx, before_tensors)}
    after_map = {idx: t for idx, t in zip(after_idx, after_tensors)}
    before_common = [before_map[i] for i in common_idx]
    after_common = [after_map[i] for i in common_idx]
    print(f"  Common samples: {len(common_idx)}")

    # ── Step 4: Activations for before/after ─────────────────────────────
    print("\n" + "=" * 65)
    print("Step 4 — Compute activations for before/after")
    print("=" * 65)

    print("  Running 'before' through trunk...")
    acts_before = partial_forward_tensors(
        before_common, trunk_layers, target_layer_idx,
        batch_size=BATCH_SIZE_2048
    )
    print(f"    Shape: {acts_before.shape}")

    print("  Running 'after' through trunk...")
    acts_after = partial_forward_tensors(
        after_common, trunk_layers, target_layer_idx,
        batch_size=BATCH_SIZE_2048
    )
    print(f"    Shape: {acts_after.shape}")

    # ── Step 5: Compare ──────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("Step 5 — Compare activations")
    print("=" * 65)

    summary = compare_activations(
        acts_b2ctcf, acts_before, acts_after, selected
    )

    print(f"\n{'='*65}")
    print(f"Done! Outputs in: {OUTPUT_DIR}/")
    print(f"{'='*65}")

    return filter_df, selected, summary