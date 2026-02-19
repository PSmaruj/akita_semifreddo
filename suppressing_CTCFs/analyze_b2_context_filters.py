"""
Identify first-layer filters that are highly activated by SINE B2+CTCF
(group 2) but NOT by isolated CTCFs (group 1).

These are "SINE B2 context" filters — they detect the flanking repeat
context that may contribute to CTCF silencing.

For each such filter, extract the 15bp subsequences (=kernel size) that
maximally activate it, build a PWM, and check for homopolymer enrichment.

Prerequisites: run filter_activation_analysis.py first, or have the
model and sequences ready.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from collections import OrderedDict
import os

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIG — adjust to match your setup
# ═══════════════════════════════════════════════════════════════════════════════
OUTPUT_DIR = "filter_activation_analysis/b2_context_filters"
KERNEL_SIZE = 15  # conv_block_1 kernel size
TOP_K_FILTERS = 20  # top N filters per group for comparison
N_TOP_SEQS = 50  # number of top-activating subsequences per filter for logo

# These should already be in memory if you ran the main script.
# Otherwise, reload:
#   from filter_activation_analysis import *
#   g1, g2, g3 = prepare_all_groups()
#   model = load_model()


# ═══════════════════════════════════════════════════════════════════════════════
# 1. IDENTIFY B2-CONTEXT FILTERS
# ═══════════════════════════════════════════════════════════════════════════════

def identify_b2_context_filters(first_layer_csv, top_k=TOP_K_FILTERS):
    """
    Find filters in top-k for B2+CTCF (group 2) but NOT in top-k for
    isolated CTCF (group 1).

    Also find filters in top-k for B2+CTCF AND B2 noCTCF but NOT CTCF
    (= general SINE B2 filters).
    """
    df = pd.read_csv(first_layer_csv)

    top_ctcf = set(df.nlargest(top_k, "CTCF")["filter_idx"])
    top_b2ctcf = set(df.nlargest(top_k, "SINEB2_CTCF")["filter_idx"])
    top_b2no = set(df.nlargest(top_k, "SINEB2_noCTCF")["filter_idx"])

    # Filters high in B2+CTCF but not in isolated CTCF
    b2_context = top_b2ctcf - top_ctcf
    # Subset: also high in B2 noCTCF → general SINE B2 filters
    general_b2 = b2_context & top_b2no
    # Subset: high in B2+CTCF only → specific to CTCF-containing B2s
    b2ctcf_specific = b2_context - top_b2no

    print(f"Filters in top-{top_k} for B2+CTCF but NOT for CTCF: {len(b2_context)}")
    print(f"  Of these, also top for B2 noCTCF (general B2):     {len(general_b2)}")
    print(f"  Of these, B2+CTCF specific (not in B2 noCTCF):     {len(b2ctcf_specific)}")
    print(f"\n  General B2 filter indices:      {sorted(general_b2)}")
    print(f"  B2+CTCF specific filter indices: {sorted(b2ctcf_specific)}")

    # Show activation values for these filters
    for label, fset in [("General B2", general_b2), ("B2+CTCF specific", b2ctcf_specific)]:
        if fset:
            sub = df[df["filter_idx"].isin(fset)].sort_values("SINEB2_CTCF", ascending=False)
            print(f"\n  {label} filters — mean activations:")
            print(sub[["filter_idx", "CTCF", "SINEB2_CTCF", "SINEB2_noCTCF"]].to_string(index=False))

    return sorted(b2_context), sorted(general_b2), sorted(b2ctcf_specific)


# ═══════════════════════════════════════════════════════════════════════════════
# 2. EXTRACT MAXIMALLY ACTIVATING SUBSEQUENCES
# ═══════════════════════════════════════════════════════════════════════════════

def get_first_layer_conv(model):
    """Extract the Conv1d weights from conv_block_1."""
    for module in model.conv_block_1.modules():
        if isinstance(module, nn.Conv1d):
            return module
    raise ValueError("No Conv1d found in conv_block_1")


def scan_filter_activations(sequences, conv_layer, filter_idx, device):
    """
    For a given filter, scan all sequences and return per-position activations.

    Returns:
        all_activations: list of (seq_idx, position, activation_value, subseq_onehot)
    """
    weight = conv_layer.weight[filter_idx:filter_idx+1]  # (1, 4, kernel_size)
    bias = conv_layer.bias[filter_idx:filter_idx+1] if conv_layer.bias is not None else None
    k = weight.shape[2]

    results = []
    for seq_idx, seq_oh in enumerate(sequences):
        # seq_oh: (4, seq_len)
        t = torch.tensor(seq_oh, dtype=torch.float32).unsqueeze(0).to(device)  # (1, 4, L)
        with torch.no_grad():
            # Manual convolution for this single filter
            act = torch.nn.functional.conv1d(t, weight.to(device),
                                              bias=bias.to(device) if bias is not None else None)
            act = act.squeeze().cpu().numpy()  # (L - k + 1,)

        for pos in range(len(act)):
            results.append((seq_idx, pos, act[pos], seq_oh[:, pos:pos+k]))

    return results


def get_top_activating_subseqs(sequences, conv_layer, filter_idx, device,
                                n_top=N_TOP_SEQS):
    """Get the top-N subsequences that maximally activate a given filter."""
    results = scan_filter_activations(sequences, conv_layer, filter_idx, device)
    # Sort by activation value (descending)
    results.sort(key=lambda x: x[2], reverse=True)
    return results[:n_top]


# ═══════════════════════════════════════════════════════════════════════════════
# 3. BUILD PWM & CONSENSUS
# ═══════════════════════════════════════════════════════════════════════════════

def build_pwm(top_subseqs):
    """
    Build a position weight matrix from top-activating subsequences.

    Args:
        top_subseqs: list of (seq_idx, pos, activation, onehot_array)
                     where onehot_array is (4, kernel_size)
    Returns:
        pwm: (4, kernel_size) normalized frequencies
        consensus: string
    """
    arrays = [item[3] for item in top_subseqs]
    stacked = np.stack(arrays)  # (N, 4, K)
    # Sum across sequences
    pfm = stacked.sum(axis=0)  # (4, K)
    # Normalize to frequencies
    pwm = pfm / pfm.sum(axis=0, keepdims=True)

    # Consensus
    bases = "ACGT"
    consensus = "".join(bases[i] for i in pwm.argmax(axis=0))

    return pwm, consensus


def check_homopolymers(consensus, min_run=3):
    """Check for homopolymer runs in consensus sequence."""
    runs = []
    current_base = consensus[0]
    current_len = 1

    for b in consensus[1:]:
        if b == current_base:
            current_len += 1
        else:
            if current_len >= min_run:
                runs.append((current_base, current_len))
            current_base = b
            current_len = 1
    if current_len >= min_run:
        runs.append((current_base, current_len))

    return runs


def information_content(pwm):
    """Compute per-position information content (bits)."""
    ic = np.zeros(pwm.shape[1])
    for j in range(pwm.shape[1]):
        for i in range(4):
            if pwm[i, j] > 0:
                ic[j] += pwm[i, j] * np.log2(pwm[i, j] / 0.25)
    return ic


# ═══════════════════════════════════════════════════════════════════════════════
# 4. SEQUENCE LOGO PLOTTING
# ═══════════════════════════════════════════════════════════════════════════════

def plot_sequence_logo(pwm, title="", save_path=None):
    """
    Plot a sequence logo from a PWM.
    """
    bases = "ACGT"
    colors = {"A": "#109648", "C": "#255C99", "G": "#F7B32B", "T": "#D62839"}

    ic = information_content(pwm)
    k = pwm.shape[1]

    fig, ax = plt.subplots(figsize=(max(4, k * 0.5), 2.5))

    for j in range(k):
        # Sort bases by frequency at this position
        order = np.argsort(pwm[:, j])
        y_offset = 0
        for idx in order:
            base = bases[idx]
            height = pwm[idx, j] * ic[j]
            if height < 0.01:
                y_offset += height
                continue
            ax.text(j + 0.5, y_offset + height / 2, base,
                    ha="center", va="center",
                    fontsize=max(8, height * 20),
                    fontweight="bold",
                    color=colors[base],
                    fontfamily="monospace",
                    path_effects=[pe.withStroke(linewidth=0.5, foreground="black")])
            y_offset += height

    ax.set_xlim(0, k)
    ax.set_ylim(0, 2)
    ax.set_ylabel("bits")
    ax.set_xlabel("Position")
    ax.set_xticks(np.arange(k) + 0.5)
    ax.set_xticklabels(range(1, k + 1), fontsize=7)
    ax.set_title(title, fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# 5. MAIN ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════

def analyze_b2_context_filters(model, g1, g2, g3, device,
                                first_layer_csv="filter_activation_analysis/first_layer_filter_means.csv"):
    """
    Full analysis pipeline:
    1. Identify B2-context filters
    2. For each, extract top-activating subsequences from each group
    3. Build PWMs, consensus, check homopolymers
    4. Plot logos
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Step 1: identify filters
    print("=" * 65)
    print("Identifying B2-context filters")
    print("=" * 65)
    all_b2_context, general_b2, b2ctcf_specific = identify_b2_context_filters(first_layer_csv)

    # Also get full top-20 for B2+CTCF
    df = pd.read_csv(first_layer_csv)
    top_b2ctcf_all = sorted(df.nlargest(TOP_K_FILTERS, "SINEB2_CTCF")["filter_idx"])
    top_ctcf_set = set(df.nlargest(TOP_K_FILTERS, "CTCF")["filter_idx"])
    top_b2no_set = set(df.nlargest(TOP_K_FILTERS, "SINEB2_noCTCF")["filter_idx"])

    print(f"\nAnalyzing ALL top-{TOP_K_FILTERS} filters for B2+CTCF: {top_b2ctcf_all}")

    # Step 2-4: analyze each filter
    conv1 = get_first_layer_conv(model)
    print(f"\nFirst-layer Conv1d: {conv1}")
    print(f"  Weight shape: {conv1.weight.shape}")
    print(f"  Has bias: {conv1.bias is not None}")

    summary_records = []

    for filt_idx in top_b2ctcf_all:
        # Categorize
        in_ctcf = filt_idx in top_ctcf_set
        in_b2no = filt_idx in top_b2no_set
        if in_ctcf and in_b2no:
            category = "shared_all_3"
        elif in_ctcf and not in_b2no:
            category = "shared_CTCF_groups"
        elif not in_ctcf and in_b2no:
            category = "general_B2"
        else:
            category = "B2CTCF_specific"
        print(f"\n{'─'*50}")
        print(f"Filter {filt_idx} ({category})")
        print(f"{'─'*50}")

        # Get top-activating subsequences from group 2 (B2+CTCF)
        top_subs = get_top_activating_subseqs(g2, conv1, filt_idx, device, n_top=N_TOP_SEQS)
        pwm, consensus = build_pwm(top_subs)
        ic = information_content(pwm)
        homopolymers = check_homopolymers(consensus)
        mean_act = np.mean([s[2] for s in top_subs])

        print(f"  Consensus:     {consensus}")
        print(f"  Mean IC:       {ic.mean():.3f} bits")
        print(f"  Max IC:        {ic.max():.3f} bits")
        print(f"  Mean act:      {mean_act:.3f}")

        if homopolymers:
            print(f"  Homopolymers:  {', '.join(f'{b}×{n}' for b, n in homopolymers)}")
        else:
            print(f"  Homopolymers:  none")

        # Also get top subsequences from group 1 (CTCF) for comparison
        top_subs_ctcf = get_top_activating_subseqs(g1, conv1, filt_idx, device, n_top=N_TOP_SEQS)
        _, consensus_ctcf = build_pwm(top_subs_ctcf)
        mean_act_ctcf = np.mean([s[2] for s in top_subs_ctcf])
        print(f"  Consensus (CTCF group): {consensus_ctcf}")
        print(f"  Mean act (CTCF group):  {mean_act_ctcf:.3f}")

        # And group 3 (B2 noCTCF)
        top_subs_b2no = get_top_activating_subseqs(g3, conv1, filt_idx, device, n_top=N_TOP_SEQS)
        _, consensus_b2no = build_pwm(top_subs_b2no)
        mean_act_b2no = np.mean([s[2] for s in top_subs_b2no])
        print(f"  Consensus (B2 noCTCF):  {consensus_b2no}")
        print(f"  Mean act (B2 noCTCF):   {mean_act_b2no:.3f}")

        # Plot logo (from B2+CTCF group activations)
        plot_sequence_logo(
            pwm,
            title=f"Filter {filt_idx} ({category}) — top from B2+CTCF",
            save_path=f"{OUTPUT_DIR}/filter{filt_idx}_logo_B2CTCF.png"
        )

        # Plot logo from CTCF group for comparison
        pwm_ctcf, _ = build_pwm(top_subs_ctcf)
        plot_sequence_logo(
            pwm_ctcf,
            title=f"Filter {filt_idx} ({category}) — top from CTCF",
            save_path=f"{OUTPUT_DIR}/filter{filt_idx}_logo_CTCF.png"
        )

        summary_records.append({
            "filter_idx": filt_idx,
            "category": category,
            "consensus_B2CTCF": consensus,
            "consensus_CTCF": consensus_ctcf,
            "consensus_B2noCTCF": consensus_b2no,
            "mean_act_B2CTCF": mean_act,
            "mean_act_CTCF": mean_act_ctcf,
            "mean_act_B2noCTCF": mean_act_b2no,
            "mean_IC": ic.mean(),
            "homopolymers": "; ".join(f"{b}x{n}" for b, n in homopolymers) if homopolymers else "none",
        })

    # Summary table
    summary_df = pd.DataFrame(summary_records)
    summary_df.to_csv(f"{OUTPUT_DIR}/b2_context_filter_summary.csv", index=False)
    print(f"\n{'='*65}")
    print(f"Summary saved to {OUTPUT_DIR}/b2_context_filter_summary.csv")
    print(f"{'='*65}")
    print(summary_df.to_string(index=False))

    return summary_df


# ═══════════════════════════════════════════════════════════════════════════════
# Run from notebook:
#   from analyze_b2_context_filters import analyze_b2_context_filters
#   summary = analyze_b2_context_filters(model, g1, g2, g3, DEVICE)
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # If running standalone, need to set up model & sequences first
    from filter_activation_analysis import (
        prepare_all_groups, load_model, DEVICE
    )
    g1, g2, g3 = prepare_all_groups()
    model = load_model()
    analyze_b2_context_filters(model, g1, g2, g3, DEVICE)