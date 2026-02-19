"""
Mechanistic interpretability of AkitaV2 — early trunk layers.

Partial forward pass through conv_block_1 + conv_tower layers on short
(220bp) sequences. We stop before pooling kills the spatial dimension.

  220bp → conv_block_1 (pool2) → 110
        → conv_tower layer 0 (pool2) → 55
        → conv_tower layer 1 (pool2) → 27
        → conv_tower layer 2 (pool2) → 13
        → conv_tower layer 3 (pool2) → 6
        → conv_tower layer 4 (pool2) → 3
        → conv_tower layer 5 (pool2) → 1  (stop here)

Groups:
  1) Strong CTCFs (±30bp flanks, padded to 220bp with background)
  2) SINE B2 (B2_Mm2) containing CTCF motifs (padded to 220bp)
  3) SINE B2 (B2_Mm2) without CTCF motifs (padded to 220bp)

Analysis:
  A) Top-activated filters per group + overlap (per layer)
  B) Layer-by-layer cosine / Spearman similarity
  C) CKA across layers
  D) Filter overlap evolution across depth

Usage:
  Uncomment the SeqNN import below, then:
    python filter_activation_analysis.py
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from collections import OrderedDict
from Bio import SeqIO
from pysam import FastaFile
import matplotlib.pyplot as plt
from scipy.spatial.distance import cosine
from scipy.stats import spearmanr
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

sys.path.append(os.path.abspath("/home1/smaruj/pytorch_akita/"))
from akita_model.model import SeqNN



# ═══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════════════════
GENOME_FASTA = "/project2/fudenber_735/genomes/mm10/mm10.fa"
BACKGROUND_FASTA = (
    "/project2/fudenber_735/smaruj/sequence_design/"
    "ledidi_semifreddo_akita/background_generation/"
    "background_sequences_scd30_totvar1300.fasta"
)
CTCF_TSV = (
    "/home1/smaruj/akitaV2-analyses/input_data/select_top20percent/"
    "output/CTCFs_jaspar_filtered_mm10_top20percent.tsv"
)
SINEB2_WITH_CTCF = "sineB2_with_ctcf_300.tsv"
SINEB2_NO_CTCF = "sineB2_no_ctcf_300.tsv"

MODEL_PATH = (
    "/home1/smaruj/pytorch_akita/models/finetuned/mouse/"
    "Hsieh2019_mESC/checkpoints/"
    "Akita_v2_mouse_Hsieh2019_mESC_model0_finetuned.pth"
)

TARGET_LEN = 220   # pad all sequences to this
CTCF_FLANK = 30
N_SAMPLES = 300
TOP_K = 20
OUTPUT_DIR = "filter_activation_analysis"
BATCH_SIZE = 64    # 220bp sequences are tiny, can use big batches

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# ═══════════════════════════════════════════════════════════════════════════════
# MODEL — adjust import
# ═══════════════════════════════════════════════════════════════════════════════
# sys.path.append("/home1/smaruj/pytorch_akita")
# from akita_model.model import SeqNN


def load_model():
    model = SeqNN()
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    print(f"Model loaded on {DEVICE}")
    return model


# ═══════════════════════════════════════════════════════════════════════════════
# 1. SEQUENCE EXTRACTION & PADDING
# ═══════════════════════════════════════════════════════════════════════════════

def one_hot_encode(seq):
    mapping = {"A": 0, "C": 1, "G": 2, "T": 3}
    enc = np.zeros((4, len(seq)), dtype=np.float32)
    for i, b in enumerate(seq.upper()):
        if b in mapping:
            enc[mapping[b], i] = 1.0
        else:
            enc[:, i] = 0.25
    return enc


def load_background_sequence(path):
    return str(next(SeqIO.parse(path, "fasta")).seq)


def pad_to_target(seq, target_len, bg_seq, bg_offset=0):
    if len(seq) >= target_len:
        excess = len(seq) - target_len
        s = excess // 2
        return seq[s:s + target_len], bg_offset
    pad_total = target_len - len(seq)
    pad_left = pad_total // 2
    pad_right = pad_total - pad_left
    bg_len = len(bg_seq)
    left = "".join(bg_seq[(bg_offset + i) % bg_len] for i in range(pad_left))
    bg_offset = (bg_offset + pad_left) % bg_len
    right = "".join(bg_seq[(bg_offset + i) % bg_len] for i in range(pad_right))
    bg_offset = (bg_offset + pad_right) % bg_len
    return left + seq + right, bg_offset


def extract_sequences(df, genome, bg_seq, chrom_col, start_col, end_col,
                      flank=0, target_len=TARGET_LEN, n_samples=N_SAMPLES):
    seqs = []
    bg_off = 0
    for _, row in df.head(n_samples).iterrows():
        chrom = str(row[chrom_col])
        start = max(0, int(row[start_col]) - flank)
        end = int(row[end_col]) + flank
        try:
            seq = genome.fetch(chrom, start, end)
        except (KeyError, ValueError):
            continue
        padded, bg_off = pad_to_target(seq, target_len, bg_seq, bg_off)
        assert len(padded) == target_len
        seqs.append(one_hot_encode(padded))
    return seqs


def prepare_all_groups():
    genome = FastaFile(GENOME_FASTA)
    bg = load_background_sequence(BACKGROUND_FASTA)

    # Group 1: CTCF
    ctcf_df = pd.read_csv(CTCF_TSV, sep="\t")
    print(f"CTCF df: {ctcf_df.shape}, cols: {ctcf_df.columns.tolist()}")
    def find_col(df, cands):
        for c in df.columns:
            if c.lower() in cands:
                return c
        raise ValueError(f"No col matching {cands}")

    cc = find_col(ctcf_df, {"chrom","chr","chromosome","sequence_name"})
    cs = find_col(ctcf_df, {"start","chromstart","genostart"})
    ce = find_col(ctcf_df, {"end","chromend","genoend"})
    g1 = extract_sequences(ctcf_df, genome, bg, cc, cs, ce,
                           flank=CTCF_FLANK, target_len=TARGET_LEN)
    print(f"Group 1 (CTCF ±{CTCF_FLANK}bp → {TARGET_LEN}bp): {len(g1)}")

    # Group 2: B2 + CTCF
    df2 = pd.read_csv(SINEB2_WITH_CTCF, sep="\t")
    g2 = extract_sequences(df2, genome, bg, "genoName","genoStart","genoEnd",
                           flank=0, target_len=TARGET_LEN)
    print(f"Group 2 (B2+CTCF → {TARGET_LEN}bp): {len(g2)}")

    # Group 3: B2 no CTCF
    df3 = pd.read_csv(SINEB2_NO_CTCF, sep="\t")
    g3 = extract_sequences(df3, genome, bg, "genoName","genoStart","genoEnd",
                           flank=0, target_len=TARGET_LEN)
    print(f"Group 3 (B2 noCTCF → {TARGET_LEN}bp): {len(g3)}")

    genome.close()
    return g1, g2, g3


# ═══════════════════════════════════════════════════════════════════════════════
# 2. PARTIAL FORWARD PASS — extract early trunk layers
# ═══════════════════════════════════════════════════════════════════════════════

def get_trunk_layers(model):
    """
    Extract early trunk layers, splitting conv_tower into individual
    blocks of 4 ops each: [ReLU, Conv1d, BN, MaxPool].
    
    conv_block_1: 220 → 110  (pool2)
    conv_tower block 0: 110 → 55
    conv_tower block 1:  55 → 27
    conv_tower block 2:  27 → 13
    conv_tower block 3:  13 →  6
    conv_tower block 4:   6 →  3
    conv_tower block 5:   3 →  1  ← last usable
    """
    layers = OrderedDict()
    
    # First conv block
    layers["conv_block_1"] = model.conv_block_1
    
    # Split conv_tower.conv_tower Sequential into blocks of 4
    tower_seq = model.conv_tower.conv_tower  # the inner Sequential
    ops = list(tower_seq.children())
    
    for i in range(0, len(ops), 4):
        block = nn.Sequential(*ops[i:i+4])
        layers[f"conv_tower_block{i//4}"] = block
    
    # Residual 1D blocks (no pooling — spatial dim preserved)
    for i in range(1, 12):
        name = f"residual1d_block{i}"
        if hasattr(model, name):
            layers[name] = getattr(model, name)
    
    # Channel reduction
    layers["conv_reduce"] = model.conv_reduce
    
    return layers


def partial_forward(sequences, trunk_layers, batch_size=BATCH_SIZE, min_spatial=1):
    """
    Run sequences through trunk layers sequentially, stopping when
    spatial dimension would become < min_spatial.

    Returns:
        OrderedDict {layer_name: np.ndarray (n_seqs, n_filters)}
        where values are max-pooled activations per filter.
    """
    all_acts = OrderedDict()

    for batch_start in range(0, len(sequences), batch_size):
        batch = np.stack(sequences[batch_start:batch_start + batch_size])
        x = torch.tensor(batch, dtype=torch.float32).to(DEVICE)  # (B, 4, 220)

        with torch.no_grad():
            for name, layer in trunk_layers.items():
                try:
                    x_out = layer(x)
                except RuntimeError as e:
                    # Likely spatial dim too small for this layer's kernel/dilation
                    print(f"  Stopping at '{name}': {e}")
                    break

                spatial = x_out.shape[-1] if x_out.dim() == 3 else x_out.shape[-1] * x_out.shape[-2]
                if spatial < min_spatial:
                    break

                # Max-pool over spatial dim → (B, filters)
                if x_out.dim() == 3:
                    pooled = x_out.max(dim=2).values.cpu().numpy()
                else:
                    pooled = x_out.flatten(2).max(dim=2).values.cpu().numpy()

                all_acts.setdefault(name, []).append(pooled)
                x = x_out  # feed to next layer

    # Concatenate batches
    for name in all_acts:
        all_acts[name] = np.concatenate(all_acts[name], axis=0)

    return all_acts


# ═══════════════════════════════════════════════════════════════════════════════
# 3. ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════

def analyze_top_filters(a1, a2, a3, layer_name, top_k=TOP_K):
    m1, m2, m3 = a1.mean(0), a2.mean(0), a3.mean(0)
    df = pd.DataFrame({
        "filter_idx": range(len(m1)),
        "CTCF": m1, "SINEB2_CTCF": m2, "SINEB2_noCTCF": m3,
    })
    top1 = set(df.nlargest(top_k, "CTCF")["filter_idx"])
    top2 = set(df.nlargest(top_k, "SINEB2_CTCF")["filter_idx"])
    top3 = set(df.nlargest(top_k, "SINEB2_noCTCF")["filter_idx"])
    olap = {
        "shared_all": top1 & top2 & top3,
        "ctcf_specific": top1 - top2 - top3,
        "sineb2_noctcf_specific": top3 - top1 - top2,
        "shared_ctcf_groups": (top1 & top2) - top3,
    }
    print(f"  {layer_name:35s} | all3={len(olap['shared_all']):2d}  "
          f"ctcf_only={len(olap['ctcf_specific']):2d}  "
          f"b2no_only={len(olap['sineb2_noctcf_specific']):2d}  "
          f"ctcf1&2={len(olap['shared_ctcf_groups']):2d}")
    return df, olap


def compute_layerwise_similarity(a1, a2, a3):
    recs = []
    for layer in a1:
        m1, m2, m3 = a1[layer].mean(0), a2[layer].mean(0), a3[layer].mean(0)
        recs.append({
            "layer": layer,
            "cos_CTCF_vs_B2CTCF": 1 - cosine(m1, m2),
            "cos_CTCF_vs_B2noCTCF": 1 - cosine(m1, m3),
            "cos_B2CTCF_vs_B2noCTCF": 1 - cosine(m2, m3),
            "spear_CTCF_vs_B2CTCF": spearmanr(m1, m2).correlation,
            "spear_CTCF_vs_B2noCTCF": spearmanr(m1, m3).correlation,
            "spear_B2CTCF_vs_B2noCTCF": spearmanr(m2, m3).correlation,
            "n_filters": len(m1),
        })
    return pd.DataFrame(recs)


def linear_cka(X, Y):
    X, Y = X - X.mean(0), Y - Y.mean(0)
    n = X.shape[0]
    H = np.eye(n) - 1.0 / n
    Kx, Ky = H @ X @ X.T @ H, H @ Y @ Y.T @ H
    return np.trace(Kx @ Ky) / (np.sqrt(np.trace(Kx @ Kx) * np.trace(Ky @ Ky)) + 1e-12)


def compute_layerwise_cka(a1, a2, a3):
    layers = list(a1.keys())
    n = min(a1[layers[0]].shape[0], a2[layers[0]].shape[0], a3[layers[0]].shape[0])
    recs = []
    for layer in layers:
        x1, x2, x3 = a1[layer][:n], a2[layer][:n], a3[layer][:n]
        recs.append({
            "layer": layer,
            "CKA_CTCF_vs_B2CTCF": linear_cka(x1, x2),
            "CKA_CTCF_vs_B2noCTCF": linear_cka(x1, x3),
            "CKA_B2CTCF_vs_B2noCTCF": linear_cka(x2, x3),
        })
    return pd.DataFrame(recs)


# ═══════════════════════════════════════════════════════════════════════════════
# 4. PLOTTING
# ═══════════════════════════════════════════════════════════════════════════════

PAIR_COLORS = {"CTCF_vs_B2CTCF": "#E63946", "CTCF_vs_B2noCTCF": "#457B9D",
               "B2CTCF_vs_B2noCTCF": "#2A9D8F"}
PAIR_LABELS = {"CTCF_vs_B2CTCF": "CTCF vs B2+CTCF",
               "CTCF_vs_B2noCTCF": "CTCF vs B2 noCTCF",
               "B2CTCF_vs_B2noCTCF": "B2+CTCF vs B2 noCTCF"}


def plot_filter_heatmap(df, layer, top_k=TOP_K, path=None):
    all_top = set()
    for c in ["CTCF", "SINEB2_CTCF", "SINEB2_noCTCF"]:
        all_top.update(df.nlargest(top_k, c)["filter_idx"])
    sub = df[df["filter_idx"].isin(all_top)].set_index("filter_idx")[
        ["CTCF", "SINEB2_CTCF", "SINEB2_noCTCF"]]
    sub = sub.div(sub.max(1), axis=0).sort_values("CTCF", ascending=False)
    fig, ax = plt.subplots(figsize=(5, max(6, len(sub) * 0.22)))
    sns.heatmap(sub, cmap="YlOrRd", ax=ax,
                xticklabels=["CTCF", "B2+CTCF", "B2 noCTCF"],
                cbar_kws={"label": "Rel. activation", "shrink": 0.6})
    ax.set_ylabel("Filter")
    ax.set_title(f"Top filters — {layer}", fontsize=10)
    plt.tight_layout()
    if path: fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_overlap_bars(olap, layer, top_k=TOP_K, path=None):
    cats = [("Shared all 3", len(olap["shared_all"]), "#A8DADC"),
            ("Shared CTCF (1&2)", len(olap["shared_ctcf_groups"]), "#F4A261"),
            ("CTCF-only", len(olap["ctcf_specific"]), "#E63946"),
            ("B2 noCTCF-only", len(olap["sineb2_noctcf_specific"]), "#457B9D")]
    labs, vals, cols = zip(*cats)
    fig, ax = plt.subplots(figsize=(7, 3))
    bars = ax.barh(labs, vals, color=cols, edgecolor="k", lw=0.5)
    for b, v in zip(bars, vals):
        ax.text(b.get_width() + 0.3, b.get_y() + b.get_height()/2, str(v), va="center")
    ax.set_xlabel(f"Filters (top {top_k})")
    ax.set_title(f"Filter overlap — {layer}")
    plt.tight_layout()
    if path: fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_metric(df, prefix, ylabel, title, path=None):
    cols = [c for c in df.columns if c.startswith(prefix + "_")]
    fig, ax = plt.subplots(figsize=(max(8, len(df)*1.2), 5))
    x = np.arange(len(df))
    for c in cols:
        pk = c[len(prefix)+1:]
        ax.plot(x, df[c], "o-", color=PAIR_COLORS.get(pk), label=PAIR_LABELS.get(pk, pk),
                ms=6, lw=2)
    ax.set_xticks(x)
    ax.set_xticklabels(df["layer"], rotation=45, ha="right", fontsize=8)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(fontsize=9)
    ax.grid(alpha=.3)
    plt.tight_layout()
    if path: fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_overlap_evo(df, path=None):
    fig, ax = plt.subplots(figsize=(max(8, len(df)*1.2), 5))
    x = np.arange(len(df))
    for c, clr, lab in [("shared_all_3","#A8DADC","Shared all 3"),
                         ("shared_ctcf_1_2","#F4A261","Shared CTCF (1&2)"),
                         ("ctcf_specific","#E63946","CTCF-only"),
                         ("b2noctcf_specific","#457B9D","B2 noCTCF-only")]:
        ax.plot(x, df[c], "o-", color=clr, label=lab, ms=6, lw=2)
    ax.set_xticks(x)
    ax.set_xticklabels(df["layer"], rotation=45, ha="right", fontsize=8)
    ax.set_ylabel(f"Filters (top {TOP_K})")
    ax.set_title("Filter overlap across layers")
    ax.legend(fontsize=9)
    ax.grid(alpha=.3)
    plt.tight_layout()
    if path: fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
# 5. MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ── 1. Sequences ─────────────────────────────────────────────────────
    print("=" * 65)
    print("STEP 1 — Extracting & padding sequences")
    print("=" * 65)
    g1, g2, g3 = prepare_all_groups()

    # ── 2. Model ─────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("STEP 2 — Loading model & extracting trunk layers")
    print("=" * 65)
    model = load_model()
    trunk_layers = get_trunk_layers(model)
    print(f"\nTrunk layers to probe ({len(trunk_layers)}):")
    for name in trunk_layers:
        print(f"  {name}")

    # ── 3. Partial forward pass ──────────────────────────────────────────
    print("\n" + "=" * 65)
    print("STEP 3 — Partial forward pass (early trunk only)")
    print("=" * 65)

    print(type(model.conv_tower))
    print(list(model.conv_tower.named_children())[:3])
    
    print(model.conv_tower)
    
    print(f"  Group 1 ({len(g1)} seqs)...")
    a1 = partial_forward(g1, trunk_layers)
    print(f"  Group 2 ({len(g2)} seqs)...")
    a2 = partial_forward(g2, trunk_layers)
    print(f"  Group 3 ({len(g3)} seqs)...")
    a3 = partial_forward(g3, trunk_layers)

    # Keep only layers present in all 3 groups
    common_layers = [l for l in a1 if l in a2 and l in a3]
    print(f"\n  Activations captured for {len(common_layers)} layers:")
    for name in common_layers:
        print(f"    {name:35s} → shape {a1[name].shape}")

    # ── 4A. First-layer filter analysis ──────────────────────────────────
    print("\n" + "=" * 65)
    print("STEP 4A — First-layer top filters")
    print("=" * 65)
    first = common_layers[0]
    fdf, folap = analyze_top_filters(a1[first], a2[first], a3[first], first)
    fdf.to_csv(f"{OUTPUT_DIR}/first_layer_filter_means.csv", index=False)
    plot_filter_heatmap(fdf, first, path=f"{OUTPUT_DIR}/first_layer_heatmap.png")
    plot_overlap_bars(folap, first, path=f"{OUTPUT_DIR}/first_layer_overlap.png")

    # ── 4B. Layer-wise similarity ────────────────────────────────────────
    print("\n" + "=" * 65)
    print("STEP 4B — Layer-wise similarity (cosine & Spearman)")
    print("=" * 65)
    # Filter to common layers only
    a1c = {k: a1[k] for k in common_layers}
    a2c = {k: a2[k] for k in common_layers}
    a3c = {k: a3[k] for k in common_layers}

    sim = compute_layerwise_similarity(a1c, a2c, a3c)
    sim.to_csv(f"{OUTPUT_DIR}/layerwise_similarity.csv", index=False)
    print(sim[["layer", "cos_CTCF_vs_B2CTCF", "cos_CTCF_vs_B2noCTCF",
               "cos_B2CTCF_vs_B2noCTCF"]].to_string(index=False))
    plot_metric(sim, "cos", "Cosine similarity",
                "Cosine similarity across trunk layers",
                f"{OUTPUT_DIR}/layerwise_cosine.png")
    plot_metric(sim, "spear", "Spearman ρ",
                "Spearman correlation across trunk layers",
                f"{OUTPUT_DIR}/layerwise_spearman.png")

    # ── 4C. CKA ─────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("STEP 4C — CKA across layers")
    print("=" * 65)
    cka = compute_layerwise_cka(a1c, a2c, a3c)
    cka.to_csv(f"{OUTPUT_DIR}/layerwise_cka.csv", index=False)
    print(cka.to_string(index=False))
    plot_metric(cka, "CKA", "CKA",
                "Centered Kernel Alignment across trunk layers",
                f"{OUTPUT_DIR}/layerwise_cka.png")

    # ── 4D. Per-layer overlap evolution ──────────────────────────────────
    print("\n" + "=" * 65)
    print("STEP 4D — Per-layer filter overlap")
    print("=" * 65)
    recs = []
    for layer in common_layers:
        _, ol = analyze_top_filters(a1[layer], a2[layer], a3[layer], layer)
        recs.append({
            "layer": layer,
            "shared_all_3": len(ol["shared_all"]),
            "shared_ctcf_1_2": len(ol["shared_ctcf_groups"]),
            "ctcf_specific": len(ol["ctcf_specific"]),
            "b2noctcf_specific": len(ol["sineb2_noctcf_specific"]),
        })
    odf = pd.DataFrame(recs)
    odf.to_csv(f"{OUTPUT_DIR}/perlayer_overlap.csv", index=False)
    plot_overlap_evo(odf, f"{OUTPUT_DIR}/perlayer_overlap.png")

    print(f"\n{'='*65}")
    print(f"Done! All outputs in: {OUTPUT_DIR}/")
    print(f"{'='*65}")


if __name__ == "__main__":
    main()