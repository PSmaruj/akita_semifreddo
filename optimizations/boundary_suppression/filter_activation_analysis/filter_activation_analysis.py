"""
filter_activation_analysis.py

Compute filter activations for three sequence groups and save results for
downstream plotting.

Outputs (all in OUTPUT_DIR):
    # First layer (conv_block_1) — for heatmap and consensus sequences
    activations_g1.npy        -- (n_seqs, n_filters) max-pooled, CTCF
    activations_g2.npy        -- (n_seqs, n_filters) max-pooled, B2+CTCF
    activations_g3.npy        -- (n_seqs, n_filters) max-pooled, B2 noCTCF
    activations_g2_spatial.npy -- (n_seqs, n_filters, spatial) B2+CTCF
    sequences_g2.npy          -- (n_seqs, 4, 220) B2+CTCF one-hot sequences
    conv_weights.npy          -- (n_filters, 4, kernel_size) conv_block_1 kernels

    # Layer 4 (conv_tower_block3) — for boxplots and volcano plot
    l4_acts_g1.npy            -- (n_seqs, n_filters, spatial) CTCF
    l4_acts_g2.npy            -- (n_seqs, n_filters, spatial) B2+CTCF
    l4_acts_g3.npy            -- (n_seqs, n_filters, spatial) B2 noCTCF
    l4_acts_before.npy        -- (n_seqs, n_filters, spatial) before optimization
    l4_acts_after.npy         -- (n_seqs, n_filters, spatial) after optimization
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
from collections import OrderedDict
from Bio import SeqIO
from pysam import FastaFile
import warnings
warnings.filterwarnings("ignore")

sys.path.append(os.path.abspath("/home1/smaruj/pytorch_akita/"))
from akita.model import SeqNN

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
GENOME_FASTA    = "/project2/fudenber_735/genomes/mm10/mm10.fa"
BACKGROUND_FASTA = (
    "/project2/fudenber_735/smaruj/sequence_design/"
    "ledidi_semifreddo_akita/analysis/background_generation/"
    "background_sequences_scd30_totvar1300.fasta"
)
CTCF_TSV        = "/home1/smaruj/akita_semifreddo/data/ctcf_tables/CTCFs_jaspar_filtered_mm10_top20percent.tsv"
SINEB2_WITH_CTCF = "/home1/smaruj/akita_semifreddo/data/sine_b2_tables/sineB2_with_ctcf_300.tsv"
SINEB2_NO_CTCF   = "/home1/smaruj/akita_semifreddo/data/sine_b2_tables/sineB2_no_ctcf_300.tsv"
MODEL_PATH      = (
    "/home1/smaruj/pytorch_akita/models/finetuned/mouse/Hsieh2019_mESC"
    "/checkpoints/Akita_v2_mouse_Hsieh2019_mESC_model0_finetuned.pth"
)
OUTPUT_DIR = (
    "/project2/fudenber_735/smaruj/sequence_design/"
    "ledidi_semifreddo_akita/optimizations/boundary_suppression/"
    "filter_activation_analysis"
)

SUPPRESSION_COORD_TSV  = "/project2/fudenber_735/smaruj/sequence_design/ledidi_semifreddo_akita/optimizations/boundary_suppression/results/successful_optimizations.tsv"
INIT_SEQ_PATH          = "/project2/fudenber_735/smaruj/sequence_design/ledidi_semifreddo_akita/optimizations/boundary_suppression/initial_sequences"
SLICE_PATH             = "/project2/fudenber_735/smaruj/sequence_design/ledidi_semifreddo_akita/optimizations/boundary_suppression/results"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
TARGET_LEN       = 220
CTCF_FLANK       = 30
N_SAMPLES        = 300
BATCH_SIZE       = 64
BATCH_SIZE_2048  = 16
DEVICE           = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
LAYER4_IDX       = 4   # conv_tower_block3

# Attribute path to the Conv1d inside conv_block_1 — adjust if needed
CONV1_ATTR  = "conv"   # i.e. model.conv_block_1.conv


# ---------------------------------------------------------------------------
# Sequence utilities
# ---------------------------------------------------------------------------

def one_hot_encode(seq: str) -> np.ndarray:
    """Return (4, L) one-hot array; ambiguous bases → 0.25 each."""
    mapping = {"A": 0, "C": 1, "G": 2, "T": 3}
    enc = np.zeros((4, len(seq)), dtype=np.float32)
    for i, b in enumerate(seq.upper()):
        if b in mapping:
            enc[mapping[b], i] = 1.0
        else:
            enc[:, i] = 0.25
    return enc


def load_background_sequence(path: str) -> str:
    return str(next(SeqIO.parse(path, "fasta")).seq)


def pad_to_target(seq: str, target_len: int, bg_seq: str,
                  bg_offset: int = 0) -> tuple[str, int]:
    """Centre *seq* and pad symmetrically with background; trim if too long."""
    if len(seq) >= target_len:
        excess = len(seq) - target_len
        s = excess // 2
        return seq[s:s + target_len], bg_offset
    pad_total = target_len - len(seq)
    pad_left  = pad_total // 2
    pad_right = pad_total - pad_left
    bg_len = len(bg_seq)
    left  = "".join(bg_seq[(bg_offset + i) % bg_len] for i in range(pad_left))
    bg_offset = (bg_offset + pad_left) % bg_len
    right = "".join(bg_seq[(bg_offset + i) % bg_len] for i in range(pad_right))
    bg_offset = (bg_offset + pad_right) % bg_len
    return left + seq + right, bg_offset


def find_col(df: pd.DataFrame, candidates: set) -> str:
    for c in df.columns:
        if c.lower() in candidates:
            return c
    raise ValueError(f"No column matching {candidates} in {df.columns.tolist()}")


def extract_sequences(df: pd.DataFrame, genome, bg_seq: str,
                      chrom_col: str, start_col: str, end_col: str,
                      flank: int = 0) -> list:
    seqs, bg_off = [], 0
    for _, row in df.head(N_SAMPLES).iterrows():
        chrom = str(row[chrom_col])
        start = max(0, int(row[start_col]) - flank)
        end   = int(row[end_col]) + flank
        try:
            seq = genome.fetch(chrom, start, end)
        except (KeyError, ValueError):
            continue
        padded, bg_off = pad_to_target(seq, TARGET_LEN, bg_seq, bg_off)
        assert len(padded) == TARGET_LEN
        seqs.append(one_hot_encode(padded))
    return seqs


def prepare_all_groups(genome, bg: str) -> tuple[list, list, list]:
    # Group 1: strong CTCFs (±30 bp flanks)
    ctcf_df = pd.read_csv(CTCF_TSV, sep="\t")
    cc = find_col(ctcf_df, {"chrom", "chr", "chromosome", "sequence_name"})
    cs = find_col(ctcf_df, {"start", "chromstart", "genostart"})
    ce = find_col(ctcf_df, {"end",   "chromend",   "genoend"})
    g1 = extract_sequences(ctcf_df, genome, bg, cc, cs, ce, flank=CTCF_FLANK)
    print(f"Group 1 (CTCF ±{CTCF_FLANK}bp → {TARGET_LEN}bp): {len(g1)} seqs")

    # Group 2: B2_Mm2 with CTCF
    df2 = pd.read_csv(SINEB2_WITH_CTCF, sep="\t")
    g2  = extract_sequences(df2, genome, bg, "genoName", "genoStart", "genoEnd")
    print(f"Group 2 (B2+CTCF → {TARGET_LEN}bp):    {len(g2)} seqs")

    # Group 3: B2_Mm2 without CTCF
    df3 = pd.read_csv(SINEB2_NO_CTCF, sep="\t")
    g3  = extract_sequences(df3, genome, bg, "genoName", "genoStart", "genoEnd")
    print(f"Group 3 (B2 noCTCF → {TARGET_LEN}bp):  {len(g3)} seqs")

    return g1, g2, g3


# ---------------------------------------------------------------------------
# Model utilities
# ---------------------------------------------------------------------------

def load_model() -> torch.nn.Module:
    model = SeqNN()
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    print(f"Model loaded on {DEVICE}")
    return model


def get_conv1_layer(model) -> torch.nn.Module:
    """Return the Conv1d that is the first layer of conv_block_1."""
    block = model.conv_block_1
    conv  = getattr(block, CONV1_ATTR, None)
    if conv is None:
        # Fallback: print structure to help user locate the attribute
        print("conv_block_1 submodules:")
        for name, mod in block.named_modules():
            print(f"  {name}: {mod.__class__.__name__}")
        raise AttributeError(
            f"Could not find attribute '{CONV1_ATTR}' on conv_block_1. "
            "Set CONV1_ATTR to the correct name from the printout above."
        )
    return conv


# ---------------------------------------------------------------------------
# Activation computation
# ---------------------------------------------------------------------------

def compute_activations(sequences: list, block: torch.nn.Module,
                        keep_spatial: bool = False) -> np.ndarray:
    """
    Run *sequences* through conv_block_1.

    Parameters
    ----------
    keep_spatial : if True return (n_seqs, n_filters, spatial_len),
                   else return max-pooled (n_seqs, n_filters).
    """
    all_acts = []
    for i in range(0, len(sequences), BATCH_SIZE):
        batch = np.stack(sequences[i:i + BATCH_SIZE])
        t = torch.tensor(batch, dtype=torch.float32).to(DEVICE)
        with torch.no_grad():
            out = block(t)          # (B, n_filters, spatial)
        arr = out.cpu().numpy()
        all_acts.append(arr if keep_spatial else arr.max(axis=2))
    return np.concatenate(all_acts, axis=0)


# ---------------------------------------------------------------------------
# Layer-4 forward pass
# ---------------------------------------------------------------------------

def get_trunk_layers(model) -> OrderedDict:
    """Split conv_tower into individual blocks of 4 ops each."""
    layers = OrderedDict()
    layers["conv_block_1"] = model.conv_block_1
    tower_seq = model.conv_tower.conv_tower
    ops = list(tower_seq.children())
    for i in range(0, len(ops), 4):
        block = torch.nn.Sequential(*ops[i:i + 4])
        layers[f"conv_tower_block{i // 4}"] = block
    return layers


def partial_forward(sequences: list, trunk_layers: OrderedDict,
                    target_layer_idx: int,
                    batch_size: int = BATCH_SIZE) -> np.ndarray:
    """Run sequences through trunk up to target_layer_idx.
    Returns (n_seqs, n_filters, spatial_len)."""
    layer_list = list(trunk_layers.values())
    all_acts = []
    for i in range(0, len(sequences), batch_size):
        batch = np.stack(sequences[i:i + batch_size])
        x = torch.tensor(batch, dtype=torch.float32).to(DEVICE)
        with torch.no_grad():
            for li in range(target_layer_idx + 1):
                x = layer_list[li](x)
        all_acts.append(x.cpu().numpy())
    return np.concatenate(all_acts, axis=0)


def partial_forward_tensors(tensors: list, trunk_layers: OrderedDict,
                             target_layer_idx: int,
                             batch_size: int = BATCH_SIZE_2048) -> np.ndarray:
    """Same as partial_forward but accepts a list of torch tensors."""
    layer_list = list(trunk_layers.values())
    all_acts = []
    for i in range(0, len(tensors), batch_size):
        batch = tensors[i:i + batch_size]
        x = torch.stack([t.squeeze(0) if t.dim() == 3 else t
                         for t in batch]).to(DEVICE)
        with torch.no_grad():
            for li in range(target_layer_idx + 1):
                x = layer_list[li](x)
        all_acts.append(x.cpu().numpy())
    return np.concatenate(all_acts, axis=0)


# ---------------------------------------------------------------------------
# Before / after slice loading (boundary suppression)
# ---------------------------------------------------------------------------

def _build_path_prefix(row) -> str:
    """Build the chrom_start_end prefix, handling fold subdirectory if present."""
    prefix = f"{row['chrom']}_{row['centered_start']}_{row['centered_end']}"
    if "fold" in row.index:
        return f"fold{row['fold']}/{prefix}"
    return prefix


def load_slices(coord_df: pd.DataFrame, path: str) -> list:
    """Load optimized 2048bp slice tensors."""
    slices = []
    missing = []
    for _, row in coord_df.iterrows():
        fpath = f"{path}/{_build_path_prefix(row)}_gen_seq.pt"
        try:
            t = torch.load(fpath, weights_only=True, map_location="cpu")
            slices.append(t.squeeze(0) if t.dim() == 3 else t)
        except FileNotFoundError:
            missing.append(fpath)
    print(f"  Loaded {len(slices)} slices (of {len(coord_df)} requested)")
    if missing:
        print(f"  First missing path: {missing[0]}")
    return slices


def load_init_bins(coord_df: pd.DataFrame, path: str,
                   slice_offset: int = 256, cropping: int = 64,
                   bin_size: int = 2048) -> list:
    """Extract the central 2048bp bin from full init sequences."""
    edit_start = (slice_offset + cropping) * bin_size
    edit_end   = edit_start + bin_size
    bins = []
    missing = []
    for _, row in coord_df.iterrows():
        fpath = f"{path}/{_build_path_prefix(row)}_X.pt"
        try:
            X = torch.load(fpath, weights_only=True, map_location="cpu")
            X = X.squeeze(0) if X.dim() == 3 else X
            bins.append(X[:, edit_start:edit_end])
        except FileNotFoundError:
            missing.append(fpath)
    print(f"  Loaded {len(bins)} init bins (of {len(coord_df)} requested)")
    if missing:
        print(f"  First missing path: {missing[0]}")
    return bins


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Loading genome and background sequence...")
    genome = FastaFile(GENOME_FASTA)
    bg     = load_background_sequence(BACKGROUND_FASTA)

    print("\nPreparing sequence groups...")
    g1, g2, g3 = prepare_all_groups(genome, bg)
    genome.close()

    print("\nLoading model...")
    model        = load_model()
    conv_block   = model.conv_block_1
    conv_layer   = get_conv1_layer(model)
    trunk_layers = get_trunk_layers(model)

    print("\nComputing first-layer activations...")
    # Max-pooled (n_seqs, n_filters) for all groups — used for heatmap
    a1 = compute_activations(g1, conv_block, keep_spatial=False)
    a2 = compute_activations(g2, conv_block, keep_spatial=False)
    a3 = compute_activations(g3, conv_block, keep_spatial=False)
    print(f"  Max-pooled activation shape per group: {a1.shape}")

    # Spatial activations for g2 only — needed to find peak position per filter
    a2_spatial = compute_activations(g2, conv_block, keep_spatial=True)
    print(f"  Spatial activation shape (g2): {a2_spatial.shape}")

    np.save(os.path.join(OUTPUT_DIR, "activations_g1.npy"), a1)
    np.save(os.path.join(OUTPUT_DIR, "activations_g2.npy"), a2)
    np.save(os.path.join(OUTPUT_DIR, "activations_g3.npy"), a3)
    np.save(os.path.join(OUTPUT_DIR, "activations_g2_spatial.npy"), a2_spatial)
    np.save(os.path.join(OUTPUT_DIR, "sequences_g2.npy"), np.stack(g2))

    weights = conv_layer.weight.detach().cpu().numpy()
    np.save(os.path.join(OUTPUT_DIR, "conv_weights.npy"), weights)
    print(f"  Saved conv weights: shape {weights.shape}")

    # ── Layer-4 activations for 220bp groups ─────────────────────────────
    print("\nComputing layer-4 activations (220bp groups)...")
    l4_g1 = partial_forward(g1, trunk_layers, LAYER4_IDX)
    l4_g2 = partial_forward(g2, trunk_layers, LAYER4_IDX)
    l4_g3 = partial_forward(g3, trunk_layers, LAYER4_IDX)
    print(f"  Layer-4 activation shape per group: {l4_g1.shape}")

    np.save(os.path.join(OUTPUT_DIR, "l4_acts_g1.npy"), l4_g1)
    np.save(os.path.join(OUTPUT_DIR, "l4_acts_g2.npy"), l4_g2)
    np.save(os.path.join(OUTPUT_DIR, "l4_acts_g3.npy"), l4_g3)

    # ── Layer-4 activations for before/after optimization ─────────────────
    print("\nLoading boundary suppression sequences...")
    coord_df = pd.read_csv(SUPPRESSION_COORD_TSV, sep="\t")
    print(f"  coord_df columns: {coord_df.columns.tolist()}")
    print(f"  First row: {coord_df.iloc[0].to_dict()}")

    print("  Init bins (before)...")
    before_tensors = load_init_bins(coord_df, INIT_SEQ_PATH)
    print("  Optimized slices (after)...")
    after_tensors  = load_slices(coord_df, SLICE_PATH)

    if not before_tensors or not after_tensors:
        raise RuntimeError(
            "No before/after sequences loaded — check column names and paths above."
        )

    print("\nComputing layer-4 activations (before/after)...")
    l4_before = partial_forward_tensors(before_tensors, trunk_layers, LAYER4_IDX)
    l4_after  = partial_forward_tensors(after_tensors,  trunk_layers, LAYER4_IDX)
    print(f"  Before shape: {l4_before.shape}  |  After shape: {l4_after.shape}")

    np.save(os.path.join(OUTPUT_DIR, "l4_acts_before.npy"), l4_before)
    np.save(os.path.join(OUTPUT_DIR, "l4_acts_after.npy"),  l4_after)

    print(f"\nAll outputs saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()