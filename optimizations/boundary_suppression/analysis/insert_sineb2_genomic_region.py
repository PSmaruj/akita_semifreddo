"""
Insert consensus SINE B2 (B2_Mm2) elements into the central bin of input sequences,
avoiding CTCF motif locations ± 15 bp flanks, then run Akita predictions and
save upper-right quarter (URQ) mean values for original and modified sequences.

Usage:
    python insert_sineb2_genomic_region.py

Outputs:
    sine_b2_insertion_results.tsv  — one row per (sequence × insertion_count)
"""

import sys
import os
import ast
import numpy as np
import pandas as pd
import torch

# ── Paths ─────────────────────────────────────────────────────────────────────
sys.path.insert(0, "/home1/smaruj/pytorch_akita")
sys.path.insert(0, os.path.abspath("/home1/smaruj/akita_semifreddo/"))

from utils.data_utils import from_upper_triu_batch
from utils.model_utils import load_model
from utils.scores_utils import compute_insulation_scores

TSV_PATH   = (
    "/project2/fudenber_735/smaruj/sequence_design/ledidi_semifreddo_akita/optimizations/"
    "boundary_suppression/results/successful_optimizations.tsv"
)
OHE_ROOT   = (
    "/project2/fudenber_735/smaruj/sequence_design/ledidi_semifreddo_akita/optimizations/"
    "boundary_suppression/initial_sequences"
)
MODEL_PATH = (
    "/home1/smaruj/pytorch_akita/models/finetuned/mouse/"
    "Hsieh2019_mESC/checkpoints/"
    "Akita_v2_mouse_Hsieh2019_mESC_model0_finetuned.pth"
)
OUT_TSV = (
    "/project2/fudenber_735/smaruj/sequence_design/ledidi_semifreddo_akita/optimizations/"
    "boundary_suppression/sine_b2_insertion_results.tsv"
)

# ── Constants ─────────────────────────────────────────────────────────────────
BIN_SIZE       = 2048
CENTER_BIN     = 320
CENTER_START   = CENTER_BIN * BIN_SIZE
CENTER_END     = (CENTER_BIN + 1) * BIN_SIZE
CTCF_FLANK     = 15

URQ_ROW_SLICE  = slice(0, 250)
URQ_COL_SLICE  = slice(260, 512)

SINE_B2 = (
    "GGGGCTGGAGAGATGGCTCAGCGGTTAAGAGCACTGACTGCTCTTCCAGAGGTCCTGAGTTCAATTCCC"
    "AGCAACCACATGGTGGCTCACAACCATCTGTAATGGGATCTGATGCCCTCTTCTGGTGTGTCTGAAGAC"
    "AGCTACAGTGTACTCACATACATAAATAAATAAATAAATAAATCTTTAAAAAAAAAAAAAA"
)
SINE_LEN       = len(SINE_B2)
INSERT_COUNTS  = [1, 2, 3]

# ── One-hot encoding ──────────────────────────────────────────────────────────
NT_MAP = {'A': 0, 'C': 1, 'G': 2, 'T': 3,
          'a': 0, 'c': 1, 'g': 2, 't': 3}

def one_hot_encode(seq: str) -> np.ndarray:
    """Return (4, L) one-hot array; unknown bases → all zeros."""
    ohe = np.zeros((4, len(seq)), dtype=np.float32)
    for i, nt in enumerate(seq):
        idx = NT_MAP.get(nt)
        if idx is not None:
            ohe[idx, i] = 1.0
    return ohe

# ── CTCF forbidden zones ──────────────────────────────────────────────────────
def parse_ctcf_coords(raw: str):
    """
    Parse 'orig_CTCFs_coord' field, e.g.:
        {(1349, 1368, '+'), (709, 728, '+')}
    Returns list of (start, end, strand) tuples.
    """
    if not isinstance(raw, str) or raw.strip() in ('', 'nan', '{}', 'set()'):
        return []
    try:
        return list(ast.literal_eval(raw))
    except Exception:
        return []

def forbidden_positions(ctcf_list, region_start: int, region_end: int):
    """Return positions forbidden due to CTCF ± flank within [region_start, region_end)."""
    forbidden = set()
    for (s, e, _strand) in ctcf_list:
        for pos in range(s - CTCF_FLANK, e + CTCF_FLANK + 1):
            if region_start <= pos < region_end:
                forbidden.add(pos)
    return forbidden

# ── Find valid insertion sites ────────────────────────────────────────────────
def find_insertion_sites(ctcf_list, n_insertions: int, seed: int = None):
    """
    Randomly sample n_insertions non-overlapping SINE insertion sites within
    the central bin, avoiding CTCF ± CTCF_FLANK bp zones.

    Returns sorted list of absolute start positions, or None if not enough room.
    """
    rng = np.random.default_rng(seed)
    forbidden = forbidden_positions(ctcf_list, CENTER_START, CENTER_END)

    candidates = [
        pos for pos in range(CENTER_START, CENTER_END - SINE_LEN + 1)
        if not any(p in forbidden for p in range(pos, pos + SINE_LEN))
    ]

    if len(candidates) < n_insertions:
        return None

    selected  = []
    available = list(candidates)
    for _ in range(n_insertions):
        if not available:
            return None
        chosen = int(rng.choice(available))
        selected.append(chosen)
        available = [p for p in available
                     if p + SINE_LEN <= chosen or p >= chosen + SINE_LEN]

    return sorted(selected)

# ── Insert SINE into OHE array ────────────────────────────────────────────────
def insert_sine_into_ohe(ohe: np.ndarray, abs_position: int) -> np.ndarray:
    """Overwrite SINE_B2 into a (4, L) array at abs_position."""
    sine_ohe = one_hot_encode(SINE_B2)
    modified = ohe.copy()
    modified[:, abs_position: abs_position + SINE_LEN] = sine_ohe
    return modified

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    df = pd.read_csv(TSV_PATH, sep='\t')
    print(f"Loaded {len(df)} rows from {TSV_PATH}")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = load_model(MODEL_PATH, device)

    records = []

    for idx, row in df.iterrows():
        fold   = int(row['fold'])
        chrom  = row['chrom']
        cstart = int(row['centered_start'])
        cend   = int(row['centered_end'])

        ohe_path = f"{OHE_ROOT}/fold{fold}/{chrom}_{cstart}_{cend}_X.pt"
        if not os.path.exists(ohe_path):
            print(f"  [WARN] Missing OHE file: {ohe_path}, skipping.")
            continue

        raw      = torch.load(ohe_path, map_location='cpu')
        orig_ohe = raw.squeeze(0).float().detach().numpy()   # (4, L)

        ctcf_list = parse_ctcf_coords(str(row.get('orig_CTCFs_coord', '')))

        # Predict original map
        x        = torch.from_numpy(orig_ohe).unsqueeze(0).float().to(device)
        with torch.no_grad():
            orig_maps = from_upper_triu_batch(model(x).cpu())
        urq_orig  = compute_insulation_scores(orig_maps, URQ_ROW_SLICE, URQ_COL_SLICE)[0]

        base_record = {
            'chrom': chrom, 'fold': fold,
            'centered_start': cstart, 'centered_end': cend,
        }

        for n in INSERT_COUNTS:
            sites = find_insertion_sites(ctcf_list, n)
            if sites is None:
                print(f"  [WARN] Not enough valid insertion sites for "
                      f"{chrom}:{cstart}-{cend} with {n} SINE(s); skipping.")
                records.append({**base_record, 'n_insertions': n,
                                 'URQ_orig': urq_orig, 'URQ_sine': np.nan,
                                 'URQ_diff': np.nan, 'insertion_sites': None})
                continue

            mod_ohe = orig_ohe.copy()
            for site in sites:
                mod_ohe = insert_sine_into_ohe(mod_ohe, site)

            x_mod = torch.from_numpy(mod_ohe).unsqueeze(0).float().to(device)
            with torch.no_grad():
                mod_maps = from_upper_triu_batch(model(x_mod).cpu())
            urq_sine = compute_insulation_scores(mod_maps, URQ_ROW_SLICE, URQ_COL_SLICE)[0]

            print(f"  {chrom}:{cstart}-{cend}  n={n}  "
                  f"URQ_orig={urq_orig:.4f}  URQ_sine={urq_sine:.4f}  "
                  f"diff={urq_sine - urq_orig:.4f}")

            records.append({**base_record, 'n_insertions': n,
                             'URQ_orig': urq_orig, 'URQ_sine': urq_sine,
                             'URQ_diff': urq_sine - urq_orig,
                             'insertion_sites': str(sites)})

    out_df = pd.DataFrame(records)
    out_df.to_csv(OUT_TSV, sep='\t', index=False)
    print(f"\nSaved → {OUT_TSV}  ({len(out_df)} rows)")


if __name__ == '__main__':
    main()