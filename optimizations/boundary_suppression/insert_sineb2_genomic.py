"""
Insert consensus SINE B2 (B2_Mm2) elements into the central bin of input sequences,
avoiding CTCF motif locations ± 15 bp flanks, then run Akita predictions and
save upper-right quarter (URQ) mean values for original and modified sequences.

Usage:
    python insert_sine_b2_akita.py

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
from akita_model.model import SeqNN

TSV_PATH   = (
    "/project2/fudenber_735/smaruj/sequence_design/ledidi_semifreddo_akita/"
    "suppressing_CTCFs/results_repeated/only_successful_seqs.tsv"
)
OHE_ROOT   = (
    "/project2/fudenber_735/smaruj/sequence_design/ledidi_semifreddo_akita/"
    "suppressing_CTCFs/ohe_X"
)
MODEL_PATH = (
    "/home1/smaruj/pytorch_akita/models/finetuned/mouse/"
    "Hsieh2019_mESC/checkpoints/"
    "Akita_v2_mouse_Hsieh2019_mESC_model0_finetuned.pth"
)
OUT_TSV = "sine_b2_insertion_results.tsv"

# ── Constants ─────────────────────────────────────────────────────────────────
SEQ_LEN        = 640 * 2048 // 640   # each bin = 2048 bp; total = 640 bins × 2048 bp
BIN_SIZE       = 2048                # bp per bin
TOTAL_BINS     = 640
CENTER_BIN     = 320                 # 0-indexed central (320th) bin
CENTER_START   = CENTER_BIN * BIN_SIZE          # bp offset of center bin start
CENTER_END     = (CENTER_BIN + 1) * BIN_SIZE    # bp offset of center bin end
CTCF_FLANK     = 15                  # bp to protect around each CTCF motif
MATRIX_LEN     = 512
NUM_DIAGS      = 2

SINE_B2 = (
    "GGGGCTGGAGAGATGGCTCAGCGGTTAAGAGCACTGACTGCTCTTCCAGAGGTCCTGAGTTCAATTCCC"
    "AGCAACCACATGGTGGCTCACAACCATCTGTAATGGGATCTGATGCCCTCTTCTGGTGTGTCTGAAGAC"
    "AGCTACAGTGTACTCACATACATAAATAAATAAATAAATAAATCTTTAAAAAAAAAAAAAA"
)
SINE_LEN       = len(SINE_B2)   # 189 bp
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
    Returns list of (start, end, strand) tuples (coordinates in the full 640-bin seq).
    """
    if not isinstance(raw, str) or raw.strip() in ('', 'nan', '{}', 'set()'):
        return []
    try:
        parsed = ast.literal_eval(raw)
        return list(parsed)
    except Exception:
        return []

def forbidden_positions(ctcf_list, region_start: int, region_end: int):
    """
    Return a set of absolute positions (within the full sequence) that are
    forbidden because they are within CTCF_FLANK bp of a CTCF motif AND fall
    inside [region_start, region_end).
    """
    forbidden = set()
    for (s, e, _strand) in ctcf_list:
        for pos in range(s - CTCF_FLANK, e + CTCF_FLANK + 1):
            if region_start <= pos < region_end:
                forbidden.add(pos)
    return forbidden

# # ── Find valid insertion sites within center bin ──────────────────────────────
# def find_insertion_sites(ctcf_list, n_insertions: int):
#     """
#     Find up to n_insertions non-overlapping positions inside CENTER_START..CENTER_END
#     where a SINE_LEN insertion fits without overlapping CTCF ± 15 bp zones.

#     Returns list of absolute start positions (len == n_insertions), or None if
#     not enough room.
#     """
#     forbidden = forbidden_positions(ctcf_list, CENTER_START, CENTER_END)
#     sites = []
#     pos = CENTER_START
#     while pos + SINE_LEN <= CENTER_END and len(sites) < n_insertions:
#         candidate = range(pos, pos + SINE_LEN)
#         if not any(p in forbidden for p in candidate):
#             sites.append(pos)
#             pos += SINE_LEN   # next candidate starts after this insertion
#         else:
#             pos += 1
#     if len(sites) < n_insertions:
#         return None
#     return sites

# ── Find valid insertion sites within center bin ──────────────────────────────
def find_insertion_sites(ctcf_list, n_insertions: int, max_attempts: int = 10000, seed: int = None):
    """
    Find n_insertions non-overlapping positions inside CENTER_START..CENTER_END
    where a SINE_LEN insertion fits without overlapping CTCF ± 15 bp zones.
    Positions are chosen randomly (uniform over valid start positions).

    Strategy:
      1. Build the set of all valid start positions (window of SINE_LEN bp fully
         free of forbidden positions).
      2. Randomly sample one, add it, remove all positions that would now overlap
         with the chosen site, and repeat.

    Returns list of absolute start positions (len == n_insertions), or None if
    not enough valid non-overlapping sites exist.
    """
    rng = np.random.default_rng(seed)
    forbidden = forbidden_positions(ctcf_list, CENTER_START, CENTER_END)

    # Build set of candidate start positions whose full SINE window is clean
    candidates = []
    for pos in range(CENTER_START, CENTER_END - SINE_LEN + 1):
        if not any(p in forbidden for p in range(pos, pos + SINE_LEN)):
            candidates.append(pos)

    if len(candidates) < n_insertions:
        return None

    selected = []
    available = list(candidates)

    for _ in range(n_insertions):
        if not available:
            return None
        chosen = int(rng.choice(available))
        selected.append(chosen)
        # Remove all positions that would overlap with the chosen site
        # (any start within ±SINE_LEN-1 of chosen would overlap)
        available = [
            p for p in available
            if p + SINE_LEN <= chosen or p >= chosen + SINE_LEN
        ]

    return sorted(selected)


# ── Insert SINE into OHE tensor ───────────────────────────────────────────────
def insert_sine_into_ohe(ohe: np.ndarray, abs_position: int) -> np.ndarray:
    """
    Overwrite SINE_B2 into `ohe` (shape 4 × L) at `abs_position`.
    Keeps sequence length constant.
    """
    sine_ohe = one_hot_encode(SINE_B2)   # (4, SINE_LEN)
    modified = ohe.copy()
    modified[:, abs_position: abs_position + SINE_LEN] = sine_ohe
    return modified

# ── Akita helpers ─────────────────────────────────────────────────────────────
def set_diag(matrix, value, k):
    rows, cols = matrix.shape
    for i in range(rows):
        if 0 <= i + k < cols:
            matrix[i, i + k] = value

def from_upper_triu_batch(batch_vectors, matrix_len=MATRIX_LEN, num_diags=NUM_DIAGS):
    if isinstance(batch_vectors, torch.Tensor):
        batch_vectors = batch_vectors.detach().cpu().numpy()
    batch_size = len(batch_vectors)
    matrices = np.zeros((batch_size, matrix_len, matrix_len), dtype=np.float32)
    triu_indices = np.triu_indices(matrix_len, num_diags)
    for i in range(batch_size):
        matrices[i][triu_indices] = batch_vectors[i][0, :]
        matrices[i] = matrices[i] + matrices[i].T
        for k in range(-num_diags + 1, num_diags):
            set_diag(matrices[i], np.nan, k)
    return matrices

def predict_map(model, ohe_array: np.ndarray, device: str):
    """Run Akita on a (4, L) one-hot array and return a (512, 512) contact map."""
    x = torch.from_numpy(ohe_array).unsqueeze(0).float()  # (1, 4, L)
    x = x.to(device)
    with torch.no_grad():
        pred = model(x)
    maps = from_upper_triu_batch(pred)
    return maps[0]

def upper_right_quarter_mean(contact_map: np.ndarray) -> float:
    return float(np.nanmean(contact_map[0:250, 260:512]))

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    # Load data
    df = pd.read_csv(TSV_PATH, sep='\t')
    print(f"Loaded {len(df)} rows from {TSV_PATH}")

    # Load model
    print("\n── Loading model ──")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    model = SeqNN()
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()
    print("Model loaded successfully\n")

    records = []

    for idx, row in df.iterrows():
        fold   = int(row['fold'])
        chrom  = row['chrom']
        cstart = int(row['centered_start'])
        cend   = int(row['centered_end'])

        ohe_path = (
            f"{OHE_ROOT}/fold{fold}/{chrom}_{cstart}_{cend}_X.pt"
        )
        if not os.path.exists(ohe_path):
            print(f"  [WARN] Missing OHE file: {ohe_path}, skipping.")
            continue

        # Load original OHE — shape [1, 4, L], keep as (4, L) numpy array
        raw = torch.load(ohe_path, map_location='cpu')
        if not isinstance(raw, torch.Tensor):
            raw = torch.tensor(raw)
        orig_ohe = raw.squeeze(0).float().detach().numpy()  # (4, L)

        # Parse CTCFs
        ctcf_list = parse_ctcf_coords(str(row.get('orig_CTCFs_coord', '')))

        # Predict original map
        orig_map = predict_map(model, orig_ohe, device)
        urq_orig = upper_right_quarter_mean(orig_map)

        base_record = {
            'chrom': chrom,
            'fold': fold,
            'centered_start': cstart,
            'centered_end': cend,
        }

        for n in INSERT_COUNTS:
            sites = find_insertion_sites(ctcf_list, n)
            if sites is None:
                print(
                    f"  [WARN] Not enough valid insertion sites for "
                    f"{chrom}:{cstart}-{cend} with {n} SINE(s); skipping."
                )
                records.append({
                    **base_record,
                    'n_insertions': n,
                    'URQ_orig': urq_orig,
                    'URQ_sine': np.nan,
                    'URQ_diff': np.nan,
                    'insertion_sites': None,
                })
                continue

            # Build modified OHE
            mod_ohe = orig_ohe.copy()
            for site in sites:
                mod_ohe = insert_sine_into_ohe(mod_ohe, site)

            mod_map = predict_map(model, mod_ohe, device)
            urq_sine = upper_right_quarter_mean(mod_map)

            print(
                f"  {chrom}:{cstart}-{cend}  n={n}  "
                f"URQ_orig={urq_orig:.4f}  URQ_sine={urq_sine:.4f}  "
                f"diff={urq_sine-urq_orig:.4f}"
            )
            records.append({
                **base_record,
                'n_insertions': n,
                'URQ_orig': urq_orig,
                'URQ_sine': urq_sine,
                'URQ_diff': urq_sine - urq_orig,
                'insertion_sites': str(sites),
            })

    # Save results
    out_df = pd.DataFrame(records)
    out_df.to_csv(OUT_TSV, sep='\t', index=False)
    print(f"\nResults saved to {OUT_TSV}  ({len(out_df)} rows)")


if __name__ == '__main__':
    main()