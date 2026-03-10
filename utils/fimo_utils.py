import numpy as np
import torch
from memelite import fimo
import pandas as pd

def read_meme_pwm(filename):
    """Parse a MEME-format file and return the PWM as a (4, L) float32 tensor."""
    rows = []
    in_matrix = False
    with open(filename) as fh:
        for line in fh:
            line = line.strip()
            if line.startswith("letter-probability matrix"):
                in_matrix = True
                continue
            if in_matrix and line:
                if line.startswith("MOTIF"):
                    break
                rows.append([float(v) for v in line.split()])
    pwm = np.array(rows, dtype=np.float32)  # (L, 4)
    return torch.from_numpy(pwm.T)           # (4, L)


def run_fimo(seq_tensor, motifs_dict, threshold=1e-4):
    """Run FIMO on a (1, 4, L) tensor; returns the hits DataFrame."""
    arr = seq_tensor.cpu().detach().numpy()
    return fimo(motifs=motifs_dict, sequences=arr,
                threshold=threshold, reverse_complement=True)[0]


def ctcf_hits_per_seq(hits: pd.DataFrame, batch_size: int) -> list[dict]:
    """Summarise FIMO hits per sequence index within a batch."""
    records = []
    for seq_idx in range(batch_size):
        eh = hits[hits["sequence_name"] == seq_idx]
        if eh.empty:
            records.append(dict(n=0, score_sum=0.0, score_max=0.0,
                                positions=[], strands="no"))
        else:
            eh = eh.sort_values("start")
            records.append(dict(
                n         = len(eh),
                score_sum = float(eh["score"].sum()),
                score_max = float(eh["score"].max()),
                positions = [(int(s), int(e)) for s, e in zip(eh["start"], eh["end"])],
                strands   = "".join(eh["strand"].tolist()),
            ))
    return records


def ctcf_hits_from_fimo(fimo_df, seq_len=1310720, bin_size=2048):
    """Bin FIMO hits by strand → (hits_plus, hits_minus) arrays of length n_bins."""
    n_bins = seq_len // bin_size
    hits_plus  = np.zeros(n_bins)
    hits_minus = np.zeros(n_bins)
    for _, row in fimo_df.iterrows():
        bin_idx = int(row["start"] // bin_size)
        if bin_idx >= n_bins:
            continue
        if row["strand"] == "+":
            hits_plus[bin_idx]  += 1
        else:
            hits_minus[bin_idx] += 1
    return hits_plus, hits_minus


def hits_to_site_set(hits_df, bin_size=10):
    site_set = set()
    for _, row in hits_df.iterrows():
        start, end, strand = row['start'], row['end'], row['strand']
        center = (start + end) // 2
        binned_pos = round(center / bin_size) * bin_size
        site_set.add((binned_pos, strand))
    return site_set


def jaccard_index(set1, set2):
    if not set1 and not set2:
        return 1.0
    union = set1 | set2
    return len(set1 & set2) / len(union) if union else 0.0
