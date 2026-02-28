import numpy as np
import torch
from memelite import fimo

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