#!/usr/bin/env python
import torch
import numpy as np
import pandas as pd
from Bio import SeqIO
from tqdm import tqdm
import os
import argparse
import sys

# Add path to your model code
sys.path.append(os.path.abspath("/home1/smaruj/pytorch_akita/"))
from akita_model.model import SeqNN

# -------------------------
# Helper functions
# -------------------------

def one_hot_encode(seq, alphabet="ACGT"):
    """One-hot encode a DNA sequence into (len(seq), 4) numpy array."""
    mapping = {base: i for i, base in enumerate(alphabet)}
    arr = np.zeros((len(seq), len(alphabet)), dtype=np.float32)
    for i, base in enumerate(seq.upper()):
        if base in mapping:
            arr[i, mapping[base]] = 1.0
    return arr

def reverse_complement(ohe_seq):
    """Reverse complement of one-hot encoded sequence."""
    if ohe_seq.ndim == 2:
        ohe_seq = ohe_seq.unsqueeze(0)
    rc = torch.flip(ohe_seq, dims=[-1])
    rc = rc[:, [3, 2, 1, 0], :]
    if rc.shape[0] == 1:
        rc = rc.squeeze(0)
    return rc


def collect_ctcfs(df, ohe_dir, full_dir):
    collected_ctcfs = []

    for fold in range(8):
        fold_df = df[df["fold"] == fold]

        for _, row in tqdm(fold_df.iterrows(), total=len(fold_df), desc=f"Fold {fold}"):
            chrom = row["chrom"]
            cstart = row["centered_start"]
            cend = row["centered_end"]
            ctcf_start = row["ctcf_start"]
            ctcf_end = row["ctcf_end"]
            orientation = row["orientation"]

            # Paths
            slice_path = f"{ohe_dir}/fold{fold}/{chrom}_{cstart}_{cend}_slice.pt"
            full_path = f"{full_dir}/fold{fold}/{chrom}_{cstart}_{cend}_X.pt"

            if not os.path.exists(slice_path):
                print(f"⚠️ Missing: {slice_path}")
                continue

            # Always load on CPU
            ohe = torch.load(slice_path, map_location="cpu")  # (1, 4, L) or (4, L)
            if ohe.ndim == 2:
                ohe = ohe.unsqueeze(0)  # -> (1, 4, L)
            slice_len = ohe.shape[-1]

            # Handle cut CTCFs
            if ctcf_start < 0 or ctcf_end > slice_len:
                if not os.path.exists(full_path):
                    print(f"⚠️ Missing full seq: {full_path}")
                    continue
                full_ohe = torch.load(full_path, map_location="cpu")
                if full_ohe.ndim == 2:
                    full_ohe = full_ohe.unsqueeze(0)

                slice_mid = 320 * 2048
                slice_start = slice_mid - 2048 // 2
                slice_end = slice_mid + 2048 // 2

                parts = []
                if ctcf_start < 0:
                    parts.append(full_ohe[:, :, slice_start + ctcf_start : slice_start])
                    parts.append(ohe[:, :, :ctcf_end])
                elif ctcf_end > slice_len:
                    right_len = ctcf_end - slice_len
                    parts.append(ohe[:, :, ctcf_start:])
                    parts.append(full_ohe[:, :, slice_end : slice_end + right_len])
                ctcf_seq = torch.cat(parts, dim=2)
            else:
                ctcf_seq = ohe[:, :, ctcf_start:ctcf_end]

            if orientation == "-":
                ctcf_seq = reverse_complement(ctcf_seq)

            collected_ctcfs.append(ctcf_seq)

    return collected_ctcfs


def screen_ctcfs(collected_ctcfs, model, background_tensor, baseline_pred, device, batch_size=4):
    scd_scores = []
    batch_tensors = []
    batch_indices = []

    for idx, ctcf_ohe in enumerate(collected_ctcfs):
        if ctcf_ohe.ndim == 3:
            ctcf_ohe = ctcf_ohe[0]

        inserted_tensor = background_tensor.clone()
        insert_pos = inserted_tensor.shape[2] // 2
        frag_len = ctcf_ohe.shape[1]
        start = max(0, insert_pos - frag_len // 2)
        end = min(inserted_tensor.shape[2], start + frag_len)
        ctcf_slice = ctcf_ohe[:, :end-start]
        inserted_tensor[:, :, start:end] = ctcf_slice.unsqueeze(0)

        batch_tensors.append(inserted_tensor.to(device))
        batch_indices.append(idx)

        if len(batch_tensors) == batch_size or idx == len(collected_ctcfs) - 1:
            batch_input = torch.cat(batch_tensors, dim=0)
            with torch.no_grad():
                batch_pred = model(batch_input).cpu()
            for i, pred in enumerate(batch_pred):
                diff = pred - baseline_pred.squeeze(0)
                scd = torch.sqrt(torch.sum(diff ** 2)).item()
                scd_scores.append((batch_indices[i], scd))
            batch_tensors, batch_indices = [], []

    # return only SCD values in order
    scd_values = [score for _, score in sorted(scd_scores, key=lambda x: x[0])]
    return scd_values

# -------------------------
# Main function
# -------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ctcf_df", required=True, help="Path to preexisting_ctcf_df.tsv")
    parser.add_argument("--slice_dir", required=True, help="Directory containing slice tensors")
    parser.add_argument("--fullX_dir", required=True, help="Directory containing X tensors")
    parser.add_argument("--background_fasta", required=True, help="FASTA with background sequences")
    parser.add_argument("--model_path", required=True, help="Trained PyTorch model path")
    parser.add_argument("--output_df", required=True, help="Output TSV path with SCD columns")
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = SeqNN()
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()

    df = pd.read_csv(args.ctcf_df, sep="\t")

    # collecting CTCFs
    collected_ctcfs = collect_ctcfs(df, args.slice_dir, args.fullX_dir)
    
    # Load all background sequences
    backgrounds = list(SeqIO.parse(args.background_fasta, "fasta"))

    for i, bg_record in enumerate(backgrounds):
        print(f"Processing background {i+1}/{len(backgrounds)}: {bg_record.id}")
        bg_seq = str(bg_record.seq)
        bg_1hot = one_hot_encode(bg_seq)
        bg_tensor = torch.tensor(bg_1hot.T).unsqueeze(0).to(device)
        with torch.no_grad():
            baseline_pred = model(bg_tensor).cpu()

        scd_values = screen_ctcfs(collected_ctcfs, model, bg_tensor, baseline_pred, device)

        # Add as new column
        df[f"SCD_bg{i}"] = scd_values

    # Save updated df
    df.to_csv(args.output_df, sep="\t", index=False)
    print(f"Saved SCD results to {args.output_df}")

if __name__ == "__main__":
    main()
