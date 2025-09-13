import torch
import numpy as np
import pandas as pd
from pyfaidx import Fasta
from Bio import SeqIO
import argparse
import matplotlib.pyplot as plt
import os
import sys

sys.path.append(os.path.abspath("/home1/smaruj/pytorch_akita/"))
from model_v2_compatible import SeqNN


def one_hot_encode(seq, alphabet="ACGT"):
    """One-hot encode DNA sequence into (len(seq), 4)."""
    mapping = {base: i for i, base in enumerate(alphabet)}
    arr = np.zeros((len(seq), len(alphabet)), dtype=np.float32)
    for i, base in enumerate(seq.upper()):
        if base in mapping:
            arr[i, mapping[base]] = 1.0
    return arr


def insert_fragment(background, fragment, insert_pos=None):
    """Replace bases in background with fragment at insert_pos (default middle)."""
    if insert_pos is None:
        insert_pos = len(background) // 2
    return background[:insert_pos] + fragment + background[insert_pos + len(fragment):]


def run_screening(model, genome, df, background_records, device, batch_size=16):
    """Run screening across all background sequences."""
    for b_idx, bg_record in enumerate(background_records):
        background_seq = str(bg_record.seq)
        print(f"[INFO] Background {b_idx+1} length: {len(background_seq)}")

        # One-hot encode background
        background_1hot = one_hot_encode(background_seq)
        background_tensor = torch.tensor(background_1hot.T).unsqueeze(0).to(device)

        # Baseline prediction
        with torch.no_grad():
            baseline_pred = model(background_tensor).cpu()

        scd_scores = []
        batch_tensors, batch_indices = [], []

        for idx, row in df.iterrows():
            # Fetch fragment from genome
            frag_seq = str(genome[row["chrom"]][row["start"]:row["end"]].seq)

            # Insert into background (middle)
            insert_pos = len(background_seq) // 2
            inserted_seq = insert_fragment(background_seq, frag_seq, insert_pos)

            # One-hot encode
            inserted_1hot = one_hot_encode(inserted_seq)
            inserted_tensor = torch.tensor(inserted_1hot.T).unsqueeze(0).to(device)

            # Collect into batch
            batch_tensors.append(inserted_tensor)
            batch_indices.append(idx)

            # Run batch
            if len(batch_tensors) == batch_size or idx == len(df) - 1:
                batch_input = torch.cat(batch_tensors, dim=0)  # (B, 4, L)
                with torch.no_grad():
                    batch_pred = model(batch_input).cpu()

                # Compute SCD for each
                for i, pred in enumerate(batch_pred):
                    diff = pred - baseline_pred.squeeze(0)
                    scd = torch.sqrt(torch.sum(diff ** 2)).item()
                    scd_scores.append((batch_indices[i], scd))

                # Reset
                batch_tensors, batch_indices = [], []

        # Store SCD in df
        scd_dict = {i: scd for i, scd in scd_scores}
        df[f"SCD_bg{b_idx}"] = df.index.map(scd_dict)

    # Average SCD across backgrounds
    scd_cols = [c for c in df.columns if c.startswith("SCD_bg")]
    df["SCD_avg"] = df[scd_cols].mean(axis=1)
    return df


def plot_scd(df, out_prefix):
    """Plot average SCD across region."""
    df["midpoint"] = (df["start"] + df["end"]) // 2
    plt.figure(figsize=(12, 4))
    plt.plot(df["midpoint"], df["SCD_avg"], marker="o", markersize=3, linewidth=1)
    plt.xlabel("Genomic position")
    plt.ylabel("Average SCD")
    plt.title("Average SCD across backgrounds")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(f"{out_prefix}_scd_plot.png", dpi=200)
    plt.close()


def main(args):
    # Device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    # Load model
    model = SeqNN()
    model.load_state_dict(torch.load(args.model, map_location=device))
    model.eval()

    # Load fragments
    df = pd.read_csv(args.fragments, sep="\t")

    # Load genome
    genome = Fasta(args.genome)

    # Load backgrounds
    background_records = list(SeqIO.parse(args.backgrounds, "fasta"))
    print(f"[INFO] Loaded {len(background_records)} background sequences.")

    # Run screening
    df = run_screening(model, genome, df, background_records, device, batch_size=args.batch_size)

    # Save results
    df.to_csv(args.out, sep="\t", index=False)
    print(f"[INFO] Saved results to {args.out}")

    # Plot average SCD
    out_prefix = os.path.splitext(args.out)[0]
    plot_scd(df, out_prefix)
    print(f"[INFO] Saved plot to {out_prefix}_scd_plot.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Mouse genomic region screening with SCD metric")
    parser.add_argument("--fragments", required=True, help="TSV file with genomic fragments")
    parser.add_argument("--backgrounds", required=True, help="FASTA file with background sequences")
    parser.add_argument("--genome", required=True, help="Reference genome FASTA (mm10)")
    parser.add_argument("--model", required=True, help="Trained PyTorch model file")
    parser.add_argument("--out", required=True, help="Output TSV file with results")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for prediction")

    args = parser.parse_args()
    main(args)
