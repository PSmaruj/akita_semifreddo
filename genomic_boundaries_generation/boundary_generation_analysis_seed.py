#!/usr/bin/env python3

import os
import sys
import argparse
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tangermeme.tools import fimo

sys.path.append(os.path.abspath("/home1/smaruj/pytorch_akita/"))
from model_v2_compatible import SeqNN


# ----------------------------------------------------------------------
#                           DATASET CLASSES
# ----------------------------------------------------------------------

class OriginalDataset(Dataset):
    def __init__(self, coord_df, init_seq_path):
        self.coords = coord_df
        self.init_seq_path = init_seq_path
    
    def __len__(self):
        return len(self.coords)

    def __getitem__(self, idx):
        row = self.coords.iloc[idx]
        chrom = row["chrom"]
        start = row["centered_start"]
        end = row["centered_end"]

        X = torch.load(f"{self.init_seq_path}{chrom}_{start}_{end}_X.pt", weights_only=True)
        X = X.squeeze(0)
        return X
    
    
class BoundaryGenerationDataset(Dataset):
    def __init__(self, coord_df, init_seq_path, slice_path, slice=256, cropping=64, bin_size=2048):
        self.coords = coord_df
        self.init_seq_path = init_seq_path
        self.slice_path = slice_path
        self.slice = slice
        self.cropping = cropping
        self.bin_size = bin_size
        
    def __len__(self):
        return len(self.coords)

    def __getitem__(self, idx):
        row = self.coords.iloc[idx]
        chrom = row["chrom"]
        start = row["centered_start"]
        end = row["centered_end"]

        X = torch.load(f"{self.init_seq_path}{chrom}_{start}_{end}_X.pt", weights_only=True)
        slice = torch.load(f"{self.slice_path}{chrom}_{start}_{end}_slice.pt", weights_only=True)
        
        edit_start = (self.slice + self.cropping) * self.bin_size
        edit_end = edit_start + self.bin_size
        
        editedX = X.clone()
        editedX[:,:, edit_start:edit_end] = slice
        
        editedX = editedX.squeeze(0)
        
        return editedX


class TriuMatrixDataset(Dataset):
    def __init__(self, coord_df, map_path):
        """
        coord_df: DataFrame with ["chrom", "centered_start", "centered_end"]
        map_path: Directory containing upper-triangle map tensors (e.g. chr1_1000000_1051200_target.pt)
        """
        self.coords = coord_df
        self.map_path = map_path

    def __len__(self):
        return len(self.coords)

    def __getitem__(self, idx):
        row = self.coords.iloc[idx]
        chrom = row["chrom"]
        start = row["centered_start"]
        end = row["centered_end"]

        file_name = f"{chrom}_{start}_{end}_target.pt"
        file_path = os.path.join(self.map_path, file_name)

        # Load the flat upper-triangular vector
        triu_tensor = torch.load(file_path, map_location='cpu')

        # triu_tensor = triu_tensor.squeeze()

        return triu_tensor

# ----------------------------------------------------------------------
#                            HELPER FUNCTIONS
# ----------------------------------------------------------------------    

def set_diag(matrix, value, k):
    """Set diagonal `k` of a matrix to `value`."""
    rows, cols = matrix.shape
    for i in range(rows):
        if 0 <= i + k < cols:
            matrix[i, i + k] = value


def from_upper_triu_batch(batch_vectors, matrix_len=512, num_diags=2):
    """Convert a batch of upper-triangular vectors into symmetric matrices with np.nan on diagonals."""
    if isinstance(batch_vectors, torch.Tensor):
        batch_vectors = batch_vectors.detach().cpu().numpy()

    batch_size = len(batch_vectors)
    matrices = np.zeros((batch_size, matrix_len, matrix_len), dtype=np.float32)

    triu_indices = np.triu_indices(matrix_len, num_diags)

    for i in range(batch_size):
        matrices[i][triu_indices] = batch_vectors[i][0,:]
        # Mirror to lower triangle
        matrices[i] = matrices[i] + matrices[i].T

        # Set diagonals to np.nan
        for k in range(-num_diags + 1, num_diags):
            set_diag(matrices[i], np.nan, k)

    return matrices  # shape: [B, 512, 512]


def read_meme_pwm_as_numpy(filename):
    pwm_list = []  # List to store PWM rows
    
    with open(filename, 'r') as file:
        in_matrix_section = False
        
        for line in file:
            line = line.strip()
            
            # Check if we are reading the PWM matrix
            if line.startswith("letter-probability matrix"):
                in_matrix_section = True  # Start reading matrix data
                continue  # Skip this header line
            
            # If we are in the matrix section, process the rows
            if in_matrix_section and line:
                pwm_row = [float(value) for value in line.split()]  # Parse values
                pwm_list.append(pwm_row)  # Append to the PWM list
            
            # If we encounter a new MOTIF or the end of file, stop matrix reading
            if line.startswith("MOTIF") and in_matrix_section:
                break
    
    # Convert the list to a numpy array
    pwm_array = np.array(pwm_list)
    
    return pwm_array

# ----------------------------------------------------------------------
#                                MAIN
# ----------------------------------------------------------------------
            
def run_pipeline(seed, fold, target_c):
    # Load coordinate table
    df = pd.read_csv(
        f"/scratch1/smaruj/generate_genomic_boundary/seeds_default_l/"
        f"seed{seed}_fold{fold}_{target_c}_genomic_windows_table_steps.tsv",
        sep="\t"
    )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Load model
    model = SeqNN()
    model.load_state_dict(
        torch.load(
            "/scratch1/smaruj/Akita_pytorch_models/finetuned/mouse_models/"
            "Hsieh2019_mESC/models/Akita_v2_mouse_Hsieh2019_mESC_model0_finetuned.pth",
            map_location=device,
        )
    )
    model.to(device).eval()   

    # Paths
    init_seq_path = f"/scratch1/smaruj/generate_genomic_boundary/ohe_X/fold{fold}/"
    slice_path = f"/scratch1/smaruj/generate_genomic_boundary/seeds_default_l/seed{seed}/"
    target_path = f"/scratch1/smaruj/generate_genomic_boundary/targets/target_{target_c}/fold{fold}/"

    # Datasets
    orig_loader = DataLoader(OriginalDataset(df, init_seq_path), batch_size=4, shuffle=False)
    edited_loader = DataLoader(
        BoundaryGenerationDataset(df, init_seq_path, slice_path),
        batch_size=4, shuffle=False
    )
    target_loader = DataLoader(TriuMatrixDataset(df, target_path), batch_size=4, shuffle=False)

    # Parameters
    slice_len, cropping, bin_size = 256, 64, 2048
    edit_start = (slice_len + cropping) * bin_size
    edit_end = edit_start + bin_size

    # Storage lists
    scd_values = []
    urq_values = []
    og_urq_values = []
    target_urq_values = []
    edit_counts = []
    seq_GC_content, slice_GC_content, edited_GC_content = [], [], []

    preds_all_orig, preds_all_edited, targets_all = [], [], []

    # ----------------------------
    # Forward Pass
    # ----------------------------
    with torch.no_grad():
        for orig_batch, edited_batch, target_batch in zip(orig_loader, edited_loader, target_loader):
            orig_batch = orig_batch.to(device)
            edited_batch = edited_batch.to(device)
            target_batch = target_batch.squeeze(1).to(device)

            # GC computations
            gc_all = orig_batch[:, 1:3, :].sum(dim=1) / orig_batch.sum(dim=1)
            gc_slice = orig_batch[:, 1:3, edit_start:edit_end].sum(dim=1) / orig_batch[:, :, edit_start:edit_end].sum(dim=1)
            gc_edit = edited_batch[:, 1:3, edit_start:edit_end].sum(dim=1) / edited_batch[:, :, edit_start:edit_end].sum(dim=1)

            # seq_GC_content.extend(gc_all.mean(dim=1).cpu())
            # slice_GC_content.extend(gc_slice.mean(dim=1).cpu())
            # edited_GC_content.extend(gc_edit.mean(dim=1).cpu())
            seq_GC_content.extend(gc_all.mean(dim=1).cpu().numpy().tolist())
            slice_GC_content.extend(gc_slice.mean(dim=1).cpu().numpy().tolist())
            edited_GC_content.extend(gc_edit.mean(dim=1).cpu().numpy().tolist())
            
            # Edits
            diffs = (orig_batch[:, :, edit_start:edit_end] - edited_batch[:, :, edit_start:edit_end]).abs()
            num_edits = (diffs.sum(dim=(1, 2)) / 2).cpu().numpy()
            edit_counts.extend(num_edits)

            # Predictions
            preds_orig = model(orig_batch).cpu()
            preds_edit = model(edited_batch).cpu()

            preds_all_orig.extend(preds_orig)
            preds_all_edited.extend(preds_edit)
            targets_all.extend(target_batch.cpu())

            # SCD
            scd = torch.sqrt(((preds_edit - preds_orig) ** 2).sum(dim=(1, 2)))
            scd_values.extend(scd.numpy())

            # UR quadrant
            orig_maps = from_upper_triu_batch(preds_orig)
            edit_maps = from_upper_triu_batch(preds_edit)
            target_maps = from_upper_triu_batch(target_batch.cpu())

            urq_values.extend(np.nanmean(edit_maps[:, 0:250, 260:512], axis=(1, 2)))
            og_urq_values.extend(np.nanmean(orig_maps[:, 0:250, 260:512], axis=(1, 2)))
            target_urq_values.extend(np.nanmean(target_maps[:, 0:250, 260:512], axis=(1, 2)))

    # ----------------------------
    # CTCF Motif Analysis (FIMO)
    # ----------------------------
    pwm = read_meme_pwm_as_numpy("/home1/smaruj/IterativeMutagenesis/MA0139.1.meme")
    pwm_tensor = torch.from_numpy(pwm.T).float()
    motifs = {"CTCF": pwm_tensor}

    orig_num, num_ctcfs, sum_fimo, max_fimo, positions, strand_strings = [], [], [], [], [], []
    extra_flank = 60

    with torch.no_grad():
        for orig_batch, edited_batch in zip(orig_loader, edited_loader):
            orig_batch = orig_batch.to(device)
            edited_batch = edited_batch.to(device)

            orig_slice = orig_batch[:, :, edit_start-extra_flank : edit_end+extra_flank]
            edit_slice = edited_batch[:, :, edit_start-extra_flank : edit_end+extra_flank]

            # Original hits
            orig_hits = fimo.fimo(motifs, orig_slice, threshold=1e-4, reverse_complement=True)[0]
            orig_hits["start"] -= extra_flank
            orig_hits["end"] -= extra_flank

            for seq_idx in range(4):
                seq_hits = orig_hits[orig_hits["sequence_name"] == seq_idx]
                orig_num.append(len(seq_hits) if not seq_hits.empty else 0)

            # Edited hits
            hits = fimo.fimo(motifs, edit_slice, threshold=1e-4, reverse_complement=True)[0]
            hits["start"] -= extra_flank
            hits["end"] -= extra_flank

            for seq_idx in range(4):
                seq_hits = hits[hits["sequence_name"] == seq_idx]
                if seq_hits.empty:
                    num_ctcfs.append(0)
                    sum_fimo.append(0.0)
                    max_fimo.append(0.0)
                    positions.append(tuple())
                    strand_strings.append("no")
                    continue

                seq_hits = seq_hits.sort_values("start")
                num_ctcfs.append(len(seq_hits))
                sum_fimo.append(seq_hits["score"].sum())
                max_fimo.append(seq_hits["score"].max())
                positions.append(list(zip(seq_hits["start"], seq_hits["end"])))
                strand_strings.append("".join(seq_hits["strand"].tolist()))

    # ----------------------------
    # Write outputs
    # ----------------------------
    df["SCD"] = scd_values
    df["URQ_result"] = urq_values
    df["URQ_target"] = target_urq_values
    df["URQ_init"] = og_urq_values
    df["num_edits"] = edit_counts
    df["GC_seq"] = seq_GC_content
    df["GC_slice"] = slice_GC_content
    df["GC_slice_edited"] = edited_GC_content

    df["init_CTCFs_num"] = orig_num[:len(df)]
    df["CTCFs_num"] = num_ctcfs[:len(df)]
    df["FIMO_sum"] = sum_fimo[:len(df)]
    df["FIMO_max"] = max_fimo[:len(df)]
    df["orientation"] = strand_strings[:len(df)]
    df["positions"] = positions[:len(df)]

    out_path = (
        f"/scratch1/smaruj/generate_genomic_boundary/seeds_default_l/"
        f"seed{seed}_fold{fold}_{target_c}_genomic_windows_table_results.tsv"
    )
    df.to_csv(out_path, sep="\t", index=False)
    print(f"✓ Results saved to {out_path}")

# ----------------------------
# CLI
# ----------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Run boundary generation analysis.")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--target_c", type=float, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    run_pipeline(args.seed, args.fold, args.target_c)


if __name__ == "__main__":
    main()
