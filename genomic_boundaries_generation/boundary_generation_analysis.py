#!/usr/bin/env python3
import argparse
import os
import sys
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

sys.path.append(os.path.abspath("/home1/smaruj/pytorch_akita/"))

from model_v2_compatible import SeqNN
from memelite import fimo


# -----------------------------
# Dataset classes
# -----------------------------

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


# -----------------------------
# Helper functions
# -----------------------------

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


# -----------------------------
# Main pipeline
# -----------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--target_c", type=float, required=True)
    args = parser.parse_args()

    FOLD = args.fold
    TARGET_C = args.target_c

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # -----------------------------
    # Load dataframe
    # -----------------------------
    df_path = f"/scratch1/smaruj/generate_genomic_boundary/results/target_{TARGET_C}/fold{FOLD}_{TARGET_C}_genomic_windows_table_steps.tsv"
    df = pd.read_csv(df_path, sep="\t")

    # -----------------------------
    # Load model
    # -----------------------------
    model = SeqNN()
    model_path = (
        "/scratch1/smaruj/Akita_pytorch_models/finetuned/mouse_models/"
        "Hsieh2019_mESC/models/Akita_v2_mouse_Hsieh2019_mESC_model0_finetuned.pth"
    )
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # -----------------------------
    # Datasets + Loaders
    # -----------------------------
    orig_dataset = OriginalDataset(df, f"/scratch1/smaruj/generate_genomic_boundary/ohe_X/fold{FOLD}/")
    edited_dataset = BoundaryGenerationDataset(
        df,
        f"/scratch1/smaruj/generate_genomic_boundary/ohe_X/fold{FOLD}/",
        f"/scratch1/smaruj/generate_genomic_boundary/results/target_{TARGET_C}/fold{FOLD}/"
    )
    target_dataset = TriuMatrixDataset(
        df,
        f"/scratch1/smaruj/generate_genomic_boundary/targets/target_{TARGET_C}/fold{FOLD}/"
    )
    
    batch_size = 4
    orig_loader = DataLoader(orig_dataset, batch_size=batch_size, shuffle=False)
    edited_loader = DataLoader(edited_dataset, batch_size=batch_size, shuffle=False)
    target_loader = DataLoader(target_dataset, batch_size=batch_size, shuffle=False)

    # Window settings
    slice_len = 256
    cropping = 64
    bin_size = 2048

    edit_start = (slice_len + cropping) * bin_size
    edit_end = edit_start + bin_size

    # -----------------------------
    # Storage lists
    # -----------------------------
    preds_all_orig = []
    preds_all_edited = []
    targets_all = []

    scd_values = []
    urq_mean_values = []
    og_urq_mean_values = []
    target_urq_mean_values = []

    edit_counts = []
    seq_GC_content = []
    slice_GC_content = []
    edited_GC_content = []
    
    # -----------------------------
    # Prediction loop
    # -----------------------------
    with torch.no_grad():
        for orig_batch, edited_batch, target_batch in zip(orig_loader, edited_loader, target_loader):
            orig_batch = orig_batch.to(device)
            edited_batch = edited_batch.to(device)
            target_batch = target_batch.to(device).squeeze(1)

            # Compute GC content
            gc_all = orig_batch[:, 1:3, :].sum(dim=1) / orig_batch.sum(dim=1)
            gc_slice = orig_batch[:, 1:3, edit_start:edit_end].sum(dim=1) / orig_batch[:, :, edit_start:edit_end].sum(dim=1)
            gc_slice_edit = edited_batch[:, 1:3, edit_start:edit_end].sum(dim=1) / edited_batch[:, :, edit_start:edit_end].sum(dim=1)

            seq_GC_content.extend(gc_all.mean(dim=1).cpu().numpy())
            slice_GC_content.extend(gc_slice.mean(dim=1).cpu().numpy())
            edited_GC_content.extend(gc_slice_edit.mean(dim=1).cpu().numpy())

            # Edit count
            diffs = torch.abs(orig_batch[:, :, edit_start:edit_end] - edited_batch[:, :, edit_start:edit_end])
            num_flips = diffs.sum(dim=(1, 2))
            edit_counts.extend((num_flips / 2).cpu().numpy())

            preds_orig = model(orig_batch).cpu()
            preds_edited = model(edited_batch).cpu()

            preds_all_orig.extend(preds_orig)
            preds_all_edited.extend(preds_edited)
            targets_all.extend(target_batch.cpu())

            # SCD
            scd_batch = torch.sqrt(((preds_edited - preds_orig) ** 2).sum(dim=(1, 2)))
            scd_values.extend(scd_batch.numpy())

            # URQ means
            orig_maps = from_upper_triu_batch(preds_orig)
            edited_maps = from_upper_triu_batch(preds_edited)
            target_maps = from_upper_triu_batch(target_batch.cpu())

            urq_mean_values.extend(np.nanmean(edited_maps[:, 0:250, 260:512], axis=(1, 2)))
            og_urq_mean_values.extend(np.nanmean(orig_maps[:, 0:250, 260:512], axis=(1, 2)))
            target_urq_mean_values.extend(np.nanmean(target_maps[:, 0:250, 260:512], axis=(1, 2)))

    # -----------------------------
    # CTCF scoring
    # -----------------------------
    pwm_path = "/home1/smaruj/IterativeMutagenesis/MA0139.1.meme"
    pwm = read_meme_pwm_as_numpy(pwm_path)
    pwm_tensor = torch.from_numpy(pwm.T).float()
    motifs = {"CTCF": pwm_tensor}

    orig_num_CTCFs = []
    num_CTCFs = []
    sum_FIMO = []
    max_FIMO = []
    strand_strings = []
    positions = []

    extra_flank = 60
    
    with torch.no_grad():
        for orig_batch, edited_batch in zip(orig_loader, edited_loader):
            orig_batch = orig_batch.to(device)
            edited_batch = edited_batch.to(device)

            orig_slice = orig_batch[:, :, edit_start-extra_flank:edit_end+extra_flank].cpu().numpy()
            edited_slice = edited_batch[:, :, edit_start-extra_flank:edit_end+extra_flank].cpu().numpy()

            orig_hits = fimo(motifs=motifs, sequences=orig_slice, threshold=1e-4, reverse_complement=True)[0]
            orig_hits["start"] -= extra_flank
            orig_hits["end"] -= extra_flank

            edited_hits = fimo(motifs=motifs, sequences=edited_slice, threshold=1e-4, reverse_complement=True)[0]
            edited_hits["start"] -= extra_flank
            edited_hits["end"] -= extra_flank

            for seq_idx in range(batch_size):

                # Original
                oh = orig_hits[orig_hits["sequence_name"] == seq_idx]
                orig_num_CTCFs.append(len(oh) if not oh.empty else 0)

                # Edited
                eh = edited_hits[edited_hits["sequence_name"] == seq_idx]
                if eh.empty:
                    num_CTCFs.append(0)
                    sum_FIMO.append(0.0)
                    max_FIMO.append(0.0)
                    positions.append(tuple())
                    strand_strings.append("no")
                else:
                    eh = eh.sort_values(by="start")
                    num_CTCFs.append(len(eh))
                    sum_FIMO.append(eh["score"].sum())
                    max_FIMO.append(eh["score"].max())
                    positions.append([(s, e) for s, e in zip(eh["start"], eh["end"])])
                    strand_strings.append("".join(eh["strand"].tolist()))
    
    # -----------------------------
    # Final merge + Save
    # -----------------------------
    df["SCD"] = scd_values
    df["URQ_result"] = urq_mean_values
    df["URQ_target"] = target_urq_mean_values
    df["URQ_init"] = og_urq_mean_values
    df["num_edits"] = edit_counts
    df["GC_seq"] = seq_GC_content
    df["GC_slice"] = slice_GC_content
    df["GC_slice_edited"] = edited_GC_content
    df["init_CTCFs_num"] = orig_num_CTCFs[:len(df)]
    df["CTCFs_num"] = num_CTCFs[:len(df)]
    df["FIMO_sum"] = sum_FIMO[:len(df)]
    df["FIMO_max"] = max_FIMO[:len(df)]
    df["orientation"] = strand_strings[:len(df)]
    df["positions"] = positions[:len(df)]

    out_path = f"/scratch1/smaruj/generate_genomic_boundary/results/target_{TARGET_C}/fold{FOLD}_{TARGET_C}_genomic_windows_table_results.tsv"
    df.to_csv(out_path, sep="\t", index=False)
    print(f"Saved results → {out_path}")

# -----------------------------
# Entry point
# -----------------------------
if __name__ == "__main__":
    main()