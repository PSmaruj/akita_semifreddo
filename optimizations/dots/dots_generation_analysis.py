#!/usr/bin/env python3
import argparse
import os
import sys
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

sys.path.append(os.path.abspath("/home1/smaruj/pytorch_akita/"))

from akita_model.model import SeqNN
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
 
    
class DotsGenerationDataset(Dataset):
    def __init__(self, coord_df, init_seq_path, slice_path, slice0=256-25, slice1=256+25, cropping=64, bin_size=2048):
        self.coords = coord_df
        self.init_seq_path = init_seq_path
        self.slice_path = slice_path
        self.slice0 = slice0
        self.slice1 = slice1
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
        Eslice0 = torch.load(f"{self.slice_path}{chrom}_{start}_{end}_slice0.pt", weights_only=True)
        Eslice1 = torch.load(f"{self.slice_path}{chrom}_{start}_{end}_slice1.pt", weights_only=True)
        
        edit_start0 = (self.slice0 + self.cropping) * self.bin_size
        edit_end0 = edit_start0 + self.bin_size
        
        edit_start1 = (self.slice1 + self.cropping) * self.bin_size
        edit_end1 = edit_start1 + self.bin_size
        
        editedX = X.clone()
        editedX[:,:, edit_start0:edit_end0] = Eslice0
        editedX[:,:, edit_start1:edit_end1] = Eslice1
        
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
    parser.add_argument("--dist", type=float, required=True)
    args = parser.parse_args()

    FOLD = args.fold
    DIST = int(args.dist)
    TARGET_C = 1.0
    HALF_DIST = DIST // 2
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # -----------------------------
    # Load dataframe
    # -----------------------------
    df_path = f"/scratch1/smaruj/generate_genomic_dot/results/dist_{DIST}bins/fold{FOLD}_{TARGET_C}_genomic_windows_table_steps.tsv"
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
    
    edited_dataset = DotsGenerationDataset(
        df, 
        f"/scratch1/smaruj/generate_genomic_boundary/ohe_X/fold{FOLD}/", 
        f"/scratch1/smaruj/generate_genomic_dot/results/dist_{DIST}bins/fold{FOLD}/",
        slice0=256-HALF_DIST, slice1=256+HALF_DIST
        )
    
    target_dataset = TriuMatrixDataset(
        df, 
        f"/scratch1/smaruj/generate_genomic_dot/targets/target_{TARGET_C}_{DIST}bins/fold{FOLD}/"
        )
    
    batch_size = 4
    orig_loader = DataLoader(orig_dataset, batch_size=batch_size, shuffle=False)
    edited_loader = DataLoader(edited_dataset, batch_size=batch_size, shuffle=False)
    target_loader = DataLoader(target_dataset, batch_size=batch_size, shuffle=False)

    # Window settings
    slice0 = 256 - HALF_DIST
    slice1 = 256 + HALF_DIST
    cropping = 64
    bin_size = 2048

    edit_start0 = (slice0 + cropping) * bin_size
    edit_end0 = edit_start0 + bin_size
    
    edit_start1 = (slice1 + cropping) * bin_size
    edit_end1 = edit_start1 + bin_size

    # -----------------------------
    # Storage lists
    # -----------------------------
    preds_all_orig = []
    preds_all_edited = []
    targets_all = []
    
    scd_values = []

    dot7_mean_values = []
    og_dot7_mean_values = []
    target_dot7_mean_values = []

    dot11_mean_values = []
    og_dot11_mean_values = []
    target_dot11_mean_values = []

    dot15_mean_values = []
    og_dot15_mean_values = []
    target_dot15_mean_values = []

    edit_counts0 = []
    edit_counts1 = []
    
    seq_GC_content = []
    slice0_GC_content = []
    edited0_GC_content = []
    slice1_GC_content = []
    edited1_GC_content = []
    
    # -----------------------------
    # Prediction loop
    # -----------------------------
    with torch.no_grad():
        for orig_batch, edited_batch, target_batch in zip(orig_loader, edited_loader, target_loader):
            orig_batch = orig_batch.to(device)
            edited_batch = edited_batch.to(device)
            target_batch = target_batch.to(device)
            target_batch = target_batch.squeeze(1)
        
            # Compute GC content
            gc_content_all = orig_batch[:, 1:3, :].sum(dim=1) / orig_batch.sum(dim=1)  # [B, L]
            gc_content_slice0 = orig_batch[:, 1:3, edit_start0:edit_end0].sum(dim=1) / orig_batch[:, :, edit_start0:edit_end0].sum(dim=1)  # [B, edited_L]
            gc_content_slice1 = orig_batch[:, 1:3, edit_start1:edit_end1].sum(dim=1) / orig_batch[:, :, edit_start1:edit_end1].sum(dim=1)  # [B, edited_L]
            
            gc_content_slice_edit0 = edited_batch[:, 1:3, edit_start0:edit_end0].sum(dim=1) / edited_batch[:, :, edit_start0:edit_end0].sum(dim=1)  # [B, edited_L]
            gc_content_slice_edit1 = edited_batch[:, 1:3, edit_start1:edit_end1].sum(dim=1) / edited_batch[:, :, edit_start1:edit_end1].sum(dim=1)  # [B, edited_L]
            
            # Mean GC content per sequence
            mean_gc_all = gc_content_all.mean(dim=1).cpu().numpy()  # [B]
            mean_gc_slice0 = gc_content_slice0.mean(dim=1).cpu().numpy()  # [B]
            mean_gc_slice1 = gc_content_slice1.mean(dim=1).cpu().numpy()  # [B]
            
            mean_gc_edit0 = gc_content_slice_edit0.mean(dim=1).cpu().numpy()  # [B]
            mean_gc_edit1 = gc_content_slice_edit1.mean(dim=1).cpu().numpy()  # [B]
            
            seq_GC_content.extend(mean_gc_all)
            slice0_GC_content.extend(mean_gc_slice0)
            slice1_GC_content.extend(mean_gc_slice1)
            
            edited0_GC_content.extend(mean_gc_edit0)
            edited1_GC_content.extend(mean_gc_edit1)
            
            diffs0 = torch.abs(orig_batch[:, :, edit_start0:edit_end0] - edited_batch[:, :, edit_start0:edit_end0])  # shape [B, 4, region_len]
            num_flips0 = diffs0.sum(dim=(1, 2))  # total bit flips per sequence
            num_edits0 = (num_flips0 / 2).cpu().numpy()  # divide by 2 to get base edits

            diffs1 = torch.abs(orig_batch[:, :, edit_start1:edit_end1] - edited_batch[:, :, edit_start1:edit_end1])  # shape [B, 4, region_len]
            num_flips1 = diffs1.sum(dim=(1, 2))  # total bit flips per sequence
            num_edits1 = (num_flips1 / 2).cpu().numpy()  # divide by 2 to get base edits
            
            edit_counts0.extend(num_edits0)
            edit_counts1.extend(num_edits1)
            
            preds_orig = model(orig_batch).cpu()
            preds_edited = model(edited_batch).cpu()

            preds_all_orig.extend(preds_orig)
            preds_all_edited.extend(preds_edited)
            targets_all.extend(target_batch)
            
            scd_batch = torch.sqrt(((preds_edited - preds_orig) ** 2).sum(dim=(1, 2)))  # [B]
            scd_values.extend(scd_batch.numpy())
            
            orig_maps = from_upper_triu_batch(preds_orig)
            edited_maps = from_upper_triu_batch(preds_edited)
            
            dot_r = 256 - HALF_DIST
            dot_c = 256 + HALF_DIST

            # DOT-7 (3+1+3=7)
            dot_width_half = 3
            
            og_dot = np.nanmean(orig_maps[:, dot_r-dot_width_half:dot_r+dot_width_half, dot_c-dot_width_half:dot_c+dot_width_half], axis=(1, 2))        
            E_dot = np.nanmean(edited_maps[:, dot_r-dot_width_half:dot_r+dot_width_half, dot_c-dot_width_half:dot_c+dot_width_half], axis=(1, 2))
            
            og_dot7_mean_values.extend(og_dot)
            dot7_mean_values.extend(E_dot)
            
            target_maps = from_upper_triu_batch(target_batch)
            target_dot = np.nanmean(target_maps[:, dot_r-dot_width_half:dot_r+dot_width_half, dot_c-dot_width_half:dot_c+dot_width_half], axis=(1, 2))
            target_dot7_mean_values.extend(target_dot)

            # DOT-11 (5+1+5=11)
            dot_width_half = 5
            
            og_dot = np.nanmean(orig_maps[:, dot_r-dot_width_half:dot_r+dot_width_half, dot_c-dot_width_half:dot_c+dot_width_half], axis=(1, 2))        
            E_dot = np.nanmean(edited_maps[:, dot_r-dot_width_half:dot_r+dot_width_half, dot_c-dot_width_half:dot_c+dot_width_half], axis=(1, 2))
            
            og_dot11_mean_values.extend(og_dot)
            dot11_mean_values.extend(E_dot)
            
            target_maps = from_upper_triu_batch(target_batch)
            target_dot = np.nanmean(target_maps[:, dot_r-dot_width_half:dot_r+dot_width_half, dot_c-dot_width_half:dot_c+dot_width_half], axis=(1, 2))
            target_dot11_mean_values.extend(target_dot)      
            
            # DOT-15 (7+1+7=15)
            dot_width_half = 7
            
            og_dot = np.nanmean(orig_maps[:, dot_r-dot_width_half:dot_r+dot_width_half, dot_c-dot_width_half:dot_c+dot_width_half], axis=(1, 2))        
            E_dot = np.nanmean(edited_maps[:, dot_r-dot_width_half:dot_r+dot_width_half, dot_c-dot_width_half:dot_c+dot_width_half], axis=(1, 2))
            
            og_dot15_mean_values.extend(og_dot)
            dot15_mean_values.extend(E_dot)
            
            target_maps = from_upper_triu_batch(target_batch)
            target_dot = np.nanmean(target_maps[:, dot_r-dot_width_half:dot_r+dot_width_half, dot_c-dot_width_half:dot_c+dot_width_half], axis=(1, 2))
            target_dot15_mean_values.extend(target_dot)  
            
            
    # -----------------------------
    # CTCF scoring
    # -----------------------------
    pwm_path = "/home1/smaruj/IterativeMutagenesis/MA0139.1.meme"
    pwm = read_meme_pwm_as_numpy(pwm_path)
    pwm_tensor = torch.from_numpy(pwm.T).float()
    motifs = {"CTCF": pwm_tensor}

    # slice 0
    orig_num_CTCFs_0 = []
    num_CTCFs_0 = []
    sum_FIMO_0 = []
    max_FIMO_0 = []
    strand_strings_0 = []
    positions_0 = []
    
    # slice 1
    orig_num_CTCFs_1 = []
    num_CTCFs_1 = []
    sum_FIMO_1 = []
    max_FIMO_1 = []
    strand_strings_1 = []
    positions_1 = []

    extra_flank = 60
    
    with torch.no_grad():
        for orig_batch, edited_batch in zip(orig_loader, edited_loader):
            
            orig_batch = orig_batch.to(device)
            edited_batch = edited_batch.to(device)
            
            orig_slice = orig_batch[:, :, edit_start0-extra_flank:edit_end0+extra_flank]
            edited_slice = edited_batch[:, :, edit_start0-extra_flank:edit_end0+extra_flank]
            
            orig_slice_np = orig_slice.cpu().numpy()
            edited_slice_np = edited_slice.cpu().numpy()
            
            orig_hits = fimo(
                motifs=motifs,
                sequences=orig_slice_np,
                threshold=1e-4,
                reverse_complement=True
            )[0]
            
            orig_hits["start"] -= extra_flank
            orig_hits["end"] -= extra_flank
            
            for seq_idx in range(batch_size):
                seq_hits = orig_hits[orig_hits["sequence_name"] == seq_idx]
                
                if not seq_hits.empty:
                    orig_num_CTCFs_0.append(len(seq_hits))
                else:
                    orig_num_CTCFs_0.append(0)
            
            hits = fimo(
                motifs=motifs,
                sequences=edited_slice_np,
                threshold=1e-4,
                reverse_complement=True
            )[0]

            hits["start"] -= extra_flank
            hits["end"] -= extra_flank
            
            for seq_idx in range(batch_size):
                seq_hits = hits[hits["sequence_name"] == seq_idx]
                
                if not seq_hits.empty:
                    seq_hits = seq_hits.sort_values(by="start")
                    
                    seq_positions = []
                    
                    num_CTCFs_0.append(len(seq_hits))
                    sum_FIMO_0.append(seq_hits["score"].sum())
                    max_FIMO_0.append(seq_hits["score"].max())
                    
                    for start, end in zip(list(seq_hits["start"]), list(seq_hits["end"])):
                        seq_positions.append((start, end))

                    positions_0.append(seq_positions)
                    
                    # Concatenate strand symbols into a single string
                    strand_concat = ''.join(seq_hits["strand"].tolist())
                    strand_strings_0.append(strand_concat)
                    
                else:
                    num_CTCFs_0.append(0)
                    sum_FIMO_0.append(0.0)
                    max_FIMO_0.append(0.0)
                    positions_0.append(tuple())
                    strand_strings_0.append("no")
    
    with torch.no_grad():
        for orig_batch, edited_batch in zip(orig_loader, edited_loader):
            
            orig_batch = orig_batch.to(device)
            edited_batch = edited_batch.to(device)
            
            orig_slice = orig_batch[:, :, edit_start1-extra_flank:edit_end1+extra_flank]
            edited_slice = edited_batch[:, :, edit_start1-extra_flank:edit_end1+extra_flank]
            
            orig_slice_np = orig_slice.cpu().numpy()
            edited_slice_np = edited_slice.cpu().numpy()
            
            orig_hits = fimo(
                motifs=motifs,
                sequences=orig_slice_np,
                threshold=1e-4,
                reverse_complement=True
            )[0]
            
            orig_hits["start"] -= extra_flank
            orig_hits["end"] -= extra_flank
            
            for seq_idx in range(batch_size):
                seq_hits = orig_hits[orig_hits["sequence_name"] == seq_idx]
                
                if not seq_hits.empty:
                    orig_num_CTCFs_1.append(len(seq_hits))
                else:
                    orig_num_CTCFs_1.append(0)

            hits = fimo(
                motifs=motifs,
                sequences=edited_slice_np,
                threshold=1e-4,
                reverse_complement=True
            )[0]

            hits["start"] -= extra_flank
            hits["end"] -= extra_flank
            
            for seq_idx in range(batch_size):
                seq_hits = seq_hits.sort_values(by="start")
                
                seq_hits = hits[hits["sequence_name"] == seq_idx]
                
                if not seq_hits.empty:
                    seq_positions = []
                    
                    num_CTCFs_1.append(len(seq_hits))
                    sum_FIMO_1.append(seq_hits["score"].sum())
                    max_FIMO_1.append(seq_hits["score"].max())
                    
                    for start, end in zip(list(seq_hits["start"]), list(seq_hits["end"])):
                        seq_positions.append((start, end))

                    positions_1.append(seq_positions)
                
                    # Concatenate strand symbols into a single string
                    strand_concat = ''.join(seq_hits["strand"].tolist())
                    strand_strings_1.append(strand_concat)
                    
                else:
                    num_CTCFs_1.append(0)
                    sum_FIMO_1.append(0.0)
                    max_FIMO_1.append(0.0)
                    positions_1.append(tuple())
                    strand_strings_1.append("no")
            
            
    # -----------------------------
    # Final merge + Save
    # -----------------------------
    df["SCD"] = scd_values

    # dot-7
    df["dot7_result"] = dot7_mean_values 
    df["dot7_target"] = target_dot7_mean_values
    df["dot7_init"] = og_dot7_mean_values

    # dot-11
    df["dot11_result"] = dot11_mean_values 
    df["dot11_target"] = target_dot11_mean_values
    df["dot11_init"] = og_dot11_mean_values

    # dot-15
    df["dot15_result"] = dot15_mean_values 
    df["dot15_target"] = target_dot15_mean_values
    df["dot15_init"] = og_dot15_mean_values

    df["num_edits_slice0"] = edit_counts0
    df["num_edits_slice1"] = edit_counts1
    
    df["GC_seq"] = seq_GC_content
    df["GC_slice0"] = slice0_GC_content
    df["GC_slice1"] = slice1_GC_content
    df["GC_slice0_edited"] = edited0_GC_content
    df["GC_slice1_edited"] = edited1_GC_content

    df["init_CTCFs_num_slice0"] = orig_num_CTCFs_0[:len(df)]
    df["CTCFs_num_slice0"] = num_CTCFs_0[:len(df)]
    df["FIMO_sum_slice0"] = sum_FIMO_0[:len(df)]
    df["FIMO_max_slice0"] = max_FIMO_0[:len(df)]
    df["orientation_slice0"] = strand_strings_0[:len(df)]
    df["positions_slice0"] = positions_0[:len(df)]

    df["init_CTCFs_num_slice1"] = orig_num_CTCFs_1[:len(df)]
    df["CTCFs_num_slice1"] = num_CTCFs_1[:len(df)]
    df["FIMO_sum_slice1"] = sum_FIMO_1[:len(df)]
    df["FIMO_max_slice1"] = max_FIMO_1[:len(df)]
    df["orientation_slice1"] = strand_strings_1[:len(df)]
    df["positions_slice1"] = positions_1[:len(df)]
    
    out_path = f"/scratch1/smaruj/generate_genomic_dot/results/dist_{DIST}bins/fold{FOLD}_{TARGET_C}_genomic_windows_table_results.tsv"
    df.to_csv(out_path, sep="\t", index=False)
    print(f"Saved results → {out_path}")

# -----------------------------
# Entry point
# -----------------------------
if __name__ == "__main__":
    main()