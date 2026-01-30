#!/usr/bin/env python3
import os
import sys
import argparse
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from memelite import fimo

sys.path.append(os.path.abspath("/home1/smaruj/pytorch_akita/"))

from akita_model.model import SeqNN


# ==========================
# Dataset Classes
# ==========================

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

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        X = torch.load(f"{self.init_seq_path}{chrom}_{start}_{end}_X.pt", weights_only=True, map_location=device)
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
        
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        X = torch.load(f"{self.init_seq_path}{chrom}_{start}_{end}_X.pt", weights_only=True, map_location=device)
        slice = torch.load(f"{self.slice_path}{chrom}_{start}_{end}_slice.pt", weights_only=True, map_location=device)
        
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

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        # Load the flat upper-triangular vector
        triu_tensor = torch.load(file_path, map_location=device)

        # triu_tensor = triu_tensor.squeeze()

        return triu_tensor

# ==========================
# Helper Functions
# ==========================

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

# ==========================
# Main Pipeline
# ==========================

def main(args):
    FOLD = args.fold
    TARGET_C = args.target_c
    
    print(f"\n=== Running analysis for fold {FOLD}, target C = {TARGET_C} ===") 

    df = pd.read_csv(f"/scratch1/smaruj/suppressing_CTCFs/results_control_repeated/fold{FOLD}_with_positions_steps.tsv", sep="\t")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = SeqNN()
    model_path = (
        "/scratch1/smaruj/Akita_pytorch_models/finetuned/mouse_models/"
        "Hsieh2019_mESC/models/Akita_v2_mouse_Hsieh2019_mESC_model0_finetuned.pth"
    )
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Datasets
    orig_dataset = OriginalDataset(df, f"/scratch1/smaruj/suppressing_CTCFs/ohe_X/fold{FOLD}/")
    orig_loader = DataLoader(orig_dataset, batch_size=4, shuffle=False)

    edited_dataset = BoundaryGenerationDataset(df, 
                                    f"/scratch1/smaruj/suppressing_CTCFs/ohe_X/fold{FOLD}/", 
                                    f"/scratch1/smaruj/suppressing_CTCFs/results_control_repeated/fold{FOLD}/")

    edited_loader = DataLoader(edited_dataset, batch_size=4, shuffle=False)

    target_dataset = TriuMatrixDataset(df, f"/scratch1/smaruj/suppressing_CTCFs/targets/target_{TARGET_C}/fold{FOLD}/")
    target_loader = DataLoader(target_dataset, batch_size=4, shuffle=False)

    # Constants
    slice, cropping, bin_size = 256, 64, 2048
    edit_start = (slice + cropping) * bin_size
    edit_end = edit_start + bin_size

    # preds_all_orig = []
    # preds_all_edited = []
    # targets_all = []
    
    scd_values = []
    urq_mean_values = []
    og_urq_mean_values = []
    target_urq_mean_values = []
    edit_counts = []
    seq_GC_content = []
    slice_GC_content = []
    edited_GC_content = []

    print("Running model predictions...")
    with torch.no_grad():
        for orig_batch, edited_batch, target_batch in zip(orig_loader, edited_loader, target_loader):
            orig_batch = orig_batch.to(device)
            edited_batch = edited_batch.to(device)
            target_batch = target_batch.to(device)
            target_batch = target_batch.squeeze(1)
            
            # Compute GC content
            gc_content_all = orig_batch[:, 1:3, :].sum(dim=1) / orig_batch.sum(dim=1)  # [B, L]
            gc_content_slice = orig_batch[:, 1:3, edit_start:edit_end].sum(dim=1) / orig_batch[:, :, edit_start:edit_end].sum(dim=1)  # [B, edited_L]
            gc_content_slice_edit = edited_batch[:, 1:3, edit_start:edit_end].sum(dim=1) / edited_batch[:, :, edit_start:edit_end].sum(dim=1)  # [B, edited_L]
            
            # Mean GC content per sequence
            mean_gc_all = gc_content_all.mean(dim=1).cpu().numpy()  # [B]
            mean_gc_slice = gc_content_slice.mean(dim=1).cpu().numpy()  # [B]
            mean_gc_edit = gc_content_slice_edit.mean(dim=1).cpu().numpy()  # [B]
            
            seq_GC_content.extend(mean_gc_all)
            slice_GC_content.extend(mean_gc_slice)
            edited_GC_content.extend(mean_gc_edit)
            
            diffs = torch.abs(orig_batch[:, :, edit_start:edit_end] - edited_batch[:, :, edit_start:edit_end])  # shape [B, 4, region_len]
            num_flips = diffs.sum(dim=(1, 2))  # total bit flips per sequence
            num_edits = (num_flips / 2).cpu().numpy()  # divide by 2 to get base edits
            
            edit_counts.extend(num_edits)
        
            preds_orig = model(orig_batch).cpu()
            preds_edited = model(edited_batch).cpu()

            # preds_all_orig.extend(preds_orig)
            # preds_all_edited.extend(preds_edited)
            # targets_all.extend(target_batch)
            
            scd_batch = torch.sqrt(((preds_edited - preds_orig) ** 2).sum(dim=(1, 2)))  # [B]
            scd_values.extend(scd_batch.numpy())
            
            orig_maps = from_upper_triu_batch(preds_orig)
            edited_maps = from_upper_triu_batch(preds_edited)
            
            urq_mean = np.nanmean(edited_maps[:, 0:250, 260:512], axis=(1, 2))
            urq_mean_values.extend(urq_mean)
            
            og_urq_mean = np.nanmean(orig_maps[:, 0:250, 260:512], axis=(1, 2))
            og_urq_mean_values.extend(og_urq_mean)
            
            target_maps = from_upper_triu_batch(target_batch)
            target_urq_mean = np.nanmean(target_maps[:, 0:250, 260:512], axis=(1, 2))
            target_urq_mean_values.extend(target_urq_mean)
    
    # scanning for CTCF
     
    CTCF_PWM = "/home1/smaruj/IterativeMutagenesis/MA0139.1.meme"

    pwm_CTCF = read_meme_pwm_as_numpy(CTCF_PWM)
    pwm_CTCF_tensor = torch.from_numpy(pwm_CTCF.T).float()
    motifs_dict = {"CTCF": pwm_CTCF_tensor}

    pwm_ctcf = np.load("./PWM_with_flanks.npy")

    left_flank_slice = pwm_ctcf[:15, :]  
    left_flank_transposed = left_flank_slice.T 
    left_flank_tensor = torch.tensor(left_flank_transposed, dtype=torch.float32)

    right_flank_slice = pwm_ctcf[-15:, :]  
    right_flank_transposed = right_flank_slice.T 
    right_flank_tensor = torch.tensor(right_flank_transposed, dtype=torch.float32)

    motifs_dict = {"CTCF": pwm_CTCF_tensor,
                "left_flank": left_flank_tensor,
                "right_flank": right_flank_tensor}

    batch_size = 4

    orig_num_CTCFs = []
    num_CTCFs = []

    orig_CTCFs_coord = []
    new_CTCFs_coord = []

    avg_orig_fimo_scores = []
    avg_new_fimo_scores = []

    avg_orig_left_fimo_scores = []
    avg_new_left_fimo_scores = []

    avg_orig_right_fimo_scores = []
    avg_new_right_fimo_scores = []

    extra_flank = 60

    with torch.no_grad():
        for orig_batch, edited_batch in zip(orig_loader, edited_loader):
            
            orig_batch = orig_batch.to(device)
            edited_batch = edited_batch.to(device)
            
            orig_slice = orig_batch[:, :, edit_start-extra_flank:edit_end+extra_flank]
            edited_slice = edited_batch[:, :, edit_start-extra_flank:edit_end+extra_flank]
            
            # original sequences
            ctcf_orig_hits, left_orig_hits, right_orig_hits = fimo(
                motifs=motifs_dict,
                sequences=orig_slice,
                threshold=1e-4,
                reverse_complement=True
            )
            
            # Shift coordinates back to original reference
            ctcf_orig_hits["start"] -= extra_flank
            ctcf_orig_hits["end"]   -= extra_flank
            left_orig_hits["start"] -= extra_flank
            left_orig_hits["end"]   -= extra_flank
            right_orig_hits["start"] -= extra_flank
            right_orig_hits["end"]   -= extra_flank
        
            # edited sequences
            ctcf_edit_hits, left_edit_hits, right_edit_hits = fimo(
                motifs=motifs_dict,
                sequences=edited_slice,
                threshold=1e-4,
                reverse_complement=True
            )

            # Shift coordinates back to original reference
            ctcf_edit_hits["start"] -= extra_flank
            ctcf_edit_hits["end"]   -= extra_flank
            left_edit_hits["start"] -= extra_flank
            left_edit_hits["end"]   -= extra_flank
            right_edit_hits["start"] -= extra_flank
            right_edit_hits["end"]   -= extra_flank
        
            for seq_idx in range(batch_size):
                # --- ORIGINAL, CTCF ---
                orig_seq_hits = ctcf_orig_hits[ctcf_orig_hits["sequence_name"] == seq_idx]
                if not orig_seq_hits.empty:
                    orig_seq_hits = orig_seq_hits.sort_values(by="start")
                    orig_num_CTCFs.append(len(orig_seq_hits))
                    orig_ctcf_coords = set(zip(orig_seq_hits["start"], orig_seq_hits["end"], orig_seq_hits["strand"]))
                    orig_fimo_score_avg = orig_seq_hits["score"].mean()
                else:
                    orig_num_CTCFs.append(0)
                    orig_ctcf_coords = set()
                    orig_fimo_score_avg = 0.0
                orig_CTCFs_coord.append(orig_ctcf_coords)
                avg_orig_fimo_scores.append(orig_fimo_score_avg)
            
                # --- ORIGINAL, LEFT FLANK ---
                orig_left_hits = left_orig_hits[left_orig_hits["sequence_name"] == seq_idx]
                if not orig_left_hits.empty:
                    orig_fimo_left_score_avg = orig_left_hits["score"].mean()
                else:
                    orig_fimo_left_score_avg = 0.0
                avg_orig_left_fimo_scores.append(orig_fimo_left_score_avg)
                    
                # --- ORIGINAL, RIGHT FLANK ---
                orig_right_hits = right_orig_hits[right_orig_hits["sequence_name"] == seq_idx]
                if not orig_right_hits.empty:
                    orig_fimo_right_score_avg = orig_right_hits["score"].mean()
                else:
                    orig_fimo_right_score_avg = 0.0
                avg_orig_right_fimo_scores.append(orig_fimo_right_score_avg)
                
                # --- EDITED, CTCF ---
                edited_ctcf_seq_hits = ctcf_edit_hits[ctcf_edit_hits["sequence_name"] == seq_idx]
                if not edited_ctcf_seq_hits.empty:
                    edited_ctcf_seq_hits = edited_ctcf_seq_hits.sort_values(by="start")
                    num_CTCFs.append(len(edited_ctcf_seq_hits))

                    # New CTCF sites only
                    new_hits = [
                        (start, end, strand, score)
                        for start, end, strand, score in zip(
                            edited_ctcf_seq_hits["start"],
                            edited_ctcf_seq_hits["end"],
                            edited_ctcf_seq_hits["strand"],
                            edited_ctcf_seq_hits["score"]
                        )
                        if (start, end, strand) not in orig_ctcf_coords
                    ]
                    
                    df_new_hits = pd.DataFrame(new_hits, columns=["start", "end", "strand", "score"])
                
                    if not df_new_hits.empty:
                        df_new_hits = df_new_hits.sort_values(by="start")
                        new_ctcf_coords = set(zip(df_new_hits["start"], df_new_hits["end"], df_new_hits["strand"]))
                        new_fimo_scores = df_new_hits["score"].mean()
                    else:
                        new_ctcf_coords = set()
                        new_fimo_scores = 0.0
                    new_CTCFs_coord.append(new_ctcf_coords)
                    avg_new_fimo_scores.append(new_fimo_scores)
            
                else:
                    num_CTCFs.append(0)
                    new_CTCFs_coord.append(set())
                    avg_new_fimo_scores.append(0.0)   
            
                # --- ORIGINAL, LEFT FLANK ---
                edit_left_hits = left_edit_hits[left_edit_hits["sequence_name"] == seq_idx]
                if not edit_left_hits.empty:
                    edit_fimo_left_score_avg = edit_left_hits["score"].mean()
                else:
                    edit_fimo_left_score_avg = 0.0
                avg_new_left_fimo_scores.append(edit_fimo_left_score_avg)
                    
                # --- ORIGINAL, RIGHT FLANK ---
                edit_right_hits = right_edit_hits[right_edit_hits["sequence_name"] == seq_idx]
                if not edit_right_hits.empty:
                    edit_fimo_right_score_avg = edit_right_hits["score"].mean()
                else:
                    edit_fimo_right_score_avg = 0.0
                avg_new_right_fimo_scores.append(edit_fimo_right_score_avg)        
    
    # useful for plotting only                          
    # orig_preds_all = torch.cat(preds_all_orig, dim=0)
    # edited_preds_all = torch.cat(preds_all_edited, dim=0)
    # targets_all = torch.cat(targets_all, dim=0)

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
    df["avg_orig_fimo_scores"] = avg_orig_fimo_scores[:len(df)]
    df["avg_new_fimo_scores"] = avg_new_fimo_scores[:len(df)]

    df["orig_CTCFs_coord"] = orig_CTCFs_coord[:len(df)]
    df["new_CTCFs_coord"] = new_CTCFs_coord[:len(df)]

    df.to_csv(f"/scratch1/smaruj/suppressing_CTCFs/results_control_repeated/fold{FOLD}_with_positions_steps_results.tsv", sep="\t", index=False)
    
# ==========================
# Entry Point
# ==========================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze model predictions for CTCF boundary edits")
    parser.add_argument("--fold", type=int, required=True, help="Fold number (e.g. 0)")
    parser.add_argument("--target_c", type=float, required=True, help="Target C value (e.g. -0.5)")
    args = parser.parse_args()
    main(args)