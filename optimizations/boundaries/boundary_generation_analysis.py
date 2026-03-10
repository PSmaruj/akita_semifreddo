#!/usr/bin/env python3
import argparse
import os
import sys
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

sys.path.append(os.path.abspath("/home1/smaruj/pytorch_akita/"))
sys.path.insert(0, os.path.abspath("/home1/smaruj/ledidi_akita/"))

from akita.model import SeqNN

from utils.dataset_utils import OriginalDataset, TriuMatrixDataset
from utils.data_utils import from_upper_triu_batch
from utils.fimo_utils import read_meme_pwm_as_numpy

# this class should be moved to dataset_utils as well
class CentralInsertionDataset(Dataset):
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
    

# ---------------------------------------------------------------------------
# This function could be used for insulation estimation
# ---------------------------------------------------------------------------

# insulation -> Upper-right quarter slice of the 512-bin contact map.
# Rows 0–249  → upstream half  (excluding diagonal buffer)
# Cols 260–511 → downstream half (excluding diagonal buffer)
URQ_ROW_SLICE = slice(0, 250)
URQ_COL_SLICE = slice(260, 512)

def predict_insulation(
    loader: DataLoader,
    model: torch.nn.Module,
    device: torch.device,
) -> list[float]:
    """
    Run inference over all batches and return per-sequence URQ mean values.

    The URQ (upper-right quarter) of the contact map captures the insulation
    signal across the boundary: high values indicate a strong boundary.
    """
    urq_means: list[float] = []

    with torch.no_grad():
        for batch in loader:
            preds = model(batch.to(device)).cpu()
            maps  = from_upper_triu_batch(preds)                          # (B, 512, 512)
            urq   = maps[:, URQ_ROW_SLICE, URQ_COL_SLICE]                 # (B, 250, 252)
            urq_means.extend(np.nanmean(urq, axis=(1, 2)).tolist())

    return urq_means

from memelite import fimo

def run_fimo(seq_tensor, motifs_dict, threshold=1e-4):
    """Run FIMO on a (1, 4, L) tensor; returns the hits DataFrame."""
    arr = seq_tensor.cpu().detach().numpy()
    return fimo(motifs=motifs_dict, sequences=arr,
                threshold=threshold, reverse_complement=True)[0]

# -----------------------------
# Main pipeline
# -----------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--boundary_strength", type=float, required=True)
    parser.add_argument("--l", type=float, required=True)
    parser.add_argument("--e", type=float, required=True)
    parser.add_argument("--t", type=float, required=True)
    args = parser.parse_args()

    FOLD = args.fold
    TARGET_C = args.boundary_strength
    L = args.l # default 0.01
    epsilon = args.e # default 1e-4
    tau = args.t # default 1.0
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # -----------------------------
    # Table with number of edits and the last step with edits
    # -----------------------------
    df_path = f"/project2/fudenber_735/smaruj/sequence_design/ledidi_semifreddo_akita/optimizations/boundaries/lambda/lambda_{L}/fold{FOLD}_selected_genomic_windows_centered_chrom_states_opt.tsv"
    df = pd.read_csv(df_path, sep="\t")

    # -----------------------------
    # Load model
    # -----------------------------
    model = SeqNN()
    MODEL_CKPT = (
    "/home1/smaruj/pytorch_akita/models/finetuned/mouse/Hsieh2019_mESC"
    "/checkpoints/Akita_v2_mouse_Hsieh2019_mESC_model0_finetuned.pth"
    )
    model.load_state_dict(torch.load(MODEL_CKPT, map_location=device))
    model.eval()

    # -----------------------------
    # Datasets + Loaders
    # -----------------------------
    
    _PROJ = "/project2/fudenber_735/smaruj/sequence_design/ledidi_semifreddo_akita"
    DEFAULT_SEQ_BASE_DIR      = f"{_PROJ}/analysis/flat_regions"
    
    DEFAULT_TARGET_BASE_DIR   = f"{_PROJ}/optimizations/boundaries/targets"
    
    orig_dataset = OriginalDataset(df, f"{DEFAULT_SEQ_BASE_DIR}/mouse_sequences/fold{FOLD}/")
    edited_dataset = CentralInsertionDataset(
        df,
        f"{DEFAULT_SEQ_BASE_DIR}/mouse_sequences/fold{FOLD}/",
        f"/project2/fudenber_735/smaruj/sequence_design/ledidi_semifreddo_akita/optimizations/boundaries/lambda/lambda_{L}/fold{FOLD}/"
    )
    target_dataset = TriuMatrixDataset(
        df,
        f"{DEFAULT_TARGET_BASE_DIR}/boundary_neg0p5/fold{FOLD}/"
    )
    # where boundary_neg0p5 is the tagged boundary_strength
    
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

    urq_mean_values = []
    og_urq_mean_values = []
    target_urq_mean_values = []

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

            preds_orig = model(orig_batch).cpu()
            preds_edited = model(edited_batch).cpu()

            preds_all_orig.extend(preds_orig)
            preds_all_edited.extend(preds_edited)
            targets_all.extend(target_batch.cpu())
            
            # insulation score
            orig_maps = from_upper_triu_batch(preds_orig)
            edited_maps = from_upper_triu_batch(preds_edited)
            target_maps = from_upper_triu_batch(target_batch.cpu())

            urq_mean_values.extend(np.nanmean(edited_maps[:, 0:250, 260:512], axis=(1, 2)))
            og_urq_mean_values.extend(np.nanmean(orig_maps[:, 0:250, 260:512], axis=(1, 2)))
            target_urq_mean_values.extend(np.nanmean(target_maps[:, 0:250, 260:512], axis=(1, 2)))

    
    # -----------------------------
    # CTCF scoring
    # -----------------------------
    pwm_path = "/home1/smaruj/ledidi_akita/data/pwm/MA0139.1.meme"
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
    df["URQ_result"] = urq_mean_values
    df["URQ_target"] = target_urq_mean_values
    df["URQ_init"] = og_urq_mean_values
    df["GC_seq"] = seq_GC_content
    df["GC_slice"] = slice_GC_content
    df["GC_slice_edited"] = edited_GC_content
    df["init_CTCFs_num"] = orig_num_CTCFs[:len(df)]
    df["CTCFs_num"] = num_CTCFs[:len(df)]
    df["FIMO_sum"] = sum_FIMO[:len(df)]
    df["FIMO_max"] = max_FIMO[:len(df)]
    df["orientation"] = strand_strings[:len(df)]
    df["positions"] = positions[:len(df)]

    out_path = f"/project2/fudenber_735/smaruj/sequence_design/ledidi_semifreddo_akita/optimizations/boundaries/lambda/lambda_{L}/fold{FOLD}_selected_genomic_windows_centered_chrom_states_results.tsv"
    df.to_csv(out_path, sep="\t", index=False)
    print(f"Saved results → {out_path}")

# -----------------------------
# Entry point
# -----------------------------
if __name__ == "__main__":
    main()