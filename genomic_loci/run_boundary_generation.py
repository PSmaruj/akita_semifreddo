import torch
import pandas as pd

import sys
import os
sys.path.append(os.path.abspath("/home1/smaruj/pytorch_akita/"))

from model_v2_compatible import SeqNN

sys.path.insert(0, "/home1/smaruj/ledidi")
from ledidi import Ledidi

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    model = SeqNN()
    model.load_state_dict(torch.load("/home1/smaruj/pytorch_akita/model_0_v2_finetuned_correctly.pt", map_location=device))
    model.eval()

    df = pd.read_csv("/scratch1/smaruj/genomic_insertion_loci/fold0_selected_genomic_windows_centered.tsv", sep="\t")
    df = df.drop_duplicates(subset=["chrom", "centered_start", "centered_end"])
    
    boundary_mask_path = "/scratch1/smaruj/genomic_insertion_loci/boundary_indices.pt"
    
    bin_size = 2048
    cropping_applied = 64
    padding_bins = 2
    padding = padding_bins * bin_size

    slice_0_bins = [256]
    slice_0_start = (min(slice_0_bins) + cropping_applied - padding_bins) * bin_size
    slice_0_end = (max(slice_0_bins) + 1 + cropping_applied + padding_bins) * bin_size
    
    for row in df.itertuples(index=False):
        chrom, pred_start, pred_end = row.chrom, row.centered_start, row.centered_end
        
        print(f"Boundary generation for genome location: {chrom}:{pred_start}-{pred_end}")
        
        X = torch.load(f"/scratch1/smaruj/genomic_insertion_loci/ohe_X/{chrom}_{pred_start}_{pred_end}_X.pt", weights_only=True, map_location=device)
        target = torch.load(f"/scratch1/smaruj/genomic_insertion_loci/targets/{chrom}_{pred_start}_{pred_end}_target.pt", weights_only=True, map_location=device)
        tower_output_path = f"/scratch1/smaruj/genomic_insertion_loci/tower_outputs/{chrom}_{pred_start}_{pred_end}_tower_out.pt"
        
        wrapper = Ledidi(model, 
                    input_loss=torch.nn.L1Loss(reduction='sum'), 
                    output_loss=torch.nn.L1Loss(reduction='sum'),
                    batch_size=1,
                    max_iter=3000,
                    early_stopping_iter=2000,
                    return_history=False,
                    verbose=True,
                    bin_size=2048,
                    input_mask_slices_0=[256], # mid-bin
                    cropping_applied=64,
                    output_mask_path=boundary_mask_path,
                    use_semifreddo=True,
                    semifreddo_temp_output_path=tower_output_path,
                    punish_ctcf=False,
                    ctcf_meme_path=None
                    ).cuda()
        
        slice_0_torch = X[:, :, slice_0_start:slice_0_end]
        
        x_bar_slice_0 = wrapper.fit_transform(X=slice_0_torch, y_bar=target)
        
        torch.save(x_bar_slice_0[:,:,padding:-padding], f"/scratch1/smaruj/genomic_insertion_loci/results/{chrom}_{pred_start}_{pred_end}_slice.pt")

if __name__ == "__main__":
    main()