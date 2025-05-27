import torch
import pandas as pd
import argparse
import sys
import os

sys.path.append(os.path.abspath("/home1/smaruj/pytorch_akita/"))
from model_v2_compatible import SeqNN

sys.path.insert(0, "/home1/smaruj/ledidi")
from ledidi import Ledidi


def parse_args():
    parser = argparse.ArgumentParser(description="Run Ledidi on selected genomic loci.")
    parser.add_argument("--fold", type=int, required=True, help="Fold number to process")
    parser.add_argument("--model_path", type=str, required=True, help="Path to model .pt file")
    parser.add_argument("--input_dir", type=str, required=True, help="Base path to input .pt and .tsv files")
    parser.add_argument("--output_dir", type=str, required=True, help="Base path to write output")
    parser.add_argument("--boundary_mask_path", type=str, required=True, help="Path to boundary mask .pt file")
    
    parser.add_argument("--bin_size", type=int, default=2048, help="Bin size for model input")
    parser.add_argument("--cropping_applied", type=int, default=64, help="Cropping applied in the model")
    parser.add_argument("--padding_bins", type=int, default=2, help="Number of bins to pad input slices")
    parser.add_argument("--max_iter", type=int, default=4000, help="Maximum number of optimization steps")
    parser.add_argument("--early_stopping_iter", type=int, default=100, help="Early stopping threshold")
    
    return parser.parse_args()


def main():
    args = parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    model = SeqNN()
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()

    FOLD = args.fold
    
    df = pd.read_csv(f"{args.input_dir}/fold{FOLD}_selected_genomic_windows_centered.tsv", sep="\t")
    
    boundary_mask_path = args.boundary_mask_path
    
    bin_size = args.bin_size
    cropping_applied = args.cropping_applied
    padding_bins = args.padding_bins
    padding = padding_bins * bin_size

    slice_0_bins = [256]
    slice_0_start = (min(slice_0_bins) + cropping_applied - padding_bins) * bin_size
    slice_0_end = (max(slice_0_bins) + 1 + cropping_applied + padding_bins) * bin_size
    
    df["last_accepted_step"] = -1  # initialize with placeholder
    
    for i, row in enumerate(df.itertuples(index=False)):
        chrom, pred_start, pred_end = row.chrom, row.centered_start, row.centered_end
        
        print(f"Boundary generation for genome location: {chrom}:{pred_start}-{pred_end}")
        
        X = torch.load(f"{args.input_dir}/ohe_X/fold{FOLD}/{chrom}_{pred_start}_{pred_end}_X.pt", weights_only=True, map_location=device)
        target = torch.load(f"{args.input_dir}/targets/fold{FOLD}/{chrom}_{pred_start}_{pred_end}_target.pt", weights_only=True, map_location=device)
        tower_output_path = f"{args.input_dir}/tower_outputs/fold{FOLD}/{chrom}_{pred_start}_{pred_end}_tower_out.pt"
        
        wrapper = Ledidi(model, 
                    input_loss=torch.nn.L1Loss(reduction='sum'), 
                    output_loss=torch.nn.L1Loss(reduction='sum'),
                    batch_size=1,
                    max_iter=args.max_iter,
                    early_stopping_iter=args.early_stopping_iter,
                    return_history=False,
                    verbose=True,
                    bin_size=args.bin_size,
                    input_mask_slices_0=[256], # mid-bin
                    cropping_applied=args.cropping_applied,
                    output_mask_path=boundary_mask_path,
                    use_semifreddo=True,
                    semifreddo_temp_output_path=tower_output_path,
                    punish_ctcf=False,
                    ctcf_meme_path=None
                    ).cuda()
        
        slice_0_torch = X[:, :, slice_0_start:slice_0_end]
        
        x_bar_slice_0, last_update = wrapper.fit_transform(X=slice_0_torch, y_bar=target)
        
        # Update df with last_accepted_step
        df.at[i, "last_accepted_step"] = last_update
        
        torch.save(x_bar_slice_0[:,:,padding:-padding], f"{args.output_dir}/results/fold{FOLD}/{chrom}_{pred_start}_{pred_end}_slice.pt")

    df.to_csv(f"{args.output_dir}/fold{FOLD}_selected_genomic_windows_centered_with_steps.tsv", sep="\t", index=False)
    
    
if __name__ == "__main__":
    main()