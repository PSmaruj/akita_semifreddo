import torch
import pandas as pd
import argparse
import sys
import os
import ast

sys.path.append(os.path.abspath("/home1/smaruj/pytorch_akita/"))
from akita_model.model import SeqNN

sys.path.insert(0, "/home1/smaruj/ledidi")
from ledidi import Ledidi


def parse_args():
    parser = argparse.ArgumentParser(description="Run Ledidi on selected genomic loci.")
    parser.add_argument("--fold", type=int, required=True, help="Fold number to process")
    parser.add_argument("--target", type=str, required=True, help="Desired target constant value")
    parser.add_argument("--gamma", type=str, required=True, help="Desired gamma value")
    parser.add_argument("--model_path", type=str, required=True, help="Path to model .pt file")
    parser.add_argument("--input_tsv_dir", type=str, required=True, help="Base path to input .tsv files")
    parser.add_argument("--pt_files_dir", type=str, required=True, help="Base path to input and output .pt files")
    parser.add_argument("--boundary_mask_path", type=str, required=True, help="Path to boundary mask .pt file")
    
    parser.add_argument("--bin_size", type=int, default=2048, help="Bin size for model input")
    parser.add_argument("--cropping_applied", type=int, default=64, help="Cropping applied in the model")
    parser.add_argument("--padding_bins", type=int, default=2, help="Number of bins to pad input slices")
    parser.add_argument("--max_iter", type=int, default=2000, help="Maximum number of optimization steps")
    parser.add_argument("--early_stopping_iter", type=int, default=2000, help="Early stopping threshold")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    
    return parser.parse_args()


def build_ctcf_input_mask(seq_len, ctcf_locs, flanking=15):
    """
    Build a boolean mask tensor indicating CTCF regions + optional flanking bases.

    Args:
        seq_len (int): total length of the sequence (number of positions).
        ctcf_locs (list of tuples): list of (start, end) positions of CTCF sites (0-based, end exclusive).
        flanking (int): number of bases to also mask on each side of the site (default 15).

    Returns:
        torch.BoolTensor: mask of shape (seq_len,) where True means "masked / not editable".
    """
    mask = torch.zeros(seq_len, dtype=torch.bool)
    
    for start, end in ctcf_locs:
        # Apply flanking region, clamp to sequence boundaries
        start_flank = max(0, start - flanking)
        end_flank = min(seq_len, end + flanking)
        
        mask[start_flank:end_flank] = True
    
    return mask


def main():
    args = parse_args()
    target_c = float(args.target)
    gamma = float(args.gamma)
    
    # Set seed
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    model = SeqNN()
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()

    FOLD = args.fold
    
    df = pd.read_csv(f"{args.input_tsv_dir}/fold{FOLD}_with_positions.tsv", sep="\t") 
                     
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
        ctcf_locations = ast.literal_eval(row.ctcf_motif_locs)
        
        print(f"CTCT-restricted boundary suppression for genome location: {chrom}:{pred_start}-{pred_end}")
        
        input_mask = build_ctcf_input_mask(2048, ctcf_locations, 15).to(device=device)
        
        X = torch.load(f"{args.pt_files_dir}/ohe_X/fold{FOLD}/{chrom}_{pred_start}_{pred_end}_X.pt", weights_only=True, map_location=device)
        target = torch.load(f"{args.pt_files_dir}/targets/target_{target_c}/fold{FOLD}/{chrom}_{pred_start}_{pred_end}_target.pt", weights_only=True, map_location=device)
        tower_output_path = f"{args.pt_files_dir}/tower_outputs/fold{FOLD}/{chrom}_{pred_start}_{pred_end}_tower_out.pt"
        
        CTCF_PWM = "/home1/smaruj/IterativeMutagenesis/MA0139.1.meme"
        
        wrapper = Ledidi(model, 
                    input_loss=torch.nn.L1Loss(reduction='sum'), 
                    output_loss=torch.nn.L1Loss(reduction='sum'),
                    batch_size=1,
                    l=0.1, # lowered for restricted boundary suppression
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
                    g=gamma,
                    punish_ctcf=True,
                    ctcf_meme_path=CTCF_PWM,
                    suppressing_mask=input_mask
                    ).cuda()
        
        slice_0_torch = X[:, :, slice_0_start:slice_0_end]
        
        # x_bar_slice_0, last_update = wrapper.fit_transform(X=slice_0_torch, y_bar=target)
        x_bar_slice_0, last_update, _, _, _ = wrapper.fit_transform(X=slice_0_torch, y_bar=target)
        
        # Update df with last_accepted_step
        df.at[i, "last_accepted_step"] = last_update
        
        torch.save(x_bar_slice_0[:,:,padding:-padding], f"{args.pt_files_dir}/ctcf_restricted_results/gamma_{gamma}/fold{FOLD}/{chrom}_{pred_start}_{pred_end}_slice.pt")
        
    df.to_csv(f"{args.pt_files_dir}/ctcf_restricted_results/gamma_{gamma}/fold{FOLD}_with_positions_steps.tsv", sep="\t", index=False)
    
if __name__ == "__main__":
    main()