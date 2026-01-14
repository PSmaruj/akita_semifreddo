import torch
import pandas as pd
import argparse
import sys
import os

sys.path.append(os.path.abspath("/home1/smaruj/pytorch_akita/"))
from akita_model.model import SeqNN

sys.path.insert(0, "/home1/smaruj/ledidi/ledidi")
from ledidi_multiple_models_sum_mod import Ledidi


def parse_args():
    parser = argparse.ArgumentParser(description="Run Ledidi on selected genomic loci.")
    parser.add_argument("--fold", type=int, required=True, help="Fold number to process")
    parser.add_argument("--target", type=str, required=True, help="Desired target constant value")
    parser.add_argument("--input_tsv_dir", type=str, required=True, help="Base path to input .tsv files")
    parser.add_argument("--pt_files_dir", type=str, required=True, help="Base path to input and output .pt files")
    parser.add_argument("--boundary_mask_path", type=str, required=True, help="Path to boundary mask .pt file")
    
    parser.add_argument("--bin_size", type=int, default=2048, help="Bin size for model input")
    parser.add_argument("--cropping_applied", type=int, default=64, help="Cropping applied in the model")
    parser.add_argument("--padding_bins", type=int, default=2, help="Number of bins to pad input slices")
    parser.add_argument("--max_iter", type=int, default=2000, help="Maximum number of optimization steps")
    parser.add_argument("--early_stopping_iter", type=int, default=2000, help="Early stopping threshold")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--l", type=float, default=0.1, help="Lambda to balance input/output losses")
        
    return parser.parse_args()


def main():
    args = parse_args()
    target_c = float(args.target)
    
    # Set seed
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    model0 = SeqNN()
    model0.load_state_dict(torch.load("/scratch1/smaruj/Akita_pytorch_models/finetuned/human_models/Rao2014_GM12878/models/Akita_v2_human_Rao2014_GM12878_model0_finetuned.pth", map_location=device))
    model0.eval()
    
    model1 = SeqNN()
    model1.load_state_dict(torch.load("/scratch1/smaruj/Akita_pytorch_models/finetuned/human_models/Rao2014_GM12878/models/Akita_v2_human_Rao2014_GM12878_model1_finetuned.pth", map_location=device))
    model1.eval()
    
    model2 = SeqNN()
    model2.load_state_dict(torch.load("/scratch1/smaruj/Akita_pytorch_models/finetuned/human_models/Rao2017_HCT116/models/Akita_v2_human_Rao2017_HCT116_model0_finetuned.pth", map_location=device))
    model2.eval()

    model3 = SeqNN()
    model3.load_state_dict(torch.load("/scratch1/smaruj/Akita_pytorch_models/finetuned/human_models/Rao2017_HCT116/models/Akita_v2_human_Rao2017_HCT116_model1_finetuned.pth", map_location=device))
    model3.eval()

    models = [model0, model1, model2, model3]
    
    FOLD = args.fold
    
    df = pd.read_csv(f"{args.input_tsv_dir}/fold{FOLD}_HUMAN_selected_genomic_windows_centered.tsv", sep="\t")
    
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
        
        X = torch.load(f"/scratch1/smaruj/generate_cell_type_specific_features/ohe_X_HUMAN/fold{FOLD}/{chrom}_{pred_start}_{pred_end}_X.pt", weights_only=True, map_location=device)

        target_m0 = torch.load(f"/scratch1/smaruj/generate_cell_type_specific_features/target_GM12878_weak_boundary/model0/{chrom}_{pred_start}_{pred_end}_target.pt", weights_only=True, map_location=device)
        target_m1 = torch.load(f"/scratch1/smaruj/generate_cell_type_specific_features/target_GM12878_weak_boundary/model1/{chrom}_{pred_start}_{pred_end}_target.pt", weights_only=True, map_location=device)
        target_m2 = torch.load(f"/scratch1/smaruj/generate_cell_type_specific_features/target_HCT116_strong_boundary/model0/{chrom}_{pred_start}_{pred_end}_target.pt", weights_only=True, map_location=device)
        target_m3 = torch.load(f"/scratch1/smaruj/generate_cell_type_specific_features/target_HCT116_strong_boundary/model1/{chrom}_{pred_start}_{pred_end}_target.pt", weights_only=True, map_location=device)
    
        y_bar_list = [target_m0, target_m1, target_m2, target_m3]
        
        tower_output_path_list = [f"/scratch1/smaruj/generate_cell_type_specific_features/tower_output_GM12878/model0/{chrom}_{pred_start}_{pred_end}_tower_out.pt",
                                f"/scratch1/smaruj/generate_cell_type_specific_features/tower_output_GM12878/model1/{chrom}_{pred_start}_{pred_end}_tower_out.pt",
                                f"/scratch1/smaruj/generate_cell_type_specific_features/tower_output_HCT116/model0/{chrom}_{pred_start}_{pred_end}_tower_out.pt",
                                f"/scratch1/smaruj/generate_cell_type_specific_features/tower_output_HCT116/model1/{chrom}_{pred_start}_{pred_end}_tower_out.pt"]
        
        wrapper = Ledidi(models, 
            input_loss=torch.nn.L1Loss(reduction='sum'), 
            output_loss=torch.nn.L1Loss(reduction='sum'),
            batch_size=1,
            max_iter=2000,
            early_stopping_iter=2000,
            return_history=False,
            verbose=True,
            bin_size=2048,
            input_mask_slices_0=[256],
            cropping_applied=64,
            output_mask_path=boundary_mask_path,
            use_semifreddo=True,
            semifreddo_temp_output_path_list=tower_output_path_list,
            punish_ctcf=False,
            ctcf_meme_path=None
            ).cuda()
        
        slice_0_torch = X[:, :, slice_0_start:slice_0_end]
        
        x_bar_slice_0, last_update, _, _, _ = wrapper.fit_transform(X=slice_0_torch, y_bar_list=y_bar_list)
        
        # Update df with last_accepted_step
        df.at[i, "last_accepted_step"] = last_update
        
        torch.save(x_bar_slice_0[:,:,padding:-padding], f"{args.pt_files_dir}/GM12878_weak_HCT116_strong_results/{chrom}_{pred_start}_{pred_end}_slice.pt")
        
    df.to_csv(f"{args.pt_files_dir}/fold{FOLD}_{target_c}_GM12878_weak_HCT116_strong_results.tsv", sep="\t", index=False)
    
    
if __name__ == "__main__":
    main()