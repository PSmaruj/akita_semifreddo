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
    parser.add_argument("--target", type=str, required=True, help="Desired target constant value")
    parser.add_argument("--model_path", type=str, required=True, help="Path to model .pt file")
    parser.add_argument("--input_dir", type=str, required=True, help="Base path to input .pt and .tsv files")
    parser.add_argument("--output_dir", type=str, required=True, help="Base path to write output")
    parser.add_argument("--boundary_mask_path", type=str, required=True, help="Path to boundary mask .pt file")
    
    parser.add_argument("--bin_size", type=int, default=2048, help="Bin size for model input")
    parser.add_argument("--cropping_applied", type=int, default=64, help="Cropping applied in the model")
    parser.add_argument("--padding_bins", type=int, default=2, help="Number of bins to pad input slices")
    parser.add_argument("--max_iter", type=int, default=4000, help="Maximum number of optimization steps")
    parser.add_argument("--early_stopping_iter", type=int, default=100, help="Early stopping threshold")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    
    parser.add_argument("--input_loss", type=str, default="l1", choices=["l1", "mse"],
                        help="Loss function for input (l1, mse)")
    parser.add_argument("--output_loss", type=str, default="l1", choices=["l1", "mse"],
                        help="Loss function for output (l1, mse)")
    parser.add_argument("--tau", type=float, default=1.0, help="Tau for Gumbel softmax")
    parser.add_argument("--l", type=float, default=0.1, help="Lambda to balance input/output losses")
    parser.add_argument("--lr", type=float, default=1.0, help="Learning rate for optimizer")
    parser.add_argument("--eps", type=float, default=1e-4, help="Epsilon for convergence criterion")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    
    return parser.parse_args()


def get_loss_fn(name):
    if name == "l1":
        return torch.nn.L1Loss(reduction='sum')
    elif name == "mse":
        return torch.nn.MSELoss(reduction='sum')
    else:
        raise ValueError(f"Unsupported loss function: {name}")


def main():
    args = parse_args()
    target_c = float(args.target)
    
    input_loss_fn = get_loss_fn(args.input_loss)
    output_loss_fn = get_loss_fn(args.output_loss)
    
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
    df["best_output_loss"] = -1.0  # initialize with placeholder
    df["num_edits"] = -1  # initialize with placeholder
    df["best_PearsonR"] = -1.0  # initialize with placeholder
    
    for i, row in enumerate(df.itertuples(index=False)):
        chrom, pred_start, pred_end = row.chrom, row.centered_start, row.centered_end
        
        print(f"Boundary generation for genome location: {chrom}:{pred_start}-{pred_end}")
        
        X = torch.load(f"{args.input_dir}/ohe_X/fold{FOLD}/{chrom}_{pred_start}_{pred_end}_X.pt", weights_only=True, map_location=device)
        target = torch.load(f"{args.input_dir}/targets_{target_c}/fold{FOLD}/{chrom}_{pred_start}_{pred_end}_target.pt", weights_only=True, map_location=device)
        tower_output_path = f"{args.input_dir}/tower_outputs/fold{FOLD}/{chrom}_{pred_start}_{pred_end}_tower_out.pt"
        
        wrapper = Ledidi(model, 
                    input_loss=input_loss_fn, 
                    output_loss=output_loss_fn,
                    tau=args.tau, 
                    l=args.l, 
                    lr=args.lr, 
                    eps=args.eps, 
                    batch_size=args.batch_size,
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
        
        x_bar_slice_0, last_update, best_output_loss, num_edits, best_PearsonR = wrapper.fit_transform(X=slice_0_torch, y_bar=target)
        
        # Update df with last_accepted_step
        df.at[i, "last_accepted_step"] = last_update
        df.at[i, "best_output_loss"] = best_output_loss
        df.at[i, "num_edits"] = num_edits
        df.at[i, "best_PearsonR"] = best_PearsonR
        
        torch.save(x_bar_slice_0[:,:,padding:-padding], f"{args.output_dir}/early_stop_{args.early_stopping_iter}/{chrom}_{pred_start}_{pred_end}_slice.pt")
        
        # saving for a particular seed
        # torch.save(x_bar_slice_0[:,:,padding:-padding], f"{args.output_dir}/reproducibility_fold0_-0.5/seed{args.seed}/{chrom}_{pred_start}_{pred_end}_slice.pt")
        
    df.to_csv(f"{args.output_dir}/early_stop_{args.early_stopping_iter}_df.tsv", sep="\t", index=False)
    
    
if __name__ == "__main__":
    main()