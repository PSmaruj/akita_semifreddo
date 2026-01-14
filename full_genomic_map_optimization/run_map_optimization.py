import torch
import pandas as pd
import argparse
import sys
import os

sys.path.append(os.path.abspath("/home1/smaruj/pytorch_akita/"))
from akita_model.model import SeqNN

import sys
sys.path.insert(0, "/home1/smaruj/ledidi/ledidi/")
from ledidi_whole_seq import Ledidi


def parse_args():
    parser = argparse.ArgumentParser(description="Run Ledidi on selected genomic loci.")
    parser.add_argument("--fold", type=int, required=True, help="Fold number to process")
    parser.add_argument("--model_path", type=str, required=True, help="Path to model .pt file")
    parser.add_argument("--input_dir", type=str, required=True, help="Base path to input .pt and .tsv files")
    parser.add_argument("--output_dir", type=str, required=True, help="Base path to write output")
    
    parser.add_argument("--max_iter", type=int, default=2000, help="Maximum number of optimization steps")
    parser.add_argument("--early_stopping_iter", type=int, default=2000, help="Early stopping threshold")
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    model = SeqNN()
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()

    FOLD = args.fold
    
    df = pd.read_csv(f"{args.input_dir}/df_select_fold{FOLD}.tsv", sep="\t")
    
    # test
    # df = df[:5]
    
    # shifting genomic cordinates and assigning them as a target folding
    df['target_chrom'] = df['chrom'].shift(-1)
    df['target_start'] = df['start'].shift(-1)
    df['target_end'] = df['end'].shift(-1)
    
    # Fill last row with values from the first row
    df.loc[df.index[-1], 'target_chrom'] = df.loc[df.index[0], 'chrom']
    df.loc[df.index[-1], 'target_start'] = df.loc[df.index[0], 'start']
    df.loc[df.index[-1], 'target_end'] = df.loc[df.index[0], 'end']
    
    # Convert to int
    df['target_start'] = df['target_start'].astype(int)
    df['target_end'] = df['target_end'].astype(int)
    
    df["last_accepted_step"] = -1  # initialize with placeholder
    
    for i, row in enumerate(df.itertuples(index=False)):
        chrom, pred_start, pred_end = row.chrom, row.start, row.end
        target_chrom, target_start, target_end = row.target_chrom, row.target_start, row.target_end
        
        print(f"Map transformation starting from genome location: {chrom}:{pred_start}-{pred_end}")
        print(f"To a genome location: {target_chrom}:{target_start}-{target_end}")
        
        # starting from genomic folding
        X = torch.load(f"{args.input_dir}/ohe_X_fold{FOLD}/{chrom}_{pred_start}_{pred_end}_X.pt", weights_only=True, map_location=device)
        
        target = torch.load(f"{args.input_dir}/genomic_targets_fold{FOLD}/{target_chrom}_{target_start}_{target_end}_target.pt", weights_only=True, map_location=device)
        
        wrapper = Ledidi(model, 
                    input_loss=torch.nn.L1Loss(reduction='sum'), 
                    output_loss=torch.nn.L1Loss(reduction='sum'),
                    batch_size=1,
                    max_iter=args.max_iter,
                    early_stopping_iter=args.early_stopping_iter,
                    return_history=False,
                    verbose=True,
                    l=0.05
                    ).cuda()
        
        generated_seq, last_update = wrapper.fit_transform(X, target)
        
        # Update df with last_accepted_step
        df.at[i, "last_accepted_step"] = last_update
        
        torch.save(generated_seq, f"{args.output_dir}/results_fold{FOLD}_repeated/{chrom}_{pred_start}_{pred_end}_seq.pt")

    df.to_csv(f"{args.output_dir}/genomic_optimization_results_fold{FOLD}_with_steps_repeated.tsv", sep="\t", index=False)
    
    
if __name__ == "__main__":
    main()