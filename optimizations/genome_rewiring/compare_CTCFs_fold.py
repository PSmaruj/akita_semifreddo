from pyfaidx import Fasta
import torch
import pandas as pd
import numpy as np
from memelite import fimo


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


def get_sequence(genome, chrom, start, end):
    return str(genome[chrom][start:end].seq.upper())


def jaccard_index(set1, set2):
    if not set1 and not set2:
        return 1.0
    union = set1 | set2
    return len(set1 & set2) / len(union) if union else 0.0


def one_hot_encode(seq):
    mapping = {"A":0, "C":1, "G":2, "T":3}
    ohe = np.zeros((4, len(seq)), dtype=np.float32)
    for i, base in enumerate(seq):
        if base in mapping:
            ohe[mapping[base], i] = 1.0
    return ohe


def hits_to_site_set(hits_df, bin_size=10):
    site_set = set()
    for _, row in hits_df.iterrows():
        start, end, strand = row['start'], row['end'], row['strand']
        center = (start + end) // 2
        binned_pos = round(center / bin_size) * bin_size
        site_set.add((binned_pos, strand))
    return site_set


def main():
    FOLD = 0
    input_dir = "/scratch1/smaruj/genomic_map_transformation"
    df = pd.read_csv(f"{input_dir}/df_select_fold{FOLD}.tsv", sep="\t")
    
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
    
    genome = Fasta("/project2/fudenber_735/genomes/mm10/mm10.fa")
    
    CTCF_PWM = "/home1/smaruj/IterativeMutagenesis/MA0139.1.meme"
    
    pwm_CTCF = read_meme_pwm_as_numpy(CTCF_PWM)
    pwm_CTCF_tensor = torch.from_numpy(pwm_CTCF.T).float()
    motifs_dict = {"CTCF": pwm_CTCF_tensor}
    
    df['Jaccard_init_opt'] = 0.0
    df['Jaccard_opt_target'] = 0.0
    
    for idx, row in df.iterrows():
        # original sequence
        ohe_seq = one_hot_encode(get_sequence(genome, row["chrom"], row["start"], row["end"]))
        ohe_tensor = torch.tensor(ohe_seq, dtype=torch.float32).unsqueeze(0)
        
        orig_hits = fimo(
            motifs=motifs_dict,
            sequences=ohe_tensor.cpu().detach().numpy(),
            threshold=1e-4,
            reverse_complement=True
        )[0]
        
        orig_site_set = hits_to_site_set(orig_hits)
        
        # optimized seq
        opt_path = f"/scratch1/smaruj/genomic_map_transformation/results_fold0_repeated/{row['chrom']}_{row['start']}_{row['end']}_seq.pt"
        opt_tensor = torch.load(opt_path)  # shape (4, L)
        
        opt_hits = fimo(
            motifs=motifs_dict,
            sequences=opt_tensor.cpu().detach().numpy(),
            threshold=1e-4,
            reverse_complement=True
        )[0]
    
        opt_site_set = hits_to_site_set(opt_hits)
        
        # target sequence
        tgt_seq = one_hot_encode(get_sequence(genome, row["target_chrom"], row["target_start"], row["target_end"]))
        tgt_tensor = torch.tensor(tgt_seq, dtype=torch.float32).unsqueeze(0)
    
        tgt_hits = fimo(
            motifs=motifs_dict,
            sequences=tgt_tensor.cpu().detach().numpy(),
            threshold=1e-4,
            reverse_complement=True
        )[0]
        
        tgt_site_set = hits_to_site_set(tgt_hits)
        
        df.at[idx, 'Jaccard_init_opt'] = jaccard_index(orig_site_set, opt_site_set)
        df.at[idx, 'Jaccard_opt_target'] = jaccard_index(opt_site_set, tgt_site_set)
        
    df.to_csv("/scratch1/smaruj/genomic_map_transformation/ctcf_jaccard_results_repeated.tsv", sep="\t", index=False)


if __name__ == "__main__":
    main()