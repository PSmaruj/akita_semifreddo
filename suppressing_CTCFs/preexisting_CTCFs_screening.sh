python CTCFs_screening.py \
    --ctcf_df /scratch1/smaruj/suppressing_CTCFs/results/preexisting_ctcf_df.tsv \
    --slice_dir /scratch1/smaruj/suppressing_CTCFs/results \
    --fullX_dir /scratch1/smaruj/suppressing_CTCFs/ohe_X \
    --background_fasta /scratch1/smaruj/background_generation/background_sequences_scd30_totvar1300.fasta \
    --model_path /home1/smaruj/pytorch_akita/model_0_v2_finetuned_correctly.pt \
    --output_df /scratch1/smaruj/suppressing_CTCFs/results/preexisting_ctcf_df_with_SCD.tsv