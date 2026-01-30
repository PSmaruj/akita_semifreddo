python CTCFs_screening.py \
    --ctcf_df /scratch1/smaruj/suppressing_CTCFs/results_repeated/preexisting_ctcf_df.tsv \
    --slice_dir /scratch1/smaruj/suppressing_CTCFs/results_repeated \
    --fullX_dir /scratch1/smaruj/suppressing_CTCFs/ohe_X \
    --background_fasta /scratch1/smaruj/background_generation/background_sequences_scd30_totvar1300.fasta \
    --model_path /scratch1/smaruj/Akita_pytorch_models/finetuned/mouse_models/Hsieh2019_mESC/models/Akita_v2_mouse_Hsieh2019_mESC_model0_finetuned.pth \
    --output_df /scratch1/smaruj/suppressing_CTCFs/results_repeated/preexisting_ctcf_df_with_SCD.tsv
