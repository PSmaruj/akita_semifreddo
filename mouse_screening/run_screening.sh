#!/bin/bash

python run_screening.py \
  --fragments /scratch1/smaruj/mouse_screening/chr11_69057142_69276687_fragments.tsv \
  --backgrounds /scratch1/smaruj/background_generation/background_sequences_scd30_totvar1300.fasta \
  --genome /project2/fudenber_735/genomes/mm10/mm10.fa \
  --model /home1/smaruj/pytorch_akita/model_0_v2_finetuned_correctly.pt \
  --out /scratch1/smaruj/mouse_screening/chr11_screening_results_all_bg.tsv