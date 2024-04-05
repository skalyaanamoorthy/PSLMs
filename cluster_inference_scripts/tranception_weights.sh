#!/bin/bash
#SBATCH --time=6:0:0
#SBATCH --account=rrg-skal
#SBATCH --gpus-per-node=a100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --output=%A-%x

module load StdEnv/2020  gcc/9.3.0  cuda/11.4
module load hh-suite/3.3.0
module load python/3.8
module load scipy-stack
module load arrow/10.0.1

source ../pslm/bin/activate

#python ../inference_scripts/tranception_.py --checkpoint ~/scratch/software/Tranception_Large --num_workers 1 --db_loc '../data/preprocessed/proteingym_mapped.csv' --output '../data/inference/proteingym_mapped_preds.csv' --tranception_loc ~/scratch/software/Tranception --ref_file '../data/DMS_substitutions.csv'

#if ! test -e '../data/s669_mapped_preds.csv'; then cp '../data/s669_mapped.csv' '../data/s669_mapped_preds.csv'; fi
#python ../inference_scripts/tranception_.py --checkpoint ~/software/Tranception_Large --num_workers 1 --db_loc '../data/s669_mapped.csv' --output '../data/s669_mapped_preds.csv' --tranception_loc ~/software/Tranception
#if ! test -e '../data/inference/korpm_mapped_preds.csv'; then cp '../data/preprocessed/korpm_mapped.csv' '../data/inference/korpm_mapped_preds.csv'; fi
#python ../inference_scripts/tranception_.py --checkpoint ~/scratch/software/Tranception_Large --num_workers 1 --db_loc '../data/preprocessed/korpm_mapped.csv' --output '../data/inference/korpm_mapped_preds.csv' --tranception_loc ~/scratch/software/Tranception --use_weights

python ../inference_scripts/tranception_.py --checkpoint ~/scratch/software/Tranception_Large --num_workers 1 --db_loc '../data/preprocessed/q3421_mapped.csv' --output '../data/inference/q3421_mapped_preds.csv' --tranception_loc ~/scratch/software/Tranception --use_weights
