#!/bin/bash
#SBATCH --time=2:0:0
#SBATCH --account=rrg-skal
#SBATCH --gpus-per-node=a100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --output=%A-%x

module load StdEnv/2020  gcc/9.3.0  cuda/11.4
module load hh-suite/3.3.0
module load python/3.8
module load scipy-stack
module load arrow/10.0.1

source ../pslm/bin/activate

#python ../inference_scripts/tranception_.py --checkpoint ~/scratch/software/Tranception_Large --num_workers 1 --db_loc '../data/s669_mapped.csv' --output '../data/s669_mapped_preds.csv' --tranception_loc ~/scratch/software/Tranception
python ../inference_scripts/tranception_.py --checkpoint ~/scratch/software/Tranception_Large --num_workers 1 --db_loc '../data/fireprot_mapped.csv' --output '../data/fireprot_mapped_preds.csv' --tranception_loc ~/scratch/software/Tranception
#python ../inference_scripts/tranception_.py --checkpoint ~/scratch/software/Tranception_Large --num_workers 1 --db_loc '../data/Ssym_mapped.csv' --output '../data/Ssym_mapped_preds.csv' --tranception_loc ~/scratch/software/Tranception
#python ../inference_scripts/tranception_.py --checkpoint ~/scratch/software/Tranception_Large --num_workers 1 --db_loc '../data/Q3421_mapped.csv' --output '../data/Q3421_mapped_preds.csv' --tranception_loc ~/scratch/software/Tranception
