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

if ! test -e '../data/s669_mapped_preds.csv'; then cp '../data/s669_mapped.csv' '../data/s669_mapped_preds.csv'; fi
python ../inference_scripts/tranception_.py --checkpoint ~/software/Tranception_Large --num_workers 1 --db_loc '../data/s669_mapped.csv' --output '../data/s669_mapped_preds.csv' --tranception_loc ~/software/Tranception
if ! test -e '../data/fireprot_mapped_preds.csv'; then cp '../data/fireprot_mapped.csv' '../data/fireprot_mapped_preds.csv'; fi
python ../inference_scripts/tranception_.py --checkpoint ~/software/Tranception_Large --num_workers 1 --db_loc '../data/fireprot_mapped.csv' --output '../data/fireprot_mapped_preds.csv' --tranception_loc ~/software/Tranception
