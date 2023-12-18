#!/bin/bash
#SBATCH --time=36:0:0
#SBATCH --account=rrg-skal
#SBATCH --gpus-per-node=a100:1
#SBATCH --cpus-per-task=16
#SBATCH --nodes=1
#SBATCH --mem 64GB
#SBATCH --output=%A-%x

module purge
module load python
module load scipy-stack
module load StdEnv/2020  gcc/9.3.0  cuda/11.4

source ../pslm/bin/activate

#python ../inference_scripts/msa_transformer.py --db_loc '../data/s669_mapped.csv' --output '../data/s669_mapped_preds.csv'
python ../inference_scripts/msa_transformer.py --db_loc '../data/fireprot_mapped.csv' --output '../data/fireprot_mapped_preds.csv'
