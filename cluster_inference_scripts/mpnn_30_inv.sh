#!/bin/bash
#SBATCH --time=1:0:0
#SBATCH --account=def-skal
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

if ! test -e '../data/s669_mapped_preds.csv'; then cp '../data/s669_mapped.csv' '../data/s669_mapped_preds.csv'; fi
python ../inference_scripts/mpnn_inv.py --db_loc '../data/s669_mapped.csv' --output '../data/s669_mapped_preds.csv' --noise 30 --mpnn_loc '~/software/ProteinMPNN'
