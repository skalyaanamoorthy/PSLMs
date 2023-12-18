#!/bin/bash
#SBATCH --time=0:10:0
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

if ! test -e '../data/s669_mapped_preds.csv'; then cp '../data/s669_mapped.csv' '../data/s669_mapped_preds.csv'; fi
python ../inference_scripts/mpnn.py --db_loc '../data/s669_mapped.csv' --output '../data/s669_mapped_preds.csv' --mpnn_loc ~/software/ProteinMPNN --noise '20'
if ! test -e '../data/fireprot_mapped_preds.csv'; then cp '../data/fireprot_mapped.csv' '../data/fireprot_mapped_preds.csv'; fi
python ../inference_scripts/mpnn.py --db_loc '../data/fireprot_mapped.csv' --output '../data/fireprot_mapped_preds.csv' --mpnn_loc ~/software/ProteinMPNN --noise '20'
