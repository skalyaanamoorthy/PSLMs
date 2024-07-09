#!/bin/bash
#SBATCH --time=0:30:0
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

if ! test -e '../data/inference/s669_mapped_preds.csv'; then cp '../data/preprocessed/s669_mapped.csv' '../data/inference/s669_mapped_preds.csv'; fi
python ../inference_scripts/mpnn.py --db_loc '../data/preprocessed/s669_mapped.csv' --output '../data/inference/s669_mapped_preds.csv' --mpnn_loc /home/sareeves/scratch/software/ProteinMPNN --noise '10'
if ! test -e '../data/inference/ssym_mapped_preds.csv'; then cp '../data/preprocessed/ssym_mapped.csv' '../data/inference/ssym_mapped_preds.csv'; fi
python ../inference_scripts/mpnn.py --db_loc '../data/preprocessed/ssym_mapped.csv' --output '../data/inference/ssym_mapped_preds.csv' --mpnn_loc /home/sareeves/scratch/software/ProteinMPNN --noise '10'
if ! test -e '../data/inference/q3421_mapped_preds.csv'; then cp '../data/preprocessed/q3421_mapped.csv' '../data/inference/q3421_mapped_preds.csv'; fi
python ../inference_scripts/mpnn.py --db_loc '../data/preprocessed/q3421_mapped.csv' --output '../data/inference/q3421_mapped_preds.csv' --mpnn_loc /home/sareeves/scratch/software/ProteinMPNN --noise '10'
if ! test -e '../data/inference/k3822_mapped_preds.csv'; then cp '../data/preprocessed/k3822_mapped.csv' '../data/inference/k3822_mapped_preds.csv'; fi
python ../inference_scripts/mpnn.py --db_loc '../data/preprocessed/k3822_mapped.csv' --output '../data/inference/k3822_mapped_preds.csv' --mpnn_loc /home/sareeves/scratch/software/ProteinMPNN --noise '10'

#if ! test -e '../data/inference/fireprot_mapped_preds.csv'; then cp '../data/preprocessed/fireprot_mapped.csv' '../data/inference/fireprot_mapped_preds.csv'; fi
#python ../inference_scripts/mpnn.py --db_loc '../data/inference/fireprot_mapped.csv' --output '../data/inference/fireprot_mapped_preds.csv' --mpnn_loc /home/sareeves/scratch/software/ProteinMPNN --noise '10'

python ../inference_scripts/mpnn.py --db_loc '../data/preprocessed/s669_mapped.csv' --output '../data/inference/s669_mapped_preds.csv' --mpnn_loc /home/sareeves/scratch/software/ProteinMPNN --noise '20'
python ../inference_scripts/mpnn.py --db_loc '../data/preprocessed/ssym_mapped.csv' --output '../data/inference/ssym_mapped_preds.csv' --mpnn_loc /home/sareeves/scratch/software/ProteinMPNN --noise '20'
python ../inference_scripts/mpnn.py --db_loc '../data/preprocessed/q3421_mapped.csv' --output '../data/inference/q3421_mapped_preds.csv' --mpnn_loc /home/sareeves/scratch/software/ProteinMPNN --noise '20'
python ../inference_scripts/mpnn.py --db_loc '../data/preprocessed/k3822_mapped.csv' --output '../data/inference/k3822_mapped_preds.csv' --mpnn_loc /home/sareeves/scratch/software/ProteinMPNN --noise '20'

#python ../inference_scripts/mpnn.py --db_loc '../data/inference/fireprot_mapped.csv' --output '../data/inference/fireprot_mapped_preds.csv' --mpnn_loc /home/sareeves/scratch/software/ProteinMPNN --noise '20'

python ../inference_scripts/mpnn.py --db_loc '../data/preprocessed/s669_mapped.csv' --output '../data/inference/s669_mapped_preds.csv' --mpnn_loc /home/sareeves/scratch/software/ProteinMPNN --noise '30'
python ../inference_scripts/mpnn.py --db_loc '../data/preprocessed/ssym_mapped.csv' --output '../data/inference/ssym_mapped_preds.csv' --mpnn_loc /home/sareeves/scratch/software/ProteinMPNN --noise '30'
python ../inference_scripts/mpnn.py --db_loc '../data/preprocessed/q3421_mapped.csv' --output '../data/inference/q3421_mapped_preds.csv' --mpnn_loc /home/sareeves/scratch/software/ProteinMPNN --noise '30'
python ../inference_scripts/mpnn.py --db_loc '../data/preprocessed/k3822_mapped.csv' --output '../data/inference/k3822_mapped_preds.csv' --mpnn_loc /home/sareeves/scratch/software/ProteinMPNN --noise '30'

#python ../inference_scripts/mpnn.py --db_loc '../data/preprocessed/fireprot_mapped.csv' --output '../data/inference/fireprot_mapped_preds.csv' --mpnn_loc /home/sareeves/scratch/software/ProteinMPNN --noise '30'