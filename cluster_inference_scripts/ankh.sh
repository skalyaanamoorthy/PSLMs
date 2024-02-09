#!/bin/bash
#SBATCH --time=18:0:0
#SBATCH --account=rrg-skal
#SBATCH --gpus-per-node=1
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
python ../inference_scripts/ankh_.py --db_loc '../data/s669_mapped.csv' --output '../data/s669_mapped_preds.csv'
if ! test -e '../data/fireprot_mapped_preds.csv'; then cp '../data/fireprot_mapped.csv' '../data/fireprot_mapped_preds.csv'; fi
python ../inference_scripts/ankh_.py --db_loc '../data/fireprot_mapped.csv' --output '../data/fireprot_mapped_preds.csv'
if ! test -e '../data/ssym_mapped_preds.csv'; then cp '../data/ssym_mapped.csv' '../data/ssym_mapped_preds.csv'; fi
python ../inference_scripts/ankh_.py --db_loc '../data/ssym_mapped.csv' --output '../data/ssym_mapped_preds.csv'
if ! test -e '../data/q3421_mapped_preds.csv'; then cp '../data/q3421_mapped.csv' '../data/q3421_mapped_preds.csv'; fi
python ../inference_scripts/ankh_.py --db_loc '../data/q3421_mapped.csv' --output '../data/q3421_mapped_preds.csv'
