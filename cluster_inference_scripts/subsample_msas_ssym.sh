#!/bin/bash
#SBATCH --time=1:0:0
#SBATCH --account=def-skal
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4GB
#SBATCH --array=88,678
#
#0,19,22,26,27,43,49,55,76,88,121,163,167,170,481,482,487,647

# the array above is the sequence output by preprocess.py

source ../pslm/bin/activate

### The below section looks up information on the protein like the sequence and MSA of interest
IFS=','
text=$(cat ../data/preprocessed/ssym_mapped.csv | head -n $((${SLURM_ARRAY_TASK_ID}+2)) | tail -n 1)
read -a strarr <<< "$text"
code=${strarr[1]}; echo $code
chain=${strarr[2]}; echo $chain
wt=${strarr[3]}; echo $wt
pos=${strarr[4]}; echo $pos
mut=${strarr[5]}; echo $mut
up=${strarr[6]}; echo $up
seq=${strarr[7]}; echo $seq
msa=${strarr[8]}; echo $msa

python ../inference_scripts/subsample_one.py --msa_file $msa --dataset 'ssym' #--neff_only
