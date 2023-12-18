#!/bin/bash
#SBATCH --time=2-0:0:0
#SBATCH --account=def-skal
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=8GB
#SBATCH --nodes=1
#SBATCH --array=[969,1030,1265,1394,4043,4063,4128,4432]

module load hh-suite/3.3.0
module load python/3.8
module load scipy-stack

### The below section looks up information on the protein like the chain and mutation of interest
IFS=','
text=$(cat data/fireprot_unique_muts_offsets.csv | head -n $((${SLURM_ARRAY_TASK_ID}+2)) | tail -n 1)
read -a strarr <<< "$text"
folder=${strarr[1]}; echo $folder
cd 'fireprot_alignments_full_july_24'
name=$folder'_MSA'
echo 'filtering MSAs'

#perl ~/projects/rrg-skal/sareeves/scripts/reformat.pl sto a3m $name ${name}.a3m -r > log_msat.txt 
hhfilter -i ${name}.a3m -o ${name}_full_cov75_id90.a3m -cov 75 -id 90 -maxseq 100000000 >> log_msat.txt
