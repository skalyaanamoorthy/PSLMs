#!/bin/bash
#SBATCH --time=4-0:0:0 
#SBATCH --account=rrg-skal 
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=4GB
#SBATCH --nodes 1
#SBATCH --array=[0,1,14,18,20,21,23,73,74,78,79,80,81,94,105,132,133,182,183,185,187,188,189,192,195,196,202,203,206,209,214,218,235,237,243,263,267,272,273,280,281,285,287,294,298,303,307,308,311,313,317,319,320,366,417,420,429,431,437,439,442,443,445,446,463,531,533,535,564,572,574,575,587,590,595,626,627,628,632,635,637,643,647,648,649,651,654,657,660,661,662,664,665,666]

module purge

module load hmmer/3.2.1
IFS=','
text=$(cat ./data/s669_unique_muts_offsets.csv | head -n $((${SLURM_ARRAY_TASK_ID}+2)) | tail -n 1)
read -a strarr <<< "$text"
folder=${strarr[1]}; echo $folder
#cd $folder
cd msas_s669_aug_5

name=$(find ../sequences/fasta_up/ -name $folder'_[0-9A-Z].fa')
th=$(cat $name | tail -n 1 | wc -m) 
th=$(( th / 2 ))
echo $th
echo ${folder}_MSA

jackhmmer --incT $th --cpu 16 -A ${folder}_MSA -N 8 $name ../../uniref/uniref100.fasta
