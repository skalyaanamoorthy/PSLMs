#!/bin/bash
#SBATCH --time=12:0:0
#SBATCH --account=rrg-skal
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=8GB
#SBATCH --array=0-6312
#SBATCH --mail-user=sareeves@uwaterloo.ca
#SBATCH --mail-type=FAIL

module purge
module load rosetta
module load python/3.8
module load scipy-stack

# This file is intended to be used to make ddg predictions on relaxed structures
# generated by just_relax_cart_fireprot.sh

### The below section looks up information on the protein like the chain and mutation of interest
IFS=','

# replace file path to the file generated by preprocess.py
text=$(cat ./data/fireprot_unique_muts_offsets.csv | head -n $((${SLURM_ARRAY_TASK_ID}+2)) | tail -n 1)
read -a strarr <<< "$text"
code=${strarr[1]}; echo $folder
chain=${strarr[2]}; echo $chain
wt=${strarr[3]}; echo $wt
pos=${strarr[4]}; echo $pos
mut=${strarr[5]}; echo $mut
# offset_up not used for s669
offset_up=${strarr[6]}; echo $offset_up
offset_ros=${strarr[7]}; echo $offset_ros

# make sure this corresponds to your predictions folder
folder='./predictions/'$code'_'$chain

cd $folder

minfile=$code'_'$chain'_minimized_0001.pdb'
echo $minfile

# sets up the input file that Rosetta will read
ros_pos=$(($pos+$offset_up-$offset_ros))
mutation="$wt$ros_pos$mut"
echo $mutation
mkdir $code'_'$pos$mut
cd $code'_'$pos$mut
echo -e "total 1\n1\n$mutation" > $mutation.mut

start=`date +%s`

### Cartesian ddg protocol from Hoie et al., 2021
$EBROOTROSETTA/bin/cartesian_ddg.mpi.linuxiccrelease \
-in:file:s ../$minfile \
-database $ROSETTA3_DB \
-ddg:iterations 3 \
-ddg:dump_pdbs true \
-in:missing_density_to_jump \
-ddg:mut_file $mutation.mut \
-ddg:bbnbrs 1 \
-corrections:beta_nov16_cart true \
-packing:ex1 \
-packing:ex2 \
-score:fa_max_dis 9.0 \
-ddg:optimize_proline true \
-ddg:legacy true 
#-ddg:mut_only

end=`date +%s`
runtime=$((end-start))
echo $runtime > 'runtime_'$mutation'_cart.txt'