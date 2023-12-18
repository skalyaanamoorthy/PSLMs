#!/bin/bash
#SBATCH --time=6:0:0
#SBATCH --account=rrg-skal
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=8GB
#SBATCH --array=0-669

module purge
module load rosetta
module load python
module load scipy-stack

### The below section looks up information on the protein like the chain and mutation of interest
IFS=','
#text=$(cat s669_unique_muts_offsets.csv | head -n $((${SLURM_ARRAY_TASK_ID}+2)) | tail -n 1)
# make sure the path s669_mapped.csv is replaced by the actual location
text=$(python -c "import pandas as pd; row = pd.read_csv('./data/s669_mapped.csv').iloc[${SLURM_ARRAY_TASK_ID}, :][['code', 'chain', 'wild_type', 'position', 'mutation', 'mutant_pdb_file']]; print(','.join([str(i) for i in list(row)]))")
read -a strarr <<< "$text"

code=${strarr[0]}; echo $code
chain=${strarr[1]}; echo $chain
wt=${strarr[2]}; echo $wt
pos=${strarr[3]}; echo $pos
mut=${strarr[4]}; echo $mut
pdb=${strarr[5]}; echo $pdb

#pdb='structures_mut/'$code'_'$pos$mut'.pdb'
folder='predictions/'$code'_'$chain'/'$code'_'$pos$mut'/inv_robetta'

mkdir $folder
cd $folder

infile='../../../../'$pdb

# the following section is not necessary due to previous preprocessing
### Only read up to the first structure to handle NMR
#egrep '^ATOM.{17}'$chain'|^HETATM.{15}'$chain'|^TER.{18}'$chain'|^SSBOND|^CISPEP' ../../$pdb | sed '/TER/q' > $infile
#egrep '^HETNAM' ../../$pdb > hetnams.log

#infile=$folder'_'$chain'_clean.pdb'
#infile=$pdb

start=`date +%s`

### Relax protocol specified by Hoie et al. 2021 exactly
srun $EBROOTROSETTA/bin/relax.mpi.linuxiccrelease \
-s $infile \
-database $ROSETTA3_DB \
-relax:constrain_relax_to_start_coords \
-in:ignore_unrecognized_res \
-in:missing_density_to_jump \
-out:nstruct 1 \
-relax:coord_constrain_sidechains \
-relax:cartesian \
-corrections:beta_nov16_cart \
-packing:ex1 \
-packing:ex2 \
-relax:min_type lbfgs_armijo_nonmonotone \
-packing:flip_HNQ \
-packing:no_optH false \
-out:suffix _inv_minimized \
#-out:overwrite \

end=`date +%s`

runtime=$((end-start))
echo $runtime > 'relax_runtime_cart_'$code'_'$chain'_inv.txt'
