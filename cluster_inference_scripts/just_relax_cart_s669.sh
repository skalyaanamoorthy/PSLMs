#!/bin/bash
#SBATCH --time=24:0:0
#SBATCH --account=rrg-skal
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=8GB
#SBATCH --array=0,1,14,18,20,21,23,73,74,78,79,80,81,94,105,132,133,182,183,185,187,188,189,192,195,196,202,203,206,209,214,218,235,237,243,263,267,272,273,280,281,285,287,294,298,303,307,308,311,313,317,319,320,366,417,420,429,431,437,439,442,443,445,446,463,531,533,535,564,572,574,575,587,590,595,626,627,628,632,635,636,638,644,648,649,650,652,655,658,661,662,663,665,666

module purge
module load rosetta

### The below section looks up information on the protein like the chain and mutation of interest
IFS=','
# make sure the path s669_unique_muts_offsets.csv is replaced by the actual location
text=$(cat ./data/s669_unique_muts_offsets.csv | head -n $((${SLURM_ARRAY_TASK_ID}+2)) | tail -n 1)
read -a strarr <<< "$text"
code=${strarr[1]}; echo $code
chain=${strarr[2]}; echo $chain
wt=${strarr[3]}; echo $wt
pos=${strarr[4]}; echo $pos
mut=${strarr[5]}; echo $mut

pdb='structures/'$code'_'$chain'.pdb'
folder='predictions/'$code'_'$chain

cd $folder

# make sure this is where you want to put your rosetta structures
# this is typically the base directory of the repository
infile='../../structures_rosetta/'$code'_'$chain'.pdb'

if [ ! -d "../../structures_rosetta/" ]; then
  mkdir -p "../../structures_rosetta/"
fi

### Only read up to the first structure to handle NMR
egrep '^ATOM.{17}'$chain'|^HETATM.{15}'$chain'|^TER.{18}'$chain'|^SSBOND|^CISPEP' ../../$pdb | sed '/TER/q' > $infile
egrep '^HETNAM' ../../$pdb > hetnams.log

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
-out:suffix _minimized \
-out:overwrite \

end=`date +%s`

runtime=$((end-start))
echo $runtime > 'relax_runtime_cart_'$code'_'$chain'_hets.txt'
