#!/bin/bash
#SBATCH --time=6:0:0
#SBATCH --account=rrg-skal
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=8GB
#SBATCH --array=0,6,11,21,31,36,41,51,121,131,134,162,166,167,171,198,199,267,268,282,286,302,325,326,329,333,334,336,346,347,523,524,525,534,604,605,621,623,699,701,711,718,720,725,728,754,767,781,797,802,810,822,882,888,892,905,907,909,940,941,960,961,964,968,969,985,992,1030,1078,1079,1130,1140,1142,1145,1154,1182,1198,1222,1251,1261,1265,1275,1281,1289,1292,1360,1362,1363,1364,1378,1394,1399,1408,1422,1437,1438,1440,1444,1446,1449,1452,1455,1461,1463,1471,1475,1478,1512,1516,1518,1533,1668,1680,1683,1691,1692,1701,1707,1715,1722,1730,1736,1740,1792,1795,1798,2735,2789,2802,2957,2963,2964,2976,3021,3022,3023,3024,3031,3077,3087,3103,3167,3188,3190,3256,3260,3372,3386,3464,3465,4034,4043,4055,4058,4063,4103,4122,4128,4140,4141,4147,4187,4235,4236,4254,4255,4257,4262,4309,4401,4432,4433,4510,4522,4577,4580,4581,4583,4597,4607,4609,4617,4649,4653,4678,4684,4708,4710,4741,4827,4828,4830,4847,4848,4868,4872,4883,5212,5290,5299,5378,5379,5392,5395,5405,5450,5458,5459,5473,5483,5519,5529,5532,5541,5562,5598,5676,5766,5777,5781,5784,5846,5865,5897,5907,5917,5946,6157,6160,6311

# the array above is the sequence output by preprocess.py

module purge
module load rosetta

### The below section looks up information on the protein like the chain and mutation of interest
IFS=','
mkdir structures_rosetta
# make sure the path fireprot_unique_muts_offsets.csv is replaced by the actual location
text=$(cat ./data/fireprot_unique_muts_offsets.csv | head -n $((${SLURM_ARRAY_TASK_ID}+2)) | tail -n 1)
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
echo $runtime > 'relax_runtime_cart_'$code'_'$chain'.txt'
