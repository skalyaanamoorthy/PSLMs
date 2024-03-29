#!/bin/bash
#SBATCH --time=2-0:0:0
#SBATCH --account=def-skal
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=8GB
#SBATCH --nodes=1
#SBATCH --array=[0,5,10,13,23,32,99,103,105,125,129,132,159,223,224,225,236,237,238,241,243,244,415,416,417,426,487,503,577,584,586,612,620,634,637,644,694,698,711,713,744,763,764,768,775,811,822,823,824,860,871,873,899,909,919,922,923,928,934,935,939,955,960,969,983,998,999,1001,1002,1006,1038,1042,1058,1173,1176,1182,1188,1191,1194,1202,1203,1239,1277,1333,1338,1339,1351,1364,1365,1403,1413,1429,1473,1494,1496,1546,1549,1644,1672,1686,1716,1723,1724,2276,2282,2316,2335,2347,2353,2390,2395,2403,2408,2455,2547,2562,2613,2650,2652,2659,2662,2663,2694,2698,2721,2725,2749,2828,2830,2839,2840,2850,2861,3140,3149,3226,3231,3239,3247,3248,3251,3261,3284,3292,3311,3321,3323,3382,3411]

module load hh-suite/3.3.0
module load python/3.8
module load scipy-stack

### The below section looks up information on the protein like the chain and mutation of interest
IFS=','
text=$(cat ./data/q3421_mapped.csv | head -n $((${SLURM_ARRAY_TASK_ID}+2)) | tail -n 1)
read -a strarr <<< "$text"
code=${strarr[1]}; echo $code
chain=${strarr[2]}; echo $code
folder=${code}_${chain}

cd ./data/msas
name=$folder'_MSA'
echo 'filtering MSAs'

perl ~/projects/def-skal/sareeves/scripts/reformat.pl sto a2m $name ${name}.a2m > log_msat.txt
perl ~/projects/def-skal/sareeves/scripts/reformat.pl sto a3m $name ${name}.a3m -r >> log_msat.txt
hhfilter -i ${name}.a3m -o ${name}_full_cov75_id90.a3m -cov 75 -id 90 -maxseq 100000000 >> log_msat.txt
hhfilter -i ${name}.a3m -o ${name}_full_cov50_id80.a3m -cov 50 -id 80 -maxseq 100000000 >> log_msat.txt