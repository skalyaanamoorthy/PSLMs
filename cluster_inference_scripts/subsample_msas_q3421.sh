#!/bin/bash
#SBATCH --time=4:0:0
#SBATCH --account=def-skal
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=8GB
#SBATCH --array=0,5,10,13,23,32,99,103,105,125,129,132,159,223,224,225,236,237,238,241,243,244,415,416,417,426,487,503,577,584,586,612,620,634,637,644,694,698,711,713,744,763,764,768,775,811,822,823,824,860,871,873,899,909,919,922,923,928,934,935,939,955,960,969,983,998,999,1001,1002,1006,1038,1042,1058,1173,1176,1182,1188,1191,1194,1202,1203,1239,1277,1333,1338,1339,1351,1364,1365,1403,1413,1429,1473,1494,1496,1546,1549,1644,1672,1686,1716,1723,1724,2276,2282,2316,2335,2347,2353,2390,2395,2403,2408,2455,2547,2562,2613,2650,2659,2652,2662,2663,2694,2698,2721,2725,2749,2828,2830,2839,2840,2850,2861,3140,3149,3226,3231,3239,3247,3248,3251,3261,3284,3292,3311,3321,3323,3382,3411

# the array above is the sequence output by preprocess.py

source ../pslm/bin/activate

### The below section looks up information on the protein like the sequence and MSA of interest
IFS=','
text=$(cat ../data/preprocessed/q3421_mapped.csv | head -n $((${SLURM_ARRAY_TASK_ID}+2)) | tail -n 1)
read -a strarr <<< "$text"
code=${strarr[1]}; echo $code
chain=${strarr[2]}; echo $chain
wt=${strarr[3]}; echo $wt
pos=${strarr[4]}; echo $pos
mut=${strarr[5]}; echo $mut
up=${strarr[6]}; echo $up
seq=${strarr[7]}; echo $seq
msa=${strarr[8]}; echo $msa

python ../inference_scripts/subsample_one.py --msa_file $msa --dataset 'q3421' #--neff_only
