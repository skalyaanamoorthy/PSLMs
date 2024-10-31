Sequence similarity was done using

mmseqs createdb ../all_seqs.fasta sequencesDB
mmseqs createindex sequencesDB tmp --threads 8
mmseqs search sequencesDB sequencesDB resultDB tmp --threads 8 -s 9.5 -e 0.1
mmseqs convertalis sequencesDB sequencesDB resultDB result.m8

Structure similarity was done using

cd ~/PSLMs/structures/single_chains/
export FATCAT=/home/sareeves/software/FATCAT-dist
~/software/FATCAT-dist/FATCATMain/FATCATQue.pl timeused ../../data/all_pairs.txt -q > allpair.aln
