mapped_loc="$1"
dest="$2"
m1=$(basename "$mapped_loc")
filename="${m1%.*}"
dir_path="$(dirname "$mapped_loc")"
dir_path="$(dirname "$dir_path")"
dir_path="$(dirname "$dir_path")"
echo $dir_path

echo "rsync $mapped_loc $dest/data/preprocessed/$m1"
rsync -vv $mapped_loc $dest/data/preprocessed/$m1
rsync -vv $mapped_loc $dest/data/inference/"${filename}_preds.csv"
rsync -vv $dir_path/sequences/fasta_up/* $dest/sequences/fasta_up/ 
rsync -vv $dir_path/structures/* $dest/structures/
rsync -vv $dir_path/DMS_Tranception/* $dest/DMS_Tranception/
rsync -vv $dir_path/DMS_MSA/* $dest/DMS_MSA/
