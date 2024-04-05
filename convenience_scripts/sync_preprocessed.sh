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
rsync -rvv $dir_path/sequences $dest 
rsync -vv $dir_path/structures $dest
rsync -vv $dir_path/DMS_Tranception $dest
rsync -vv $dir_path/DMS_MSA $dest
