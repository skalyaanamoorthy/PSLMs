#!/bin/bash

# Check if the correct number of arguments is given
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <input_folder> <output_folder>"
    exit 1
fi

INPUT_FOLDER="$1"
OUTPUT_FOLDER="$2"

# Check if input folder exists
if [ ! -d "$INPUT_FOLDER" ]; then
    echo "Input folder '$INPUT_FOLDER' does not exist."
    exit 1
fi

# Create output folder if it doesn't exist
if [ ! -d "$OUTPUT_FOLDER" ]; then
    mkdir -p "$OUTPUT_FOLDER"
fi

# Process each FASTA file in the input folder
for fasta_file in "$INPUT_FOLDER"/*.a3m; do
    filename=$(basename "$fasta_file")
    output_file="$OUTPUT_FOLDER/$filename"
    echo "Processing $fasta_file -> $output_file"
    python clean_msa.py "$fasta_file" "$output_file"
done

echo "All files processed."

