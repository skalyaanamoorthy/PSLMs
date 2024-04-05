#!/bin/bash

# Directory to move files to
destination="/home/sareeves/PSLMs/data/rosetta_predictions"

# Iterate over .ddg and .txt files in the predictions directory
for file in /mnt/h/rosetta_predictions_korpm/*/*.ddg; do
    # Extract the last two parts of the filename
    base=$(basename "$file")
    dir=$(dirname "$file")
    lastdir=$(basename "$dir")
    sec_lastdir=$(basename "$(dirname "$dir")")

    # Concatenate the last two parts of the filename
    newname="${lastdir}.ddg"

    # Move the file to the destination directory
    cp -v "$file" "${destination}/${newname}"
done

for file in /mnt/h/rosetta_predictions_korpm/*/*.txt; do
    # Extract the last two parts of the filename
    base=$(basename "$file")
    dir=$(dirname "$file")
    lastdir=$(basename "$dir")

    # Concatenate the last two parts of the filename
    newname="runtime_${lastdir}.txt"

    # Move the file to the destination directory
    cp -v "$file" "${destination}/${newname}"
done
