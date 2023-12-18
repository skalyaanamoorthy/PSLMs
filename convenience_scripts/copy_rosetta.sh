#!/bin/bash

# Directory to move files to
destination="."

# Iterate over .ddg and .txt files in the predictions directory
for file in predictions/*/*/*.ddg; do
    # Extract the last two parts of the filename
    base=$(basename "$file")
    dir=$(dirname "$file")
    lastdir=$(basename "$dir")
    sec_lastdir=$(basename "$(dirname "$dir")")

    # Concatenate the last two parts of the filename
    newname="${lastdir}.ddg"

    # Move the file to the destination directory
    cp "$file" "${destination}/ddgs/${newname}"
done

for file in predictions/*/*/*.txt; do
    # Extract the last two parts of the filename
    base=$(basename "$file")
    dir=$(dirname "$file")
    lastdir=$(basename "$dir")
    sec_lastdir=$(basename "$(dirname "$dir")")

    # Concatenate the last two parts of the filename
    newname="runtime_${lastdir}.txt"

    # Move the file to the destination directory
    cp "$file" "${destination}/ddgs/${newname}"
done
