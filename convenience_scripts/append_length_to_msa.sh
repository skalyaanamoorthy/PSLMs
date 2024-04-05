#!/bin/bash

# Check if the directory argument is provided
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <directory>"
    exit 1
fi

DIRECTORY=$1

# Check if the provided directory exists
if [ ! -d "$DIRECTORY" ]; then
    echo "Directory does not exist: $DIRECTORY"
    exit 1
fi

# Loop through all files in the specified directory
for file in "$DIRECTORY"/*; do
    if [ ! -f "$file" ]; then
        continue  # Skip if not a file
    fi
    
    # Extract the first two lines of the file
    first_line=$(head -n 1 "$file")
    second_line=$(sed -n 2p "$file")
    
    # Calculate the end value based on the length of the second line + 1
    end=$((${#second_line} + 1))
    
    # Check if there's an existing /1-* suffix and replace it, otherwise append
    if [[ $first_line =~ /1-.*$ ]]; then
        # Replace existing suffix
        new_first_line=$(sed "s|/1-.*$|/1-${end}|" <<< "$first_line")
    else
        # Append new suffix
        new_first_line="${first_line}/1-${end}"
    fi
    
    # Use awk to replace the first line in the file
    awk -v new_line="$new_first_line" 'NR==1 {$0=new_line} {print}' "$file" > tmp_file && mv tmp_file "$file"
    
    echo "Updated file: $file"
done

echo "All files processed."

