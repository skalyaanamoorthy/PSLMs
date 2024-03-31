import argparse
from Bio import SeqIO
from Bio.Seq import Seq
import sys
import re

def clean_msa(input_file, output_file):
    try:
        sequences = []
        with open(input_file, "r") as handle:
            for record in SeqIO.parse(handle, "fasta"):
                # Check for insertions in the first sequence
                if len(sequences) == 0 and any(c.islower() for c in str(record.seq)):
                    raise ValueError("Insertion found in the first sequence.")
                # Remove lower case letters from sequences
                cleaned_seq = re.sub('[a-z]', '', str(record.seq))  # Use regular expression to remove lowercase
                if not cleaned_seq:  # Check if the sequence is empty after removal
                    raise ValueError(f"Sequence {record.id} is empty after lowercase removal.")
                record.seq = Seq(cleaned_seq)  # Convert back to Seq object
                sequences.append(record)
        
        # Write the cleaned sequences to a new file
        with open(output_file, "w") as output_handle:
            SeqIO.write(sequences, output_handle, "fasta")

    except ValueError as e:
        sys.exit(f"Error: {str(e)}")
    except Exception as e:
        sys.exit(f"An error occurred: {str(e)}")

def main():
    parser = argparse.ArgumentParser(description='Clean lower case insertions from MSA FASTA files.')
    parser.add_argument('input_file', type=str, help='The input FASTA file.')
    parser.add_argument('output_file', type=str, help='The output FASTA file.')

    args = parser.parse_args()

    clean_msa(args.input_file, args.output_file)

if __name__ == "__main__":
    main()
