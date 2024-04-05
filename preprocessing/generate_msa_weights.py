
import argparse
import re

def add_range_to_file(filename):
    # Open the file and read the lines
    with open(filename, 'r+') as file:
        lines = file.readlines()
        
        # Check if the first line already contains a number range
        if not re.search(r'\b\d+-\d+\b', lines[0]):
            # Get the length of the second line if it exists
            if len(lines) > 1:
                num = len(lines[1].rstrip('\n'))  # Remove newline character before counting
                
                # Modify the first line
                lines[0] = lines[0].rstrip('\n') + f"/1-{num}\n"  # Ensure to remove any trailing newline before appending
                
                # Move back to the start of the file to overwrite
                file.seek(0)
                file.writelines(lines)  # Write the modified lines back to the file
                file.truncate()  # Truncate the file to the current position in case new content is shorter than old

def main(args):

    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)

    a_file = args.msa_file
    print(a_file)
    add_range_to_file(a_file)
    print(os.path.basename(a_file).replace('a3m', 'npy'))

    data = data_utils.MSA_processing(
            MSA_location=a_file,
            theta=float(args.theta),
            use_weights=True,
            weights_location=args.output_folder + os.path.basename(a_file).replace('a3m', 'npy')
    )
    
    print(data.Neff)
    print(data.num_sequences)
    print(data.one_hot_encoding.shape)
    print('Completed')

if __name__=='__main__':
    parser = argparse.ArgumentParser(
                    description = 'Calculates weights for MSA sequence '
                                  + 'reweighting according to Tranception '
                                  + 'methods'
                        'downstream prediction')
    parser.add_argument('--msa_file', help='location of the MSA', required=True)
    parser.add_argument('-o', '--output_folder', 
                        help='root of folder to store outputs',
                        default='.')
    parser.add_argument('-t', '--theta', default=0.2)
    parser.add_argument('--tranception_loc', type=str,
                        help='location of the tranception repository',
                        required=True
    )

    args = parser.parse_args()
    import sys
    import os
    sys.path.append(args.tranception_loc + os.sep + 'tranception' + os.sep + 'utils')
    import msa_utils as data_utils
    import pandas as pd
    
    main(args)
