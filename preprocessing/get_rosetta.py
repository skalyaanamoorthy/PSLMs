"""
This script is used to preprocess data from FireProtDB or S669. Since S461 is a
subset of S669, this script will preprocess all data needed for S461. The script 
pulls structure and sequence data from PDB and UniProt. At the end of 
preprocessing, the tabular data originating from the respective databases is 
ready for prediction by other tools.
"""

import os
import argparse
import utils
import glob
import re

import pandas as pd
import numpy as np
from Bio.PDB import PDBParser, PDBIO
from Bio.SeqUtils import seq1
from Bio import pairwise2, SeqIO


def get_rosetta_mapping(code, chain, group, dataset, 
                        SEQUENCES_DIR, ROSETTA_DIR, inverse=False):
    """
    Find additional offsets caused by Rosetta failing to handle any residues.
    Can only be used after generating relaxed structures.
    """

    # inv(erse) refers to the predicted mutant structures for reversions in S669
    colname = 'offset_inv' if inverse else 'offset_rosetta' 
    offsets_rosetta = pd.DataFrame(
        columns=['code', 'chain', 'position', 'mutation', colname]
        )

    for (wt, pos, mut, orig_seq, path), _ in group.groupby(
            ['wild_type', 'position', 'mutation', 'pdb_ungapped',
             'mutant_pdb_file']):

        # indexing is already based on structure, UniProt offset is irrelevant
        if dataset != 'fireprot':
            ou = 0 

        rosparser = PDBParser(QUIET=True)
        
        if inverse:
            # get only the file name, not the path
            fname = path.split('/')[-1]
            full_path = os.path.join(ROSETTA_DIR, 
                f'{code}_{chain}', f'{code}_{pos}{mut}', 'inv_robetta', fname)
            # assume the minimized structure is generated and is located here
            ros = re.sub('.pdb', '_inv_minimized_0001.pdb', full_path)
        else:
            full_path = os.path.join(ROSETTA_DIR, 
                f'{code}_{chain}', f'{code}_{chain}.pdb')
            # assume the minimized structure is generated and is located here
            ros = re.sub('.pdb', '_minimized_0001.pdb', full_path)

        # try to obtain the sequence from the minimized structure
        try:
            structure = rosparser.get_structure(code, ros)
            chains = {c.id:seq1(''.join(residue.resname for residue in c)) 
                    for c in structure.get_chains()}
            ros_seq = chains[chain if not inverse else 'A']
        except Exception as e:
            print(e)
            print(f'Structure or chain not found: {code} {chain} {ros}')
            return None
    
        with open(os.path.join(
                SEQUENCES_DIR, 
                f"fasta_wt/{code}_{chain}_rosetta{'_inv' if inverse else ''}.fa"
            ),'w') as f:
            f.write(f'>{code}_{chain}\n{ros_seq}')

        # reconstruct the wild-type sequence to match the structure
        try:
            pos = int(pos)
        except:
            continue
        if not inverse:
            orig_seq = list(orig_seq)
            orig_seq[int(pos)-1] = wt
            orig_seq = ''.join(orig_seq)

        # align the rosetta sequence to the original pdb sequence to see if 
        # anything was deleted
        aln = pairwise2.align.globalms(
            orig_seq, ros_seq, 5, 0, -2, -0.1)[0] #match, miss, open, extend
        ros_gapped = aln.seqB
        
        #offset_rosetta: offset due to residues dropped by rosetta
        offset_rosetta = ros_gapped[:int(pos)].count('-')

        if offset_rosetta != 0:
            print(f'Rosetta removed {offset_rosetta} residues: {code}')

        # validate the mutation (the inverse uses the mutant structure)
        try:
            if inverse:
                assert ros_seq[int(pos) - 1 - offset_rosetta] == mut
            else:
                assert ros_seq[int(pos) - 1 - offset_rosetta] == wt

        # sometimes rosetta deletes a mutated residue
        except:
            orig_seq_ = list(orig_seq)
            orig_seq_.insert(int(pos) - 1 - offset_rosetta, '[')
            orig_seq_.insert(int(pos) + 1 - offset_rosetta, ']')
            orig_seq_ = ''.join(orig_seq_)
            ros_seq_ = list(ros_seq)
            ros_seq_.insert(int(pos) - 1 - offset_rosetta, '[')
            ros_seq_.insert(int(pos) + 1 - offset_rosetta, ']')
            ros_seq_ = ''.join(ros_seq_)
            print('Wild type position is wrong or doesn\'t exist:', ros,
                code, wt, pos, mut, '\n', orig_seq_, '\n', ros_seq_)#ros_gapped)

        offsets_rosetta = pd.concat([offsets_rosetta, pd.DataFrame({
                    0:{'code':code, 'chain':chain, 'position':pos, 
                       'mutation':mut, colname:offset_rosetta}
                    }).T])
                
    return offsets_rosetta

def main(args):

    output_path = os.path.abspath(args.output_root)
    print('Full path to outputs:', output_path)
    
    SEQUENCES_DIR = os.path.join(output_path, 'sequences')
    # this defines where results from Rosetta will be stored, for organization
    RESULTS_DIR = os.path.join(output_path, 'predictions')
    DATA_DIR = os.path.join(output_path, 'data')

    out = pd.read_csv(args.db_loc)
    out['mutant_pdb_file'] = None
    # at this point, we have all the information about the mapping between 
    # sequence and structure, and we have validated the mutant sequences.
    # Now we just need to prepare the input files for each predictor based on 
    # the format it expects. This usually includes sequence, the location of the 
    # mutation, and the structure file location for structural methods.
    
    # this next section is for compatibility with Rosetta
    hit_rosetta = pd.DataFrame()
    # robetta is referring to the modelled structures for inverse mutations
    if args.inverse:
        hit_robetta = pd.DataFrame()
    print(out.head())

    # iterate back through the output dataframe based on wt structure
    for (code, chain), group in out.groupby(['code', 'chain']):

        group['mutant_pdb_file'] = ''
        print(code)
        # make sure Rosetta's parsing doesn't mess up the alignment
        # happens due to residue deletions in 1AYE, 1C52, 1CTS, 5AZU
        offsets_rosetta = get_rosetta_mapping(
            code, chain, group, args.dataset, SEQUENCES_DIR, RESULTS_DIR
            )
        # would only be None if the structure or chain was not found
        if offsets_rosetta is not None:
            hit_rosetta = pd.concat([hit_rosetta, offsets_rosetta])
        # do the same for inverse structures
        if args.inverse:
            offsets_robetta = get_rosetta_mapping(
                code, chain, group, args.dataset, 
                SEQUENCES_DIR, RESULTS_DIR, inverse=True
                )
            if offsets_robetta is not None:
                hit_robetta = pd.concat([hit_robetta, offsets_robetta])

    # combine Rosetta offsets with all other data
    out = out.merge(
        hit_rosetta, 
        on=['code', 'chain', 'position', 'mutation'], 
        how='left'
        )
    if args.inverse:
        out = out.merge(
            hit_robetta, 
            on=['code', 'chain', 'position', 'mutation'], 
            how='left'
            )
        
    if args.doubles:
        grouped = out.groupby('uid').first()[
            ['code', 'chain', 'wild_type_1', 'position_1', 'mutation_1',
             'wild_type_2', 'position_2', 'mutation_2', 'offset_up', 'offset_rosetta']
            ]
    else:
        grouped = out.groupby('uid').first()[
            ['code', 'chain', 'wild_type', 'position', 'mutation', 'offset_up', 'offset_rosetta']
            ]

    # this file is used by Rosetta to determine where mutations should be made
    grouped['offset_rosetta'] = \
        grouped['offset_rosetta'].fillna(0).astype(int)
    grouped.to_csv(
        os.path.join(output_path, DATA_DIR,
            f'{args.dataset}_unique_muts_offsets.csv')
            )
    print('Wrote unique mutations to', os.path.join(output_path, DATA_DIR,f'{args.dataset}_unique_muts_offsets.csv'))

    # finally, this file is used as an input for a batch job on an HPC
    # these indicies are associated with the unique entries, which can be
    # used to index individual rows of the dataframe for running Rosetta
    # relaxation in parallel
    with open(os.path.join(
        output_path, DATA_DIR, f'{args.dataset}{"_doubles" if args.doubles else ""}'
            f'_rosetta_indices.txt'), 'w') as f:
        inds = ','.join(grouped.reset_index(drop=True).reset_index()\
            .groupby(['code', 'chain']).first()['index'].astype(str))
        print('Rosetta unique mutant indices')
        print(inds)
        f.write(inds)


if __name__=='__main__':
    parser = argparse.ArgumentParser(
                    description = 'Preprocesses data from either s669 or \
                                   FireProtDB for downstream prediction')
    parser.add_argument('--dataset', help='name of database (s669/fireprot), \
                        assuming you are in the root of the repository',
                      default='fireprot')
    parser.add_argument('--db_loc', help='location of the mapped (preprocessed)' 
                        'database', default=None)
    parser.add_argument('-o', '--output_root', 
                        help='root of folder to store outputs',
                        default='.')
    parser.add_argument('-i', '--internal_path', 
                        help='modified path to outputs at inference computer',
                        default=None)
    parser.add_argument('-a', '--alignments',
                        help='folder where redundancy-reduced alignments are',
                        default='./data/msas')
    parser.add_argument('--inverse', action='store_true', 
                        help='whether to get offsets from (Robetta) predicted '
                        +'mutant structures only use when Rosetta relax has '
                        +'been run on Robetta structures')
    parser.add_argument('--verbose', action='store_true',
                        help='whether to save which mutations could not be ' 
                        +'parsed')
    parser.add_argument('--infer_pos', action='store_true',
                        help='whether to use a provided sequence column '
                        +'(aa_seq) which is used to define the positions of '
                        +'mutations')
    parser.add_argument('--doubles', action='store_true',
                        help='Whether to preprocess data for double mutants')
    parser.add_argument('--pdb_bypass', action='store_true',
                        help='Use to skip preprocessing step involving '
                        +'structure, for instance if no structure exists. ' 
                        +'Requires an additional \'pdb_seq\' column.')

    args = parser.parse_args()

    main(args)