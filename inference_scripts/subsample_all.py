import torch
import os
import argparse
import time
import string
import itertools
import glob
import re

from esm import Alphabet, FastaBatchedDataset, ProteinBertModel, pretrained, MSATransformer
import pandas as pd
from tqdm import tqdm
from Bio import SeqIO
from typing import List, Tuple
import numpy as np
import collections.abc

collections.Iterable = collections.abc.Iterable

from evcouplings import align


def subsample(infile, nseqs, reps):
        aln = align.Alignment.from_file(open(infile, 'r'), format='a3m')
        aln.set_weights()
        with open(os.path.join(os.path.dirname(os.path.dirname(infile)), 'neff.csv'), 'a') as f:
            f.write(f"{infile.split('/')[-1].split('_')[0]},{sum(aln.weights)},{len(aln[0])}\n")
        outfolder = os.path.join(os.path.dirname(infile), 'subsampled')
        os.makedirs(outfolder, exist_ok=True)
        print('Example weights', aln.weights[:10])

        if len(aln) < nseqs:
            print(len(aln))
            print('Returning all sequences as there were not enough for subsampling!')
            seqs = []
            match_cols = np.where(aln[0]!='-')
            for s in range(len(aln)):
                seqs.append((s, ''.join(aln[s][match_cols])))

            seqs[0] = (0, ''.join(aln[0][match_cols]))

            outfile = os.path.join(outfolder, infile.split('/')[-1].replace('.a3m', f'_reduced_subsampled_0.a3m'))
            align.write_a3m(seqs, open(outfile, 'w'))
            return

        for i in range(reps):
                selected = np.random.choice(range(len(aln)), size=nseqs, p=aln.weights/sum(aln.weights), replace=False)
                selected = np.sort(selected)
                seqs = []
                match_cols = np.where(aln[0]!='-')
                for s in selected:
                    seqs.append((s, ''.join(aln[s][match_cols])))

                seqs[0] = (0, ''.join(aln[0][match_cols]))

                outfile = os.path.join(outfolder, infile.split('/')[-1].replace('.a3m', f'_reduced_subsampled_{i}.a3m'))
                align.write_a3m(seqs, open(outfile, 'w'))


def remove_insertions(sequence: str) -> str:
    """ 
    Removes any insertions into the sequence. Needed to load aligned sequences in an MSA. 
    """
    # This is an efficient way to delete lowercase characters and insertion characters from a string
    deletekeys = dict.fromkeys(string.ascii_lowercase)
    deletekeys["."] = None
    deletekeys["*"] = None
    translation = str.maketrans(deletekeys)
    return sequence.translate(translation)


def read_msa(filename: str, nseq: int) -> List[Tuple[str, str]]:
    """ 
    Reads the first nseq sequences from an MSA file, automatically removes insertions.      
    The input file must be in a3m format (although we use the SeqIO fasta parser)
    for remove_insertions to work properly.
    """
    msa = [(record.description, remove_insertions(str(record.seq)[:1022]))
            for record in itertools.islice(SeqIO.parse(filename, "fasta"), nseq)
          ]
    return msa


def score_sequences(args):

    df = pd.read_csv(args.db_loc, index_col=0).reset_index()
    df2 = df.groupby('uid').first()
    dataset = args.dataset

    with tqdm(total=len(df2['code'].unique())) as pbar:
        for code, group in df2.groupby('code'):
            print(code)
            sequence = group.head(1)['uniprot_seq'].item()
            orig_msa = group.head(1)['msa_file'].item()
            try:
                subsample(orig_msa, nseqs=args.subsamples, reps=5)
            except TypeError:
                print('Skipping', code, ', no alignment found')
                continue


def main():
    parser = argparse.ArgumentParser(
            description='Score sequences based on a given structure.'
    )
    parser.add_argument(
            '--db_loc', type=str,
            help='location of the mapped database (file name should contain fireprot or s669)',
    )
    #parser.add_argument(
    #        '--alignments', '-a', type=str,
    #        help='directory where subsampled alignments should be stored.',
    #        default='./data/msas/'
    #)
    parser.add_argument(
            '--output', '-o', type=str,
            help='location of the database used to store predictions.\
                  Should be a copy of the mapped database with additional cols'
    )
    parser.add_argument(
            '--subsamples', '-n', type=int,
            help='Number of sequence to use in each subsampled alignment'
    )
    args = parser.parse_args()

    if 'fireprot' in args.db_loc.lower():
        args.dataset = 'fireprot'
    elif 's669' in args.db_loc.lower() or 's461' in args.db_loc.lower():
        args.dataset = 's669'
    else:
        print('Inferred use of user-created database')
        args.dataset = 'custom'

    score_sequences(args)

if __name__ == '__main__':
    main()
