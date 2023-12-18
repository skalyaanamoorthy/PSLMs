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


def subsample(infile, nseqs, reps, dataset, neff_only):

        aln = align.Alignment.from_file(open(infile, 'r'), format='a3m')
        aln.set_weights()
        print('Example weights', aln.weights[:10])
        with open(os.path.join(os.path.dirname(os.path.dirname(infile)), f'neff_{dataset}.csv'), 'a') as f:
            f.write(f"{infile.split('/')[-1].split('_')[0]},{sum(aln.weights)},{len(aln[0])}\n")
        outfolder = os.path.join(os.path.dirname(infile), 'subsampled')
        if os.path.exists(os.path.join(outfolder, infile.split('/')[-1].replace('.a3m', f'_reduced_subsampled_0.a3m'))):
            print('Subsampled file already exists! Will not regenerate.')
            return
        if neff_only:
            return

        os.makedirs(outfolder, exist_ok=True)

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


def main():
    parser = argparse.ArgumentParser(
            description='Subsample an alignment for MSA Transformer based on sequence reweighting'
    )
    parser.add_argument(
            '--msa_file', type=str,
            help='location of the MSA to be subsampled',
    )
    parser.add_argument(
            '--nseqs', '-n', type=int, default=384,
            help='Number of sequence to use in each subsampled alignment'
    )
    parser.add_argument(
            '--reps', '-r', type=int, default=5,
            help='How many times to subsample'
    )
    parser.add_argument(
            '--dataset', '-d', type=str, default='',
            help='Name to append to neff.csv file'
    )
    parser.add_argument(
            '--neff_only', action='store_true',
            help='Only output the effective number of sequences'
    )
    args = parser.parse_args()

    subsample(args.msa_file, args.nseqs, args.reps, args.dataset, args.neff_only)

if __name__ == '__main__':
    main()
