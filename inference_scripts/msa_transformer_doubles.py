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

    logps = pd.DataFrame(index=df2.index,columns=[f'msa_{i+1}_dir' for i in range(5)] + [f'runtime_msa_{i+1}_dir' for i in range(5)])

    model, alphabet = pretrained.load_model_and_alphabet(f'esm_msa1b_t12_100M_UR50S')
    model.eval()
    if torch.cuda.is_available():
        model = model.cuda()
        print("Transferred model to GPU")

    with tqdm(total=len(df2['code'].unique())) as pbar:
        for code, group in df2.groupby('code'):
            print(code)
            sequence = group.head(1)['uniprot_seq'].item()
            orig_msa = group.head(1)['msa_file'].item()
            if args.do_subsampling:
                try:
                    subsample(orig_msa, nseqs=384, reps=5)
                except TypeError:
                    print('Skipping', code, ', no alignment found')
                    continue

            for i in range(5):
                
                try:
                    msa = os.path.join(os.path.dirname(orig_msa), 'subsampled', orig_msa.split('/')[-1].replace('.a3m', f'_reduced_subsampled_{i}.a3m'))
                    print(msa)
                    if not os.path.exists(msa):
                        print(f'Only 1 subsampled alignment for {code}, skipping to next protein')
                        break
                except TypeError:
                    print(f'No alignments found for {code}')
                    break
        
                for uid, row in group.iterrows():
                    with torch.no_grad():
                        try:
                            pos1 = row['position_1']
                            wt1 = row['wild_type_1']
                            mt1 = row['mutation_1']
                            pos2 = row['position_2']
                            wt2 = row['wild_type_2']
                            mt2 = row['mutation_2']
                            ou = row['offset_up']
                            ws = row['window_start']
                            sequence = row['uniprot_seq'][ws:ws+1022]
                            oc = int(ou) * (0 if dataset == 'fireprot' else -1)  -1 -ws
                            idx1 = pos1 + oc
                            idx2 = pos2 + oc

                            start = time.time()

                            batch_converter = alphabet.get_batch_converter()
                            data = [read_msa(msa, 384)]
                            batch_labels, batch_strs, batch_tokens = batch_converter(data)

                            batch_tokens_masked = batch_tokens.clone()
                            #print(sequence)
                            #print(batch_tokens_masked)
                            #print(batch_tokens_masked.shape)
                            batch_tokens_masked[0,0,idx1 + 1] = alphabet.mask_idx
                            batch_tokens_masked[0,0,idx2 + 1] = alphabet.mask_idx
                            with torch.no_grad():
                                token_probs = torch.log_softmax(
                                    model(batch_tokens_masked.cuda())["logits"], dim=-1
                                )
                            token_probs = token_probs[:, 0, :]
                            #print(token_probs.shape)
                            assert sequence[idx1] == wt1
                            assert sequence[idx2] == wt2

                            wt1_encoded, mt1_encoded = alphabet.get_idx(wt1), alphabet.get_idx(mt1)
                            wt2_encoded, mt2_encoded = alphabet.get_idx(wt2), alphabet.get_idx(mt2)
                            score = token_probs[0, 1 + idx1, mt1_encoded] - token_probs[0, 1 + idx1, wt1_encoded] + token_probs[0, 1 + idx2, mt2_encoded] - token_probs[0, 1 + idx2, wt2_encoded]

                            logps.at[uid, f'msa_{i+1}_dir'] = score.item()
                            logps.at[uid, f'runtime_msa_{i+1}_dir'] = time.time() - start
                        except Exception as e:
                            print(e, code, wt1, pos1, mt1, wt2, pos2, mt2)
                            logps.at[uid, f'msa_{i+1}_dir'] = np.nan
                            logps.at[uid, f'runtime_msa_{i+1}_dir'] = np.nan
            if not os.path.exists('msa_transformer_preds_doubles.csv'):
                logps.dropna(how='all')).to_csv('msa_transformer_preds_doubles.csv')
            else:
                tmp = pd.read_csv('msa_transformer_preds_doubles.csv', index_col=0)
                tmp = pd.concat([tmp, logps.dropna(how='all')])
                tmp.to_csv('msa_transformer_preds_doubles.csv')
            pbar.update(1)
                    
    logps['msa_transformer_median_dir'] = logps[[f'msa_{i+1}_dir' for i in range(5)]].median(axis=1)
    logps['msa_transformer_mean_dir'] = logps[[f'msa_{i+1}_dir' for i in range(5)]].mean(axis=1)
    logps['runtime_msa_transformer_median_dir'] = logps[[f'runtime_msa_{i+1}_dir' for i in range(5)]].sum(axis=1)
    logps['runtime_msa_transformer_mean_dir'] = logps['runtime_msa_transformer_median_dir']
    logps.index.name = 'uid'
    logps.to_csv('msa_transformer_preds.csv')
    df = pd.read_csv(args.output, index_col=0)
    df = df.drop([col for col in df.columns if 'msa_' in col and '_dir' in col], axis=1)
    df = df.join(logps)
    df.to_csv(args.output)


def main():
    parser = argparse.ArgumentParser(
            description='Score sequences based on a given structure.'
    )
    parser.add_argument(
            '--db_loc', type=str,
            help='location of the mapped database (file name should contain fireprot or s669)',
    )
    parser.add_argument(
            '--output', '-o', type=str,
            help='location of the database used to store predictions.\
                  Should be a copy of the mapped database with additional cols'
    )
    parser.add_argument(
            '--do_subsampling', action='store_true',
            help='whether to perform subsampling (subsampled MSAs are assumed to have already been generated if this flag is not used')
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
