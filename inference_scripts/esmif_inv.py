import argparse
import os
import time
import torch

import numpy as np
from tqdm import tqdm
import pandas as pd

import esm
import esm.inverse_folding


def score_singlechain_backbones(model, alphabet, args):
    # load data
    print('Loading data and running in singlechain mode...')
    df = pd.read_csv(args.db_loc, index_col=0).reset_index()
    df2 = df.groupby('uid').first()
    logps = pd.DataFrame(index=df2.index,columns=['esmif_monomer_inv', 'runtime_esmif_monomer_inv'])

    # check if a GPU is available and if so, use it
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)  # move the model to the specified device

    with tqdm(total=len(df2)) as pbar:
        for uid, row in df2.iterrows():
                
                pdb_file = row['mutant_pdb_file']
                code = row['code']
                chain = row['chain']
                chain = 'A'
                print(f'Evaluating {code} {chain}')

                coords, mutant_seq = esm.inverse_folding.util.load_coords(pdb_file, chain)
                #print('Mutant sequence loaded from relaxed structure file:')
                #print(mutant_seq)
                #print('\n')

                ll_mut, _ = esm.inverse_folding.util.score_sequence(
                        model, alphabet, coords, mutant_seq) 
                #print('Native sequence')
                #print(f'Log likelihood: {ll:.2f}')
                #print(f'Perplexity: {np.exp(-ll):.2f}')
                try:
                        pos = row['position']
                        wt = row['wild_type']
                        mut = row['mutation']
                        seq = row['pdb_ungapped']
                        #print('Mutant sequence loaded from database:')
                        assert seq[int(pos) - 1] == mut
                        seq = list(seq)
                        seq[int(pos) - 1] = wt
                        seq = ''.join(seq)
                        #print(f'Native sequence loaded from database:')
                        #print(seq)
                        #print('\n')
                        start = time.time()
                        ll_wt, _ = esm.inverse_folding.util.score_sequence(
                                model, alphabet, coords, seq)
                        logps.at[uid, 'esmif_monomer_inv'] = ll_wt - ll_mut
                except Exception as e:
                        print(e)
                        print(pdb_file, chain)
                        logps.at[uid, 'esmif_monomer_inv'] = np.nan
                logps.at[uid, 'runtime_esmif_monomer_inv'] = time.time() - start
                pbar.update(1)
        
        df = pd.read_csv(args.output, index_col=0)
        logps.index.name = 'uid'
        df = pd.read_csv(args.output, index_col=0)
        if f'esmif_monomer_inv' in df.columns:
            df = df.drop(f'esmif_monomer_inv', axis=1)
        if f'runtime_esmif_monomer_inv' in df.columns:
            df = df.drop(f'runtime_esmif_monomer_inv', axis=1)
        df = df.join(logps)
        df.to_csv(args.output)


def main():
    parser = argparse.ArgumentParser(
            description='Score sequences based on a given structure.'
    )
    parser.add_argument(
            '--db_loc', type=str,
            help='location of the mapped database (fireprot or s669)',
    )
    parser.add_argument(
            '--output', '-o', type=str,
            help='location of the database used to store predictions.\
                  Should be a copy of the mapped database with additional cols'
    )
    args = parser.parse_args()

    model, alphabet = esm.pretrained.esm_if1_gvp4_t16_142M_UR50()
    model = model.eval()
    
    # only have monomer structure for inverse.
    score_singlechain_backbones(model, alphabet, args)

if __name__ == '__main__':
    main()
