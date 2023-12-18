import argparse
import os
import time

import torch
from tqdm import tqdm
import pandas as pd
import numpy as np

from sequence_models.pretrained import load_model_and_alphabet
from sequence_models.pdb_utils import parse_PDB, process_coords
from sequence_models.constants import PROTEIN_ALPHABET as alphabet


def assert_diff_by_one(str1, str2):
    assert len(str1) == len(str2), "Strings are not of same length"
    diff_count = 0
    for i in range(min(len(str1), len(str2))):
        if str1[i] != str2[i]:
            diff_count += 1
    assert diff_count == 1, f"Strings are not different by exactly one character\n{str1}\n{str2}"

def score_backbones(args):
    if args.model not in ['mif', 'mifst']:
        raise ValueError("Valid models ars 'mif' and 'mifst'.")

    print('Loading model...')
    model, collater = load_model_and_alphabet(args.model)

    device = 'cuda:0'
    model = model.eval().to(device)

    print('Loading data...')
    df = pd.read_csv(args.db_loc, index_col=0).reset_index()
    df2 = df.groupby('uid').first()
    dataset = args.dataset

    logps = pd.DataFrame(index=df2.index,columns=[args.model+'_dir', f'runtime_{args.model}_dir'])

    with tqdm(total=len(df2)) as pbar:
        for (code, chain), group in df2.groupby(['code', 'chain']):
            
            pdb_file = group['pdb_file'].head(1).item()
            coords, native_seq, _ = parse_PDB(pdb_file, chain=chain)
            coords = {
                'N': coords[:, 0],
                'CA': coords[:, 1],
                'C': coords[:, 2]
            }

            #print('Native sequence loaded from structure file:')
            #print(native_seq)

            dist, omega, theta, phi = process_coords(coords)

            for uid, row in group.iterrows():
                try:
                    pos = row['position']
                    wt = row['wild_type']
                    mut = row['mutation']
                    #ou = row['offset_up']            
                    try:
                        pos = int(pos)
                    except:
                        print(code, wt, pos, mut)
                        print('Position does not exist in structure!')
                        continue
                    oc = -1 #int(ou) * (1 if dataset == 'fireprot' else 0)  -1
                    
                    seq = row['pdb_ungapped']
                    #print(seq)
                    start = time.time()
                    try:
                        assert_diff_by_one(native_seq, seq)
                    except Exception as e:
                        print(code, wt, pos, mut)
                        print(e)
                    
                    masked_seq=list(native_seq)
                    masked_seq[pos+oc] = '#'
                    masked_seq = ''.join(masked_seq)
                    batch = [[masked_seq, torch.tensor(dist, dtype=torch.float),
                            torch.tensor(omega, dtype=torch.float),
                            torch.tensor(theta, dtype=torch.float), torch.tensor(phi, dtype=torch.float)]]
                    src, nodes, edges, connections, edge_mask = collater(batch)
                    src = src.to(device)
                    nodes = nodes.to(device)
                    edges = edges.to(device)
                    connections = connections.to(device)
                    edge_mask = edge_mask.to(device)
                    logits = model(src, nodes, edges, connections, edge_mask, result='logits')
                    logps.at[uid, args.model+'_dir'] = (logits[0][pos+oc][alphabet.index(mut)] - logits[0][pos+oc][alphabet.index(wt)]).item()
                except Exception as e:
                    print(e)
                    print(code, chain)
                    logps.at[uid, args.model+'_dir'] = np.nan
                logps.at[uid, f'runtime_{args.model}_dir'] = time.time() - start
                pbar.update(1)
        
    logps.index.name = 'uid'
    print(logps.head())
    df = pd.read_csv(args.output, index_col=0)
    print(df.head())
    if args.model+'_dir' in df.columns:
        df = df.drop(args.model+'_dir', axis=1)
    if f'runtime_{args.model}_dir' in df.columns:
        df = df.drop(f'runtime_{args.model}_dir', axis=1)
    df = df.join(logps)
    print(df['mif_dir'].head())
    df.to_csv(args.output)

def score_backbones_inverse(args):
    df = pd.read_csv(args.db_loc, index_col=0).reset_index()
    df2 = df.groupby('uid').first()
    dataset = args.dataset

    model, collater = load_model_and_alphabet(args.model)

    device = 'cuda:0'
    model = model.eval().to(device)
    
    suffix = f'{args.model}_inv'
    logps = pd.DataFrame(index=df2.index,columns=[suffix, f'runtime_{suffix}'])

    with tqdm(total=len(df2)) as pbar:
        for uid, row in df2.iterrows():
            try:
                pdb_file = row['mutant_pdb_file']
                code = row['code']
                chain = 'A' #row['chain'] 
                print(f'Evaluating {code} {chain}')

                coords, mutant_seq, _ = parse_PDB(pdb_file, chain=chain)
                coords = {
                    'N': coords[:, 0],
                    'CA': coords[:, 1],
                    'C': coords[:, 2]
                }

                dist, omega, theta, phi = process_coords(coords)
                #print('Mutant sequence loaded from structure file:')
                #print(mutant_seq)

                pos = row['position']
                wt = row['wild_type']
                mut = row['mutation']
                ou = row['offset_up']
                ro = row['offset_robetta']
                try:
                    pos = int(pos)
                except:
                    print(code, wt, pos, mut)
                    print('Position does not exist in structure!')
                    continue
                oc = -1 #int(ou) * (1 if dataset == 'fireprot' else 0)  -1 -int(ro)

                seq = row['pdb_ungapped'] #row['pdb_ungapped_fill_x']
                #print(seq)
                start = time.time()
                #assert mutant_seq == seq, 'Provided sequence does not match structure'

                masked_seq=list(mutant_seq)
                masked_seq[pos+oc] = '#'
                masked_seq = ''.join(masked_seq)
                batch = [[masked_seq, torch.tensor(dist, dtype=torch.float),
                        torch.tensor(omega, dtype=torch.float),
                        torch.tensor(theta, dtype=torch.float), torch.tensor(phi, dtype=torch.float)]]
                src, nodes, edges, connections, edge_mask = collater(batch)
                src = src.to(device)
                nodes = nodes.to(device)
                edges = edges.to(device)
                connections = connections.to(device)
                edge_mask = edge_mask.to(device)
                logits = model(src, nodes, edges, connections, edge_mask, result='logits')
                logps.at[uid, suffix] = (logits[0][pos+oc][alphabet.index(wt)] - logits[0][pos+oc][alphabet.index(mut)]).item()
            except Exception as e:
                print(e)
                print(code, chain)
                logps.at[uid, suffix] = np.nan
            logps.at[uid, f'runtime_{suffix}'] = time.time() - start
            pbar.update(1)

    df = pd.read_csv(args.output, index_col=0)
    logps.index.name = 'uid'
    df = pd.read_csv(args.output, index_col=0)
    if suffix in df.columns:
        df = df.drop(suffix, axis=1)
    if f'runtime_{suffix}' in df.columns:
        df = df.drop(f'runtime_{suffix}', axis=1)
    df = df.join(logps)
    df.to_csv(args.output)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
            description='Score sequences based on a given structure.'
    )
    parser.add_argument(
            '--model', type=str,
            help='One of mif or mifst',
    )
    parser.add_argument(
            '--db_loc', type=str,
            help='location of the mapped database (fireprot/s669)_mapped.csv',
    )
    parser.add_argument(
            '--output', '-o', type=str,
            help='location where the predictions will be stored, which should\
                be a copy of the mapped database with additional cols'
    )
    parser.add_argument(
            '--inverse', action='store_true', default=False,
            help='use the mutant structure and apply a reversion mutation'
    )

    args = parser.parse_args()

    if 'fireprot' in args.db_loc.lower():
        args.dataset = 'fireprot'
    elif 's669' in args.db_loc.lower() or 's461' in args.db_loc.lower():
        args.dataset = 's669'
    else:
        print('Inferred use of user-created database')
        args.dataset = 'custom'
        
    if args.inverse:
        score_backbones_inverse(args)
    else:
        score_backbones(args)
