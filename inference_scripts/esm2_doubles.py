import torch
from torch.cuda.amp import autocast

import argparse
import time

from tqdm import tqdm
import pandas as pd
import numpy as np

from esm import pretrained


def score_sequences(args):

    df = pd.read_csv(args.db_loc)
    df2 = df.groupby('uid').first()
    dataset = args.dataset

    logps = pd.DataFrame(index=df2.index,columns=['esm2_dir', 'runtime_esm2_dir'])

    if True:
        model, alphabet = pretrained.load_model_and_alphabet(f'esm2_t48_15B_UR50D')
        model.eval()
        if torch.cuda.is_available():
            model = model.half().cuda()
            print("Transferred model to GPU")
        with tqdm(total=len(df2)) as pbar:
            for code, group in df2.groupby('code'):
                for uid, row in group.iterrows():
                    with torch.no_grad():
                        #if True:
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
                            data = [
                                ("protein1", sequence),
                            ]
                            batch_labels, batch_strs, batch_tokens = batch_converter(data)

                            batch_tokens_masked = batch_tokens.clone()
                            batch_tokens_masked[0, idx1 + 1] = alphabet.mask_idx
                            batch_tokens_masked[0, idx2 + 1] = alphabet.mask_idx
                            with torch.no_grad():
                                with autocast():
                                    token_probs = torch.log_softmax(
                                        model(batch_tokens_masked.cuda())["logits"], dim=-1
                                    )
                            #print(token_probs.shape)
                            assert sequence[idx1] == wt1
                            assert sequence[idx2] == wt2

                            wt1_encoded, mt1_encoded = alphabet.get_idx(wt1), alphabet.get_idx(mt1)
                            wt2_encoded, mt2_encoded = alphabet.get_idx(wt2), alphabet.get_idx(mt2)
                            score = token_probs[0, 1 + idx1, mt1_encoded] - token_probs[0, 1 + idx1, wt1_encoded] + token_probs[0, 1 + idx2, mt2_encoded] - token_probs[0, 1 + idx2, wt2_encoded]

                            logps.at[uid, f'esm2_dir'] = score.item()
                            logps.at[uid, f'runtime_esm2_dir'] = time.time() - start
                        except:
                            print(code, wt1, pos1, mt1, wt2, pos2, mt2)
                            logps.at[uid, f'esm2_dir'] = np.nan
                            logps.at[uid, f'runtime_esm2_dir'] = np.nan
                        pbar.update(1)
    
    df = pd.read_csv(args.output)
    df = df.set_index('uid')
    df = logps.combine_first(df)
    df.to_csv(args.output)
    #return logps


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
            '--mask_coords', action='store_true', default=False,
            help='whether to mask the coordinates at the mutated position'
    )
    parser.set_defaults(multichain_backbone=False)
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

    score_sequences(args)

if __name__ == '__main__':
    main()
