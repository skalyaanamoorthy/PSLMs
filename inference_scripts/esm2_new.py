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

    logps = pd.DataFrame(index=df2.index, columns=['esm2_dir', 'runtime_esm2_dir'])

    # Replace esm's load function with Hugging Face's
    # The tokenizer is included in case you need it later for tokenization
    #tokenizer = AutoTokenizer.from_pretrained('esm2_t48_15B_UR50D')
    #model = AutoModel.from_pretrained('esm2_t48_15B_UR50D')
    model, alphabet = pretrained.load_model_and_alphabet(f'esm2_t48_15B_UR50D')

    model.eval()
    with autocast():
        if torch.cuda.is_available():
        # Use DataParallel for multi-GPU. Assuming you have 2 GPUs with ids: 0 and 1.
        #model = torch.nn.DataParallel(model, device_ids=[0, 1])
            model = model.half().cuda()
            print("Transferred model to GPUs")

    with tqdm(total=len(df2)) as pbar:
        for code, group in df2.groupby('code'):
            for uid, row in group.iterrows():
                with torch.no_grad():
                    try:
                        pos = row['position']
                        wt = row['wild_type']
                        mt = row['mutation']
                        ou = row['offset_up']
                        ws = row['window_start']
                        sequence = row['uniprot_seq'][ws:ws+1022]
                        oc = int(ou) * (0 if dataset == 'fireprot' else -1)  -1 -ws
                        idx = pos + oc

                        start = time.time()

                        batch_converter = alphabet.get_batch_converter()
                        data = [
                            ("protein1", sequence),
                        ]
                        batch_labels, batch_strs, batch_tokens = batch_converter(data)

                        batch_tokens_masked = batch_tokens.clone()
                        batch_tokens_masked[0, idx + 1] = alphabet.mask_idx
                        with torch.no_grad():
                            with autocast():
                                token_probs = torch.log_softmax(
                                    model(batch_tokens_masked.cuda())["logits"], dim=-1
                                )
                        #print(token_probs.shape)
                        assert sequence[idx] == wt

                        wt_encoded, mt_encoded = alphabet.get_idx(wt), alphabet.get_idx(mt)
                        score = token_probs[0, 1 + idx, mt_encoded] - token_probs[0, 1 + idx, wt_encoded]

                        logps.at[uid, f'esm2_dir'] = score.item()
                        logps.at[uid, f'runtime_esm2_dir'] = time.time() - start
                    except:
                        print(code, wt, pos, mt)
                        logps.at[uid, f'esm2_dir'] = np.nan
                        logps.at[uid, f'runtime_esm2_dir'] = np.nan
                    pbar.update(1)
    
    df = pd.read_csv(args.output, index_col=0)
    logps.index.name = 'uid'
    df = pd.read_csv(args.output, index_col=0)
    if f'esm2_dir' in df.columns:
        df = df.drop(f'esm2_dir', axis=1)
    if f'runtime_esm2_dir' in df.columns:
        df = df.drop(f'runtime_esm2_dir', axis=1)
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
