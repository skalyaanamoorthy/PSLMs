import ankh
import torch
import argparse
import time
import pandas as pd
import numpy as np

from tqdm import tqdm

def score_sequences(args):

    df = pd.read_csv(args.db_loc)
    df2 = df.groupby('uid').first()
    dataset = args.dataset

    if args.model == 'ankh_large':
        if args.mode in ['decoder_only', 'both', 'transformer']:
            model, tokenizer = ankh.load_large_model(generation=True)
        elif args.mode in ['encoder_only']:
            model, tokenizer = ankh.load_large_model(generation=False)
        else:
            assert False
        args.model == 'ankh'
    elif args.model == 'ankh_base':
        if args.mode in ['decoder_only', 'both', 'transformer']:
            model, tokenizer = ankh.load_base_model(generation=True)
        elif args.mode in ['encoder_only']:
            model, tokenizer = ankh.load_base_model(generation=False)
        else:
            assert False       
    amino_acids = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
    amino_acid_ids = [tokenizer.convert_tokens_to_ids(aa) for aa in amino_acids]
    model.eval()
    if torch.cuda.is_available():
        model = model.cuda()
        print("Transferred model to GPU")
        device = 'cuda:0'
    else:
        device='cpu'

    mode = args.mode
    logps = pd.DataFrame(index=df2.index,columns=[f'ankh_dir'] + [f'runtime_ankh_dir'])
   
    with tqdm(total=len(df2)) as pbar:
        for code, group in df2.groupby('code'):
            for uid, row in group.iterrows():
                with torch.no_grad():
                    #try:
                    if True:
                        pos = row['position']
                        wt = row['wild_type']
                        mt = row['mutation']
                        ou = row['offset_up']
                        ws = row['window_start']
                        sequence = row['uniprot_seq']#[ws:ws+1022]
                        #if code == '1TIT':
                        #    sequence = row['uniprot_seq'][ws:ws+1022]
                        oc = int(ou) * (0 if dataset == 'fireprot' else -1)  -1 #-ws
                        idx = pos + oc

                        # Convert sequence to list and insert mask token
                        sequence = list(sequence)
                        sequence[idx] = '<extra_id_0>'
                        masked_seq = ''.join(sequence)

                        start = time.time()
                        encoded = tokenizer.encode_plus(masked_seq, add_special_tokens=True, return_tensors='pt')
                        input_ids = encoded['input_ids'].to(device)
                        attention_mask = encoded['attention_mask'].to(device)

                        with torch.no_grad():
                            if mode == 'encoder_only':
                                # For encoder-only mode, do not pass decoder_input_ids
                                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                                logits = outputs.last_hidden_state
                            elif mode == 'decoder_only' or mode in ['both', 'transformer']:
                                # Use the full encoder-decoder model
                                if mode == 'decoder_only':
                                    # Initialize decoder_input_ids with pad tokens and then replace the first token
                                    decoder_input_ids = torch.full_like(input_ids, tokenizer.pad_token_id)
                                    decoder_input_ids[:, 0] = tokenizer.mask_token_id
                                else:
                                    # For both mode, decoder_input_ids mirror input_ids
                                    decoder_input_ids = input_ids.clone()

                                outputs = model(input_ids=input_ids, attention_mask=attention_mask, decoder_input_ids=decoder_input_ids)
                                logits = outputs.logits

                        # Extract logits for the masked position
                        position = (input_ids[0] == tokenizer.convert_tokens_to_ids('<extra_id_0>')).nonzero().item()
                        masked_position_logits = logits[0, position, :]

                        # Softmax only over canonical amino acids
                        masked_position_probabilities = torch.log_softmax(masked_position_logits, dim=0)[amino_acid_ids]

                        score = masked_position_probabilities[amino_acids.index(mt)] - masked_position_probabilities[amino_acids.index(wt)]
                        logps.at[uid, f'{args.model}_{args.mode}_dir'] = score.cpu().item()
                        logps.at[uid, f'runtime_{args.model}_{args.mode}_dir'] = time.time() - start

                        
                    #except Exception as e:
                    #    print(code, wt, pos, mt, e)
                    #    logps.at[uid, f'ankh_{args.mode}_dir'] = np.nan
                    #    logps.at[uid, f'runtime_ankh_{args.mode}_dir'] = np.nan
                    pbar.update(1)

    df = pd.read_csv(args.output)
    df = df.set_index('uid')
    logps.index.name = 'uid'
    df = df.drop([col for col in df.columns if f'{args.model}_{args.mode}' in col], axis=1)
    df = df.join(logps)
    print(df)
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
    parser.set_defaults(multichain_backbone=False)
    parser.add_argument(
            '--inverse', action='store_true', default=False,
            help='use the mutant structure and apply a reversion mutation'
    )
    parser.add_argument(
            '--mode', default='both',
            help='one of encoder_only, decoder_only, or both'
    )
    parser.add_argument(
            '--model', default='ankh_large',
            help='one of ankh_large, ankh_base'
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
