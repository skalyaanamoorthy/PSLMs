# adapted from Tranception code:
# score_tranception_proteingym.py
import os
import argparse
import json
import time
import pandas as pd
import gc
from tqdm import tqdm

import torch
import sys


def score_sequence(args):
    """
    Main script to score sets of mutated protein sequences (substitutions or indels) with Tranception.
    """

    df = pd.read_csv(args.db_loc, index_col=0)
    df2 = df.reset_index().groupby('uid').first()
    trance = pd.DataFrame(columns=[f'tranception_dir', f'runtime_tranception_dir'])

    model_name = args.checkpoint.split("/")[-1]
    
    tokenizer = PreTrainedTokenizerFast(tokenizer_file=os.path.join(args.tranception_loc, "tranception/utils/tokenizers/Basic_tokenizer"),
                                                unk_token="[UNK]",
                                                sep_token="[SEP]",
                                                pad_token="[PAD]",
                                                cls_token="[CLS]",
                                                mask_token="[MASK]"
                                            )
    with tqdm(total=len(df2.groupby('code'))) as pbar:
        for code, group in df2.groupby('code'):
            start = time.time()
            try:
                chain = group['chain'].head(1).item()
                if args.low_mem and code in ['1CEY', '1PX0', '2CHF', '6TQ3']:
                    print(f'Skipping {code} which requires >64 GB VRAM')
                    continue
                print(f'Evaluating {code} {chain}')

                target_seq=group['uniprot_seq'].head(1).item()
                DMS_file_name=group['tranception_dms'].head(1).item()
                #print(DMS_file_name)

                DMS_id = DMS_file_name.split("/")[-1].split(".")[0]

                MSA_data_file = group.head(1)['msa_file'].item()
                MSA_start = 0 #args.MSA_start - 1 # MSA_start based on 1-indexing
                MSA_end = len(target_seq) # + 1 #args.MSA_end

                config = json.load(open(args.checkpoint+os.sep+'config.json'))
                config = tranception.config.TranceptionConfig(**config)
                config.attention_mode="tranception"
                config.position_embedding="grouped_alibi"
                config.tokenizer = tokenizer
                config.scoring_window = args.scoring_window
                config.retrieval_aggregation_mode = "aggregate_substitution"
                config.MSA_filename=MSA_data_file
                config.full_protein_length=len(target_seq)
                config.retrieval_inference_weight=args.retrieval_inference_weight
                config.MSA_start = MSA_start
                config.MSA_end = MSA_end
                    
                if args.model_framework=="pytorch":
                    model = tranception.model_pytorch.TranceptionLMHeadModel.from_pretrained(pretrained_model_name_or_path=args.checkpoint,config=config)
                    if torch.cuda.is_available():
                        model.cuda()
                model.eval()
               
                DMS_data = pd.read_csv(DMS_file_name, low_memory=True)
                all_scores = model.score_mutants(
                                                DMS_data=DMS_data, 
                                                target_seq=target_seq, 
                                                scoring_mirror=True, 
                                                batch_size_inference=args.batch_size_inference,  
                                                num_workers=args.num_workers, 
                                                indel_mode=False
                                                )
            except Exception as e:
                print('Skipping', code, chain)
                print(e)
                continue
                
            #m = pd.read_csv(os.path.join('DMS_tranception', f'{code}_{chain}_{"s669" if "s669" in args.db_loc else "fireprot"}.csv'))
            m = pd.read_csv(DMS_file_name, low_memory=False)
            new = m.set_index('mutated_sequence').join(all_scores.set_index('mutated_sequence'))
            new['uid2'] = code + '_' + new['mutant'].str[1:]
            new = new[['uid2', 'avg_score']].rename({'avg_score': f'tranception_dir'}, axis=1)
            new = new.set_index('uid2').astype(float)
            new[f'runtime_tranception_dir'] = time.time() - start
            trance = pd.concat([trance, new])
            pbar.update(1)
        
    logps = trance
    logps.index.name = 'uid'
    df = pd.read_csv(args.output, index_col=0)
    if 'tranception_dir' in df.columns:
        df = df.drop('tranception_dir', axis=1)
    if f'runtime_tranception_dir' in df.columns:
        df = df.drop(f'runtime_tranception_dir', axis=1)
    if 's669' in args.db_loc:
        #trance.to_csv('./trance_s669.csv')
        df = df.reset_index()
        df['uid2'] = df['code'] + '_' + (df['position'].astype(int) - df['offset_up'].astype(int)).astype(str) + df['mutation'].str[-1]
        df = df.set_index('uid2')
        df = df.join(logps)
        df = df.reset_index(drop=True)
        df = df.set_index('uid')
    elif 'fireprot' in args.db_loc:
        #trance.to_csv('./trance_fp.csv')
        df = df.join(logps)
    df.to_csv(args.output)


if __name__ == '__main__':
    # original args
    parser = argparse.ArgumentParser(description='Tranception scoring')
    parser.add_argument('--checkpoint', type=str, help='Path of Tranception model checkpoint', required=True)
    parser.add_argument('--model_framework', default='pytorch', type=str, help='Underlying framework [pytorch|JAX]')
    parser.add_argument('--batch_size_inference', default=20, type=int, help='Batch size for inference')
    parser.add_argument('--output_scores_folder', default='./', type=str, help='Name of folder to write model scores to')
    parser.add_argument('--scoring_window', default="optimal", type=str, help='Sequence window selection mode (when sequence length longer than model context size)')
    parser.add_argument('--retrieval_inference_weight', default=0.6, type=float, help='Coefficient (alpha) used when aggregating autoregressive transformer and retrieval')
    parser.add_argument('--num_workers', default=10, type=int, help='Number of workers for model scoring data loader')
    
    # added args
    parser.add_argument(
            '--db_loc', type=str,
            help='location of the mapped database (file name should contain fireprot or s669)',
    )
    #parser.add_argument(
    #    '--alignments', '-a', type=str, default='./data/msas/',
    #    help='directory where alignments are stored. There should be only\
    #        one file matching the pattern CODE_*.a3m where CODE is the pdb\
    #        id of the structure the alignment is based upon'
    #)
    parser.add_argument(
            '--output', type=str,
            help='location of the stored predictions database',
    )
    parser.add_argument(
            '--tranception_loc', type=str,
            help='location of the tranception repository',
            required=True
    )
    parser.add_argument(
        '--low_mem', action='store_true', 
        help='Skip certain proteins which have large MSAs and require more \
            than 64GB RAM'
    )

    args = parser.parse_args()

    if not os.path.exists(args.tranception_loc):
        print('Invalid Tranception directory! Please download the GitHub repo \
            and ensure the full path to the repository is provided')

    sys.path.append(args.tranception_loc)
    from transformers import PreTrainedTokenizerFast
    import tranception
    from tranception import config, model_pytorch

    score_sequence(args)
