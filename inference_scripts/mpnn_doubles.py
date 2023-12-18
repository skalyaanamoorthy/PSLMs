# adapted from ProteinMPNN/protein_mpnn_utils.py

import sys
import os
import warnings
import torch
import copy
import argparse
import time

from tqdm import tqdm
from Bio.PDB import PDBParser
import pandas as pd
from filelock import FileLock

import warnings
warnings.filterwarnings('ignore')

def read_csv_safe(file_path):
    """When running ProteinMPNN in parallel, prevents overwriting"""
    lock_path = file_path + ".lock"
    with FileLock(lock_path):
        df = pd.read_csv(file_path)
    return df

def write_csv_safe(df, file_path, mode='w', *args, **kwargs):
    """When running ProteinMPNN in parallel, prevents overwriting"""
    lock_path = file_path + ".lock"
    with FileLock(lock_path):
        df.to_csv(file_path, mode=mode, *args, **kwargs)

def make_tied_positions_for_homomers(pdb_dict_list):
    """Causes identical sequences in a quaternary structure to have likelihoods influenced by each monomer"""
    my_dict = {}
    for result in pdb_dict_list:
        all_chain_list = sorted([item[-1:] for item in list(result) if item[:9]=='seq_chain']) #A, B, C, ...
        tied_positions_list = []
        chain_length = len(result[f"seq_chain_{all_chain_list[0]}"])
        for i in range(1,chain_length+1):
            temp_dict = {}
            for _, chain in enumerate(all_chain_list):
                temp_dict[chain] = [i] #needs to be a list
            tied_positions_list.append(temp_dict)
        my_dict[result['name']] = tied_positions_list
    return my_dict

def main(args):
    df = pd.read_csv(args.db_loc, index_col=0).reset_index()
    d = {'CYS': 'C', 'ASP': 'D', 'SER': 'S', 'GLN': 'Q', 'LYS': 'K', 'ILE': 'I', 'PRO': 'P', 
         'THR': 'T', 'PHE': 'F', 'ASN': 'N', 'GLY': 'G', 'HIS': 'H', 'LEU': 'L', 'ARG': 'R',
         'TRP': 'W', 'ALA': 'A', 'VAL':'V',  'GLU': 'E', 'TYR': 'Y', 'MET': 'M', 'MSE': 'M'}
    
    device = torch.device("cuda:0")
    #v_48_010=version with 48 edges 0.10A noise
    model_name = f"v_48_0{args.noise}" #@param ["v_48_002", "v_48_010", "v_48_020", "v_48_030"]
    backbone_noise = args.bbnoise #0.00 # Standard deviation of Gaussian noise to add to backbone atoms
    hidden_dim = 128
    num_layers = 3
    model_folder_path = os.path.join(args.mpnn_loc, 'vanilla_model_weights')
    if model_folder_path[-1] != '/':
        model_folder_path = model_folder_path + '/'
    checkpoint_path = model_folder_path + f'{model_name}.pt'

    checkpoint = torch.load(checkpoint_path, map_location=device)
    print('Number of edges:', checkpoint['num_edges'])
    noise_level_print = checkpoint['noise_level']
    print(f'Training noise level: {noise_level_print}A')
    model = ProteinMPNN(num_letters=21, node_features=hidden_dim, edge_features=hidden_dim, 
        hidden_dim=hidden_dim, num_encoder_layers=num_layers, num_decoder_layers=num_layers, 
        augment_eps=backbone_noise, k_neighbors=checkpoint['num_edges'])
    model.to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    pdbparser = PDBParser()

    df2 = df.groupby('uid').first()
    logps = pd.DataFrame(index=df2.index,columns=[
        f'mpnn_{args.noise}_{str(args.bbnoise).replace(".", "")}_dir', 
        f'runtime_mpnn_{args.noise}_{str(args.bbnoise).replace(".", "")}_dir'
        ])

    with tqdm(total=len(df2)) as pbar:
        for code, group in df2.groupby('code'):
    
            drop_chains = []

            # get chain sequences and remove chains of only heteroatoms (e.g. DNA)
            pdb_path = os.path.join(group['pdb_file'].head(1).item())
            structure = pdbparser.get_structure(code, pdb_path)
            for c in structure.get_chains():
                seq = [r.resname for r in c]
                seq = ''.join([d[res] if res in d.keys() else 'X' for res in seq])
                if set(seq) == {'X'}:
                    drop_chains.append(c.id)    
            
            homomer=1
            designed_chain_list = []
            fixed_chain_list = []
            target_chain = pdb_path.split('_')[-1].split('.')[0]

            # identify the target chain and sequence, adding it to the designed chains
            for c in structure.get_chains():
                if c.id == target_chain:
                    designed_chain_list.append(target_chain)
                    target_seq = [r.resname for r in c]
                    target_seq = ''.join([d[res] if res in d.keys() else 'X' for res in target_seq])
                    break

            # identify chains with the exact same sequence as the target, adding to designed chains
            for c in structure.get_chains():
                if c.id != target_chain:
                    candidate_seq = [r.resname for r in c]
                    candidate_seq = ''.join([d[res] if res in d.keys() else 'X' for res in candidate_seq])
                    #print(f'target_seq\n{target_seq}')
                    #print(f'candid_seq\n{candidate_seq}')
                    if candidate_seq == target_seq:
                        designed_chain_list.append(c.id)
                        homomer += 1
                    elif c.id not in drop_chains:
                        fixed_chain_list.append(c.id)
            
            #if homomer > 1:
            #    print('Detected identical sequences to target chain, homomer of', homomer)
            
            chain_list = list(set(designed_chain_list + fixed_chain_list))
            
            homomer = bool(homomer-1)

            alphabet = 'ACDEFGHIKLMNPQRSTVWYX'

            chain_id_dict = None
            fixed_positions_dict = None
            pssm_dict = None
            omit_AA_dict = None
            tied_positions_dict = None
            bias_by_res_dict = None

            pdb_dict_list = parse_PDB(pdb_path, input_chain_list=chain_list)
            dataset_valid = StructureDatasetPDB(pdb_dict_list, truncate=None, max_length=100000)
            #print('pdb_dict_list', pdb_dict_list)

            chain_id_dict = {}
            chain_id_dict[pdb_dict_list[0]['name']]= (designed_chain_list, fixed_chain_list)

            #print('designed chains:', designed_chain_list)
            #print('fixed chains:', fixed_chain_list)
            #print('given sequence:')
            #print(group['pdb_ungapped'].head(1).item())
            #print(chain_id_dict)
            
            #for chain in chain_list:
            #    l = len(pdb_dict_list[0][f"seq_chain_{chain}"])
            #    print(f"Length of chain {chain} is {l}")
            #    print(pdb_dict_list[0][f"seq_chain_{chain}"])

            if homomer:
                tied_positions_dict = make_tied_positions_for_homomers(pdb_dict_list)
            else:
                tied_positions_dict = None

            protein = dataset_valid[0]
            batch_clones = [copy.deepcopy(protein)]
            X, S, mask, lengths, chain_M, chain_encoding_all, chain_list_list, visible_list_list, masked_list_list,\
                masked_chain_length_list_list, chain_M_pos, omit_AA_mask, residue_idx, dihedral_mask, \
                tied_pos_list_of_lists_list, pssm_coef, pssm_bias, pssm_log_odds_all, bias_by_res_all, tied_beta = \
                tied_featurize(batch_clones, device, chain_id_dict, fixed_positions_dict, omit_AA_dict,
                    tied_positions_dict, pssm_dict, bias_by_res_dict)

            with torch.no_grad():
                    for uid, row in group.iterrows():
                        pos1 = row['position_1']
                        wt1 = row['wild_type_1']
                        mut1 = row['mutation_1']
                        pos2 = row['position_2']
                        wt2 = row['wild_type_2']
                        mut2 = row['mutation_2']
                        ou = row['offset_up']
                        seq = row['pdb_ungapped']
        
                        start = time.time()
                        
                        oc = int(ou) * (1 if args.dataset == 'fireprot' else 0)  -1 
                        assert seq[pos1 + oc] == mut1
                        assert seq[pos2 + oc] == mut2
                        #print(code, pos, wt, mut, oc)
                        #print('len seq', len(seq))
                        #print('residue_idx', residue_idx)
                        
                        # should force mutant position to decode last
                        decoding_order = torch.zeros_like(chain_M)
                        decoding_order[0][pos1+oc] = 1
                        decoding_order[0][pos2+oc] = 1
                        randn = torch.randn(chain_M.shape, device=X.device)
                        decoding_order = torch.argsort((decoding_order+0.0001)*(torch.abs(randn))) #[numbers will be smaller for places where chain_M = 0.0 and higher for places where chain_M = 1.0]
                        
                        # make the prediction
                        log_probs_native = model.forward(X, S, mask, chain_M*chain_M_pos, residue_idx, chain_encoding_all, randn, use_input_decoding_order=True, decoding_order=decoding_order)

                        # compare log probabilities
                        logps.at[uid, f'mpnn_{args.noise}_{str(args.bbnoise).replace(".", "")}_dir'] = (log_probs_native[0][pos1+oc][alphabet.find(mut1)] -\
                            log_probs_native[0][pos1+oc][alphabet.find(wt1)] + log_probs_native[0][pos2+oc][alphabet.find(mut2)] - log_probs_native[0][pos2+oc][alphabet.find(wt2)]).item()
                        logps.at[uid, f'runtime_mpnn_{args.noise}_{str(args.bbnoise).replace(".", "")}_dir'] = time.time() - start
                        pbar.update(1)

    logps.index.name = 'uid'
    logps.to_csv('tmp_mpnn_preds.csv')
    df = pd.read_csv(args.output, index_col=0)
    if f'mpnn_{args.noise}_{str(args.bbnoise).replace(".", "")}_dir' in df.columns:
        df = df.drop(f'mpnn_{args.noise}_{str(args.bbnoise).replace(".", "")}_dir', axis=1)
    if f'runtime_mpnn_{args.noise}_{str(args.bbnoise).replace(".", "")}_dir' in df.columns:
        df = df.drop(f'runtime_mpnn_{args.noise}_{str(args.bbnoise).replace(".", "")}_dir', axis=1)
    df = df.join(logps)
    write_csv_safe(df, args.output)

    
if __name__ == "__main__":
        parser = argparse.ArgumentParser()
        parser.add_argument('--db_loc', type=str, default='./data/fireprot_mapped.csv')
        parser.add_argument('--output', '-o', type=str, default='./data/fireprot_mapped_preds.csv')
        parser.add_argument('--noise', type=int, default=20)
        parser.add_argument('--bbnoise', type=float, default=0.00)
        parser.add_argument('--mpnn_loc', type=str,required=True)

        args = parser.parse_args()
        sys.path.append(args.mpnn_loc)
        from protein_mpnn_utils import tied_featurize, parse_PDB
        from protein_mpnn_utils import StructureDatasetPDB, ProteinMPNN

        if 'fireprot' in args.db_loc.lower():
            args.dataset = 'fireprot'
        elif 's669' in args.db_loc.lower() or 's461' in args.db_loc.lower():
            args.dataset = 's669'
        else:
            print('Inferred use of user-created database')
            args.dataset = 'custom'

        main(args)
