import os
import time
import torch

import numpy as np
from tqdm import tqdm
import pandas as pd

import esm
import esm.inverse_folding

import os
import torch
import esm
import esm.inverse_folding
from tqdm import tqdm
import gc  # Python's garbage collector

model, alphabet = esm.pretrained.esm_if1_gvp4_t16_142M_UR50()
model = model.eval()
model.to('cuda:0')

def compute_embeddings(fpath_mut, fpath_wt, chain_id, position, mutation):
    with torch.no_grad():  # Disable gradient tracking
        # Load and process mutant structure
        structure_mut = esm.inverse_folding.util.load_structure(fpath_mut, chain_id)
        coords_mut, mut_seq = esm.inverse_folding.util.extract_coords_from_structure(structure_mut)
        rep_mut = esm.inverse_folding.util.get_encoder_output(model, alphabet, coords_mut)

        # Load and process wild-type structure
        structure_wt = esm.inverse_folding.util.load_structure(fpath_wt, chain_id)
        coords_wt, wt_seq = esm.inverse_folding.util.extract_coords_from_structure(structure_wt)
        rep_wt = esm.inverse_folding.util.get_encoder_output(model, alphabet, coords_wt)

        assert mut_seq[position - 1] == mutation

        # Compute the sequence representation
        save_rep = rep_mut[position-1] - rep_wt[position-1]

    # Explicitly free memory
    del structure_mut, coords_mut, rep_mut, structure_wt, coords_wt, rep_wt
    torch.cuda.empty_cache()  # Clear GPU cache
    gc.collect()  # Invoke garbage collector

    return save_rep.cpu().numpy(), mut_seq  # Move tensor to CPU and convert to numpy

# Directory containing predictions
predictions_dir = '../../predictions_test/'

# Data structure to store embeddings and related information
embeddings_data = []

# Loop through all directories and files
for root, dirs, files in tqdm(os.walk(predictions_dir)):
    for file in files:
        if file.startswith('MUT') and file.endswith('_bj1.pdb'):
            fpath_mut = os.path.join(root, file)
            fpath_wt = os.path.join(root, 'WT_bj1.pdb')
            chain_id = root.split('/')[-2].split('_')[-1]
            position = int(file.split('_')[1][:-3])
            mutation = root.split('/')[-1].split('_')[-1][-1]
            #print(fpath_mut, fpath_wt, chain_id, position, mutation)

            # Compute embeddings
            try:
                rep, sequence = compute_embeddings(fpath_mut, fpath_wt, chain_id, position, mutation)
            except AssertionError:
                print('Unexpected mut')
                print(fpath_mut, fpath_wt, chain_id, position, mutation)
            except Exception as e:
                print(e)
                print(fpath_mut, fpath_wt, chain_id, position, mutation)

            # Extract pdb code, position, and mutation from the file path
            pdb_code, position, mutation = root.split('/')[-3:]
            mutation = mutation.split('/')[0]

            # Store the data
            embeddings_data.append({
                'pdb_code': pdb_code,
                'position': position,
                'mutation': mutation,
                'sequence': sequence,
                'embedding': rep
            })

import pickle
with open('complete_embeddings.pkl', 'wb') as file:
    pickle.dump(embeddings_data, file)
