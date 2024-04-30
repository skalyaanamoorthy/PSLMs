import analysis_utils
import pandas as pd
import os
import numpy as np
import re
import copy
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio import SeqIO


### STAGE 1: append Rosetta predictions to inference files ###
# open each of the main dataset inference files
for file1 in ['./data/inference/s669_mapped_preds.csv',
              './data/inference/ssym_mapped_preds.csv',
              './data/inference/k3822_mapped_preds.csv', 
              './data/inference/q3421_mapped_preds.csv',
              './data/inference/fireprot_mapped_preds.csv']:  

    dataset = file1.split('/')[-1].split('_mapped')[0]
    print(dataset)

    db = pd.read_csv(file1, index_col=0)
    # two entries get inexplicably duplicated in korpm datasets
    # but only two inconsequential columns are different
    db = db.loc[~db.index.duplicated(keep='last')]
    db['uid2'] = db['code'] + '_' + db['position'].fillna(-1000000).astype(int).astype(str) + db['mutation']
    
    db2 = pd.read_csv(file1.replace('_preds', '').replace('inference', 'preprocessed'), index_col=0)
    print(len(db), len(db2))
    #assert len(db) == len(db2)

    # add the (organism superfamily) origin column if missing
    if not 'origin' in db.columns and dataset!='fireprot':
        db = db.join(db2['origin'])
    elif not 'origin' in db.columns:
        db['origin'] = list(db2['origin'])

    # replace the cartesian_ddg predictions in case they were updated
    if 'cartesian_ddg_dir' in db.columns:
        db = db.drop(['cartesian_ddg_dir', 'runtime_cartesian_ddg_dir'], axis=1)
    if 'Unnamed: 0' in db.columns:
        db = db.drop(['Unnamed: 0'], axis=1)
    
    db_runtimes = db[[c for c in db.columns if 'runtime' in c or 'uid2' == c]]

    db = db.reset_index().rename({'uid': 'uid_'}, axis=1).rename({'uid2': 'uid'}, axis=1).set_index('uid')
    # extract the runtimes for methods that have it (not currently used)
    db_runtimes = db_runtimes.reset_index().rename({'uid': 'uid_'}, axis=1).rename({'uid2': 'uid'}, axis=1).set_index('uid')
    # assuming you have designated the repo location as the path
    df_cart, df_cart_runtimes = analysis_utils.parse_rosetta_predictions(db, os.path.join('.', 'data', 'rosetta_predictions'), runtime=True)
    
    db_mod = db.copy(deep=True)

    # juggle the indices if needed

    db_mod = db_mod.join(df_cart.astype(float), how='left')
    db_mod = db_mod.join(df_cart_runtimes[['runtime_cartesian_ddg_dir']], how='left')
    db_mod.index.name = 'uid'
    db_mod = db_mod.reset_index().rename({'uid': 'uid2'}, axis=1).rename({'uid_': 'uid'}, axis=1).set_index(['uid', 'uid2'])

    db_mod.to_csv(file1)

    # generate the subset datasets from their full version
    if dataset == 'k3822':
        db_mod_reduced = pd.read_csv('./data/preprocessed/k2369_mapped.csv', index_col=0)
        db_mod = db_mod.loc[db_mod_reduced.index.unique(), :]
        db_mod.to_csv('./data/inference/k2369_mapped_preds.csv')
        print('db_mod_reduced:', len(db_mod))
    elif dataset == 's669':
        db_mod_reduced = pd.read_csv('./data/preprocessed/s461_mapped.csv', index_col=0)
        db_mod = db_mod.loc[db_mod_reduced.index.unique(), :]
        db_mod.to_csv('./data/inference/s461_mapped_preds.csv')
        print('db_mod_reduced:', len(db_mod))


### STAGE 2: Compute all pairs of structures and sequences (for running FATCAT and MMSeqs2) ###

# load the original mapped databases (from preprocessing)
fireprot = pd.read_csv('./data/preprocessed/fireprot_mapped.csv', index_col=0)
#s461 = pd.read_csv('./data/preprocessed/s461_mapped.csv', index_col=0)
s669 = pd.read_csv('./data/preprocessed/s669_mapped.csv', index_col=0)
q3421 = pd.read_csv('./data/preprocessed/q3421_mapped.csv', index_col=0)
ssym = pd.read_csv('./data/preprocessed/ssym_mapped.csv', index_col=0)
k2369 = pd.read_csv('./data/preprocessed/k3822_mapped.csv', index_col=0)
cdna = pd.read_csv('./data/external_datasets/cdna117K_mapped.csv', index_col=0)
rosetta = pd.read_csv('./data/external_datasets/rosetta_mapped.csv', index_col=0)

all_structs = set()
all_seqs = {}

# add the unique structures from each database to a set
for df in [fireprot, s669, q3421, k2369, cdna, rosetta]:
    df['structure'] = df['code'] + '_' + df['chain']
    for s in df['structure'].unique():
        all_structs.add(s)
        seq = open(f'./sequences/fasta_wt/{s}_PDB.fa', 'r').readlines()[-1]
        all_seqs.update({s: seq})

# for ssym, we are only going to use the forward (not reverse/mutant) structures 

ssym['structure'] = ssym['wt_code'] + '_' + ssym['chain']
for s in ssym.loc[ssym['wt_code']==ssym['code']]['structure'].unique():
    all_structs.add(s)
    seq = open(f'./sequences/fasta_wt/{s}_PDB.fa', 'r').readlines()[-1]
    all_seqs.update({s: seq})

sorted_seqs = {key: all_seqs[key] for key in sorted(all_seqs)}

# Convert each tuple to a SeqRecord object, wrapping lines at 80 characters
seq_records = [SeqRecord(Seq(seq), id=name, description="") for name, seq in sorted_seqs.items()]

# Write the sequences to a FASTA file
with open('./data/all_seqs.fasta', 'w') as output_handle:
    SeqIO.write(seq_records, output_handle, 'fasta')

#with open('./data/all_seqs.fasta', 'w') as f: 
    #for name, seq in sorted_seqs.items():
    #    f.write('>'+name+'\n')
    #    f.write(seq+'\n')

# make a separate set that does include the mutant structures as well
all_structs_mutant = copy.deepcopy(all_structs)

# add the mutant structures to this
ssym['structure2'] = ssym['code'] + '_' + ssym['chain']
for s in ssym['structure2'].unique():
    all_structs_mutant.add(s)

# make a sorted list from the set
all_structs_mutant = sorted(list(all_structs_mutant))
all_structs_mutant = [s[:4] for s in all_structs_mutant]
print(len(all_structs_mutant))

# save to a convenience file that shows all PDBs used in this study
with open('./data/all_structs.txt', 'w') as f:
    for struct1 in all_structs_mutant:
       f.write(f'{struct1}\n') 

# make a sorted list of the wild-type structures
all_structs = sorted(list(all_structs))
print(len(all_structs))

# match each structure to each other for FATCAT structural alignment
with open('./data/all_pairs.txt', 'w') as f:    
    for struct1 in all_structs:
        for struct2 in all_structs:
            if struct1 != struct2:
                f.write(f'{struct1} {struct2}\n')





### STAGE 3: Parse results from FATCAT (expected at ./data/homology/structural_homology.aln)
# note: expected that the stage 2 results were generated

# Function to parse a line with PDB codes and chains
def parse_pdb_line(line):
    parts = line.split()
    try:
        code_1, code_2 = parts[1], parts[4]
        chain_1, chain_2 = 'Unknown', 'Unknown'  # Default values
        if '_' in code_1:
            chain_1 = code_1.split('_')[1].replace('.pdb', '')
            code_1 = code_1.split('_')[0]
        if '_' in code_2:
            chain_2 = code_2.split('_')[1].replace('.pdb', '')
            code_2 = code_2.split('_')[0]
        return code_1, chain_1, code_2, chain_2
    except IndexError as e:
        print(f"Error processing line: {line}")
        raise e

# Function to extract values from the line with P-value, Afp-num, etc.
def parse_values_line(line):
    p_value = float(re.search(r'P-value (\S+)', line).group(1))
    afp_num = int(re.search(r'Afp-num (\d+)', line).group(1))
    identity = float(re.search(r'Identity (\S+%)', line).group(1).strip('%'))
    similarity = float(re.search(r'Similarity (\S+%)', line).group(1).strip('%'))
    return p_value, afp_num, identity, similarity

# Read the file and process it
data = []
with open('./data/homology/structural_homology.aln', 'r') as file:
    for line in file:
        if line.startswith('Align'):
            #print(line)
            code_1, chain_1, code_2, chain_2 = parse_pdb_line(line)
        if 'P-value' in line:
            p_value, afp_num, identity, similarity = parse_values_line(line)
            data.append([code_1, chain_1, code_2, chain_2, p_value, afp_num, identity, similarity])

# Create a DataFrame
df = pd.DataFrame(data, columns=['code_1', 'chain_1', 'code_2', 'chain_2', 'P-value', 'Afp-num', 'Identity (%)', 'Similarity (%)'])

# Display the DataFrame
print(df)




### STAGE 4: Determine which datasets have identical mutants

# load the unique structures from each dataset
fireprot = list(pd.read_csv('./data/preprocessed/fireprot_mapped.csv')['code'].unique())
s461 = list(pd.read_csv('./data/preprocessed/s461_mapped.csv')['code'].unique())
s669 = list(pd.read_csv('./data/preprocessed/s669_mapped.csv')['code'].unique())
q3421 = list(pd.read_csv('./data/preprocessed/q3421_mapped.csv')['code'].unique())
ssym = list(pd.read_csv('./data/preprocessed/ssym_mapped.csv')['code'].unique())
k2369 = list(pd.read_csv('./data/preprocessed/k2369_mapped.csv')['code'].unique())
k3822 = list(pd.read_csv('./data/preprocessed/k3822_mapped.csv')['code'].unique())

datasets = ['fireprot', 's461', 's669', 'q3421', 'ssym', 'k2369', 'k3822'] #'s669', 
df['datasets_1'] = [[] for _ in range(len(df))]
df['datasets_2'] = [[] for _ in range(len(df))]

# Iterate over each dataset and update the DataFrame
for name, codes in zip(datasets, [fireprot, s461, s669, q3421, ssym, k2369, k3822]): #s669,
    for i in df.index:
        if df.at[i, 'code_1'] in codes:
            df.at[i, 'datasets_1'].append(name)
        if df.at[i, 'code_2'] in codes:
            df.at[i, 'datasets_2'].append(name)

df['datasets_1'] = df['datasets_1'].astype(str)
df['datasets_2'] = df['datasets_2'].astype(str)
df = df.loc[(df['datasets_1'].astype(str)!='[]') & (df['datasets_2'].astype(str)!='[]')]
#df.sort_values('Similarity (%)', ascending=False).head(50)

df['code_1'] = df['code_1'] + '_' + df['chain_1']
df['code_2'] = df['code_2'] + '_' + df['chain_2']
df




### STAGE 5: Determine which datasets have mutants with significant structural homology, forming clusters
### cluster based on E-value (structural)
from collections import defaultdict

def find_cluster(protein, assigned_clusters, threshold=0.01):
    for cluster in assigned_clusters:
        if all(similarity_matrix.at[protein, member] <= threshold for member in cluster):
            return cluster
    return None

for name, codes in zip(datasets, [fireprot, s461, s669, q3421, ssym, k2369, k3822]):
    df_cur = df.copy(deep=True).loc[df['datasets_1'].astype(str).str.contains(f"\'{name}\'")]
    df_cur = df_cur.loc[df['datasets_2'].astype(str).str.contains(f"\'{name}\'")]
    #df_cur = df_cur.loc[df['Similarity (%)']>50]
    # Create a list of all unique codes
    all_codes = set(df_cur['code_1']).union(set(df_cur['code_2']))

    # Pivot to create a similarity matrix
    similarity_matrix = df_cur.pivot(index='code_1', columns='code_2', values='P-value')

    # Reindex the DataFrame to include all codes in both rows and columns
    similarity_matrix = similarity_matrix.reindex(index=all_codes, columns=all_codes)

    # Fill NaN values with 0 and make the matrix symmetric
    similarity_matrix = similarity_matrix.fillna(0)
    similarity_matrix = similarity_matrix + similarity_matrix.T - similarity_matrix.multiply(similarity_matrix.T.gt(0))

    # Assign proteins to clusters
    clusters = defaultdict(list)
    for protein in similarity_matrix.index:
        cluster = find_cluster(protein, clusters.values())
        if cluster is not None:
            cluster.append(protein)
        else:
            clusters[len(clusters)].append(protein)

    # Convert the clusters dictionary to a list for better readability
    cluster_list = list(clusters.values())

    print(name, "protein clusters based on similarity:")
    print(len(cluster_list))
    print(cluster_list)

    data = pd.read_csv(f'./data/inference/{name}_mapped_preds.csv', index_col=0)
    if name == 's461':
        data['code'] = data.index.str[:4]
        
    data['cluster'] = 0
    i = 0
    for clus in cluster_list:
        i += 1
        for code_ in clus:
            code = code_[:4]
            chain = code_[-1]
            data.loc[(data['code']==code)&(data['chain']==chain), 'cluster'] = i
    data.to_csv(f'./data/inference/{name}_mapped_preds_clusters.csv')





### STAGE 6: Save homology information

# detect sequence overlaps > 25% (for structurally defined region)
for name1 in datasets:
    for name2 in datasets:
        if name1 != name2:
            print(name1, name2)
            ds1 = pd.read_csv(f'./data/inference/{name1}_mapped_preds_clusters.csv', index_col=0)
            ds2 = pd.read_csv(f'./data/inference/{name2}_mapped_preds_clusters.csv', index_col=0)
            overlap = df.loc[((df['datasets_1'].str.contains(name1)) & (df['datasets_2'].str.contains(name2)) | (df['datasets_1'].str.contains(name2)) & (df['datasets_2'].str.contains(name1))) & (df['Identity (%)']>25)]
            overlapping_codes = list(overlap['code_1'].str[:4].unique()) + list(overlap['code_2'].str[:4].unique())
            overlapping_codes += list(set(ds1['code'].unique()).intersection(set(ds2['code'].unique())))
            ds1[f'{name2}_cluster'] = False
            ds2[f'{name1}_cluster'] = False
            ds1.loc[ds1['code'].isin(overlapping_codes), f'{name2}_cluster'] = True
            ds2.loc[ds2['code'].isin(overlapping_codes), f'{name1}_cluster'] = True
            ds1.to_csv(f'./data/inference/{name1}_mapped_preds_clusters.csv')
            ds2.to_csv(f'./data/inference/{name2}_mapped_preds_clusters.csv')

### Collect information from running MMSeqs2 (sequence-only clustering) (expected at ./data/homology/sequence_homology.tsv)

struct_ids = df[['code_1', 'code_2', 'P-value']]
struct_ids.columns = ['source', 'target', 'P-value']
struct_ids = struct_ids.loc[struct_ids['P-value']<0.01]
struct_ids = struct_ids.loc[struct_ids['source']!=struct_ids['target']]

# detect >=25% sequence identity based on MMSeqs2
seq_ids = pd.read_csv('./data/homology/result.m8', sep='\t', header=None)
seq_ids = seq_ids.iloc[:, :3]
seq_ids.columns = ['source', 'target', 'identity']
# immediately only save rows with more than 25% identity
seq_ids = seq_ids.loc[seq_ids['identity']>0.25]
seq_ids = seq_ids.loc[seq_ids['source']!=seq_ids['target']]
#all_codes = set(seq_ids['source']).union(set(seq_ids['target']))

tmp = pd.read_csv(f'./data/inference/k2369_mapped_preds_clusters.csv', index_col=0)
tmp = tmp[[c for c in tmp.columns if not 'overlaps_seq' in c]]
tmp.to_csv(f'./data/inference/k2369_mapped_preds_clusters.csv')

tmp = pd.read_csv(f'./data/inference/k3822_mapped_preds_clusters.csv', index_col=0)
tmp = tmp[[c for c in tmp.columns if not 'overlaps_seq' in c]]
tmp.to_csv(f'./data/inference/k3822_mapped_preds_clusters.csv')

tmp = pd.read_csv(f'./data/inference/q3421_mapped_preds_clusters.csv', index_col=0)
tmp = tmp[[c for c in tmp.columns if not 'overlaps_seq' in c]]
tmp.to_csv(f'./data/inference/q3421_mapped_preds_clusters.csv')

#tmp = pd.read_csv('./data/fireprot_mapped_preds_clusters.csv', index_col=0)
#tmp = tmp[[c for c in tmp.columns if not 'overlaps' in c]]
#tmp.to_csv('./data/fireprot_mapped_preds_clusters.csv')

id_table = pd.DataFrame()
homo_struct_table = pd.DataFrame()
homo_seq_table = pd.DataFrame()

for file1 in ['./data/preprocessed/k2369_mapped.csv', 
              './data/preprocessed/k3822_mapped.csv', 
              './data/external_datasets/rosetta_mapped.csv', 
              './data/external_datasets/cdna117K_mapped.csv', 
              './data/preprocessed/fireprot_mapped.csv', 
              './data/preprocessed/q3421_mapped.csv',
              './data/preprocessed/s461_mapped.csv',
              './data/preprocessed/ssym_mapped.csv'
              ]:
    c = 'code' if ('ssym' not in file1) else 'wt_code'
    # "training" data, e.g. sets that have been used to train models
    df_train = pd.read_csv(file1, index_col=0)
    if 'fireprot_mapped.csv' in file1:
        df_train['position'] = df_train['position'].fillna(-100000).astype(int)
        df_train['uid2'] = df_train['code'] + '_' + df_train['position'].astype(str) + df_train['mutation']
        df_train = df_train.reset_index()
        df_train = df_train.groupby('uid2').first()
    # just extract the codes (structures) and dataset name
    train_codes = set(df_train['code'])
    name1 = file1.split('/')[-1].split('_mapped')[0]
    for file2 in ['./data/preprocessed/k2369_mapped.csv', 
                  './data/preprocessed/k3822_mapped.csv', 
                  './data/external_datasets/rosetta_mapped.csv', 
                  './data/external_datasets/cdna117K_mapped.csv', 
                  './data/preprocessed/fireprot_mapped.csv', 
                  './data/preprocessed/q3421_mapped.csv',
                  './data/preprocessed/s461_mapped.csv',
                  './data/preprocessed/ssym_mapped.csv'
                ]:  
        #c = 'code' if ('ssym' not in file2) else 'wt_code'
        name2 = file2.split('/')[-1].split('_mapped')[0]

        # overlap is a subset of train_codes
        overlap_struct = set()
        overlap_seq = set()
        df_test = pd.read_csv(file2, index_col=0)
        if 'fireprot_mapped.csv' in file2:
            df_test['position'] = df_test['position'].fillna(-100000).astype(int)
            df_test['uid2'] = df_test['code'] + '_' + df_test['position'].astype(str) + df_test['mutation']
            df_test = df_test.reset_index()
            df_test = df_test.groupby('uid2').first()
        #print(len(df_train), len(df_test))

        # this weird syntax just gets the intersection of the two datasets
        id_table.at[name1, name2] = len(df_train.loc[list(set(df_train.index).intersection(set(df_test.index)))])
        test_codes = set(df_test['code'])
        #cc_test = df_test.loc[df_test['code'].isin(overlap_seq_codes)]

        # determine which codes have over 25% similarity based on the "data"
        for code in train_codes:
            # trivial case
            if code in test_codes:
                overlap_struct.add(code)
                overlap_seq.add(code)
            else:
                # get the relevant locations in the data (which pertain to the given train code)
                overlap_struct_df = struct_ids.loc[(struct_ids['source'].str.contains(code))|(struct_ids['target'].str.contains(code))]
                # check whether there are any cases where this code has homology to the test df
                overlap_struct_df = overlap_struct_df.loc[(struct_ids['source'].str[:4].isin(test_codes))|(struct_ids['target'].str[:4].isin(test_codes))]                
                if len(overlap_struct_df) > 0:
                    overlap_struct.add(code)

                # get the relevant locations in the data (which pertain to the given train code)
                overlap_seq_df = seq_ids.loc[(seq_ids['source'].str.contains(code))|(seq_ids['target'].str.contains(code))]
                # check whether there are any cases where this code has homology to the test df
                overlap_seq_df = overlap_seq_df.loc[(seq_ids['source'].str[:4].isin(test_codes))|(seq_ids['target'].str[:4].isin(test_codes))]
                if len(overlap_seq_df) > 0:
                    overlap_seq.add(code)

        homo_struct_table.at[name1, name2] = len(df_train.loc[df_train[c].isin(overlap_struct)])          
        homo_seq_table.at[name1, name2] = len(df_train.loc[df_train[c].isin(overlap_seq)])
        #print(name1, name2, overlap_seq)
        #print(len(df_test.loc[df_test['code'].isin(overlap_seq)]))

        if name1 != name2:
            if 'q3421' in name1:
                name2_ = name2
                df_train[f'overlaps_seq_{name2_}'] = False
                print(name2_)
                print(len(df_train.loc[df_train['code'].isin(overlap_seq), f'overlaps_seq_{name2_}']))
                df_train.loc[df_train['code'].isin(overlap_seq), f'overlaps_seq_{name2_}'] = True
                tmp = pd.read_csv('./data/inference/q3421_mapped_preds_clusters.csv', index_col=0)
                tmp = tmp.join(df_train[[f'overlaps_seq_{name2_}']])
                tmp.to_csv('./data/inference/q3421_mapped_preds_clusters.csv')

            if 'k2369' == name1:
                name2_ = name2
                df_train[f'overlaps_seq_{name2_}'] = False
                df_train.loc[df_train['code'].isin(overlap_seq), f'overlaps_seq_{name2_}'] = True
                tmp = pd.read_csv('./data/inference/k2369_mapped_preds_clusters.csv', index_col=0)
                tmp = tmp.join(df_train[[f'overlaps_seq_{name2_}']])
                tmp.to_csv('./data/inference/k2369_mapped_preds_clusters.csv')

            if 'k3822' == name1:
                name2_ = name2
                df_train[f'overlaps_seq_{name2_}'] = False
                df_train.loc[df_train['code'].isin(overlap_seq), f'overlaps_seq_{name2_}'] = True
                tmp = pd.read_csv('./data/inference/k3822_mapped_preds_clusters.csv', index_col=0)
                tmp = tmp.join(df_train[[f'overlaps_seq_{name2_}']])
                tmp.to_csv('./data/inference/k3822_mapped_preds_clusters.csv')

id_table.to_csv('./data/homology/id_table.csv')
homo_struct_table.index.name = 'Overlapping Entries'
homo_struct_table.columns.name = 'Reference'
homo_struct_table.to_csv('./data/homology/structural_homology_table.csv')
homo_seq_table.index.name = 'Overlapping Entries'
homo_seq_table.columns.name = 'Reference'
homo_seq_table.to_csv('./data/homology/sequence_homology_table.csv')

### STAGE 7: Append features to data for analysis

# open each of the main dataset inference files
for file1 in ['./data/inference/s461_mapped_preds_clusters.csv',
              './data/inference/ssym_mapped_preds_clusters.csv',
              './data/inference/k2369_mapped_preds_clusters.csv', 
              './data/inference/k3822_mapped_preds_clusters.csv',
              './data/inference/q3421_mapped_preds_clusters.csv',
              './data/inference/fireprot_mapped_preds_clusters.csv']:  

    dataset = file1.split('/')[-1].split('_mapped')[0]
    dataset_ = dataset

    print(dataset)
    if dataset == 's461':
    # since s461 is a subset of s669, can just use calcs for s669
        dataset_ = 's669'
    if dataset == 'k2369':
        dataset_ = 'k3822'
 
    db = pd.read_csv(file1).set_index(['uid', 'uid2'])
    
    # load effective number of sequences from separate file (generated by subsample_one.py)
    neff = pd.read_csv(os.path.join('.', 'data', 'features', f'neff_{dataset_}.csv'), header=None, index_col=0)
    neff.index.name = 'code'
    neff.columns = ['neff', 'sequence_length']

    # neff file was generated with different sized alignments, the largest in terms of Neff was used
    neff = neff.groupby(level=0).max()

    db_feats = pd.read_csv(os.path.join('.', 'data', 'features', f'{dataset_}_mapped_feats.csv'))
    db_feats['uid'] = db_feats['code'] + '_' + db_feats['position_orig'].astype(str) + db_feats['mutation']
    db_feats['uid2'] = db_feats['code'] + '_' + db_feats['position'].fillna(-1000000).astype(int).astype(str) + db_feats['mutation']

    db_feats = db_feats.set_index(['uid', 'uid2'])
    db_feats = db_feats[['on_interface', 'entropy', 'conservation', 'column_completeness', 'completeness_score', 'n_seqs', 'structure_length', 'SS', 'code',
                         'features', 'hbonds', 'saltbrs', 'b_factor', 'kdh_wt', 'kdh_mut', 'vol_wt', 'vol_mut', 'chg_wt', 'chg_mut', 'rel_ASA']] #'residue_depth', 'wt_code',

    db_feats['on_interface'] = db_feats['on_interface'].astype(int)
    db_feats['features'] = db_feats['features'].fillna("")
    db_feats['delta_kdh'] = db_feats['kdh_mut'] - db_feats['kdh_wt']
    db_feats['delta_vol'] = db_feats['vol_mut'] - db_feats['vol_wt']
    db_feats['delta_chg'] = db_feats['chg_mut'] - db_feats['chg_wt']
    db_feats['to_proline'] = (db_feats.reset_index('uid2').index.str[-1] == 'P').astype(int)
    db_feats['to_glycine'] = (db_feats.reset_index('uid2').index.str[-1] == 'G').astype(int)
    db_feats['to_alanine'] = (db_feats.reset_index('uid2').index.str[-1] == 'A').astype(int)
    db_feats['from_proline'] = (db_feats.reset_index('uid2').index.str[6] == 'P').astype(int)
    db_feats['from_glycine'] = (db_feats.reset_index('uid2').index.str[6] == 'G').astype(int)
    db_feats['helix'] = db_feats['SS'] == 'H'
    db_feats['bend'] = db_feats['SS'] == 'S'
    db_feats['turn'] = db_feats['SS'] == 'T'
    db_feats['coil'] = db_feats['SS'] == '-'
    db_feats['strand'] = db_feats['SS'] == 'E'
    db_feats['active_site'] = db_feats['features'].str.contains('ACT_SITE')

    db_feats = db_feats.drop(['kdh_wt', 'kdh_mut', 'vol_wt', 'vol_mut', 'chg_wt', 'chg_mut', 'features', 'SS'], axis=1)
    db_feats = db_feats.reset_index().merge(neff['neff'].dropna(), on='code', how='left').drop('code', axis=1).set_index(['uid', 'uid2'])

    db_feats['neff'] = db_feats['neff'].fillna(0)
    db_feats['log_neff'] = np.log(db_feats['neff'])
    #unique_indices = db_feats.groupby('uid')['neff'].idxmax()#.astype(int)
    #db_feats = db_feats.loc[unique_indices].set_index(['uid', 'uid2'])

    for feature in ['on_interface', 'features', 'rel_ASA', 'delta_kdh', 'delta_vol', 'delta_chg', 'to_proline', 'to_glycine', 'to_alanine', 'from_proline', 'from_glycine', 'helix', 'bend', 'turn', 'coil', 'strand', 'active_site']:
        db_feats = db_feats.rename({feature: feature + '_dir'}, axis=1)

    len_db = len(db)

    if dataset != 'fireprot':
        print(len(db_feats), len_db)
        #assert len(db_feats) == len_db
        db_mod = db.join(db_feats, how='left')
    else:
        db_feats = db_feats.drop(['b_factor', 'conservation'], axis=1)
        db_mod = db.join(db_feats.rename({'uid': 'uid_'}, axis=1).rename({'uid2': 'uid'}, axis=1).rename({'uid_': 'uid2'}, axis=1), how='left')

    if dataset == 'ssym':
        # apply the structural clusters assigned to wild-type structures to mutants
        for code in db_mod['wt_code'].unique():
            cluster = db_mod.loc[db_mod['code']==code, 'cluster'].head(1).item()
            db_mod.loc[db_mod['wt_code']==code, 'cluster'] = cluster

        # assign a new direction column to keep track of wild type vs mutant structures
        db_mod['direction'] = 'dir'
        db_mod.loc[db_mod['code']!=db_mod['wt_code'], 'direction'] = 'inv'

        # match the naming convention for predictions made by other authors
        for col in ['KORPM', 'Cartddg', 'FoldX', 'Evo', 'Dyna2', 'PopMs', 'DDGun', 'TNet', 'ACDCNN', 'ddG', 'cluster']:
            db_mod = db_mod.rename({col: col + '_dir'}, axis=1)

        # get two new dataframes which are just the forward and reverse mutations, and then hstack them
        db1 = db_mod.loc[db_mod['code'].str[:4]==db_mod['wt_code']]
        db1 = db1.drop(['code', 'wt_code'], axis=1)
        db2 = db_mod.loc[db_mod['code'].str[:4]!=db_mod['wt_code']]
        db2.loc[:, ['uid']] = db2['wt_code'] + '_' + db2['position_orig'].astype(str) + db2['wild_type']
        db2 = db2.set_index('uid')
        db2 = db2.drop(['code', 'wt_code'], axis=1)
        db2.columns = [c.replace('_dir', '_inv') for c in db2.columns]
        db2 = db2[[c for c in db2.columns if '_inv' in c]]
        db_flat = db1.join(db2)

        # sequence methods are necessarily antisymmetric. This fills in missing or erroneous values
        for col in db_flat.columns:
            if '_dir' in col:
                if any([e in col for e in ['esm2', 'esm1v', 'msa', 'tranception', 'ankh']]) and not 'runtime' in col:
                    db_flat[col.replace('_dir', '_inv')] = -db_flat[col]

        #db_ddgs_2 = db_flat[['ddG_dir', 'ddG_inv']]

        # merge with Ssym+
        ssymp = pd.read_csv(os.path.join('.', 'data', 'external_datasets', 'Ssym+_experimental.csv'))
        ssymp['uid'] = ssymp['Protein'].str[:4].apply(lambda x: x.upper())  + '_' + ssymp['Mut_pdb'].str[1:]
        ssymp = ssymp.set_index('uid')

        test = db_flat.reset_index().set_index('uid').join(ssymp, lsuffix='_plus').reset_index().set_index('uid')

        float_columns = list(test.select_dtypes(include=['float']).columns)
        float_columns.extend(['cluster_dir', 'cluster_inv'])
        db_class = test[float_columns]
        db_class.columns = ['plus_' + c[:-5] if 'plus' in c else c for c in db_class.columns]
        db_class = db_class.drop([c+'_dir' for c in ['ACDCNN', 'ACDC-NN-2str', 'plus_FoldX', 'plus_DDGun', 'PopMs', 'TNet', 'Dyna2']], axis=1)
        db_class = db_class.drop([c+'_inv' for c in ['ACDCNN', 'ACDC-NN-2str', 'plus_FoldX', 'plus_DDGun', 'PopMs', 'TNet', 'Dyna2']], axis=1)
        print(db_class.columns)
        db_stacked = analysis_utils.stack_frames(db_class)

        cols = db_mod.columns
        cols = [c.replace('_dir', '') for c in cols]
        db_mod.columns = cols
        
        join_cols = [c for c in db_mod.columns if not c in db_stacked.columns]
        join_cols.remove('direction')
        db_mod = db_mod.reset_index(drop=True)
        db_mod['uid'] = db_mod['wt_code'] + '_' + db_mod['position_orig'].astype(str) + db_mod['mutation']

        db_stacked = db_stacked.reset_index('direction')
        db_stacked = db_stacked.join(db_mod.set_index('uid')[join_cols])
        
        db_stacked = db_stacked.reset_index()
        db_stacked['uid2'] = db_stacked['wt_code'] + '_' + db_stacked['position'].astype(str) + db_stacked['mutation']
        db_stacked = db_stacked.set_index(['direction', 'uid', 'uid2'])

        #out_loc_flat = f'./data/analysis/{dataset}_flat_analysis.csv'
        #db_stacked.to_csv(out_loc_flat)
        db_mod = db_stacked

    elif dataset == 's461':
        # create and use a third index for matching with the S461 subset
        db_full = db_mod.copy(deep=True)
        db_full['uid3'] = db['code'] + '_' + db['PDB_Mut'].str[1:]
        db_full = db_full.reset_index().set_index('uid3')

        # preprocess S461 to align with S669
        s461 = pd.read_csv(os.path.join('.', 'data', 'external_datasets', 'S461.csv'))
        s461['uid3'] = s461['PDB'] + '_' + s461['MUT_D'].str[2:]
        s461 = s461.set_index('uid3')
        s461['ddG_I'] = -s461['ddG_D']
        s461.columns = [s+'_dir' for s in s461.columns]
        s461 = s461.rename({'ddG_D_dir': 'ddG_dir', 'ddG_I_dir': 'ddG_inv'}, axis=1)

        # merge S669 with S461 (keeping predictions from both for comparison purposes)
        db_mod = s461.join(db_full).reset_index(drop=True).set_index(['uid', 'uid2'])
        #.drop(['PDB_dir', 'MUT_D_dir', 'ddG_dir', 'KORPMD_dir', 'CartddgD_dir',
        #    'FoldXD_dir', 'EvoD_dir', 'Dyna2D_dir', 'PopMsD_dir', 'DDGunD_dir',
        #    'TNetD_dir', 'ACDCNND_dir', 'ddG_inv'], axis=1), how='left')

    if 'ddG' in db_mod.columns and not 'ddG_dir' in db_mod.columns:
        db_mod['ddG_dir'] = db_mod['ddG']
    elif 'ddG_dir' in db_mod.columns and not 'ddG' in db_mod.columns:
        db_mod['ddG'] = db_mod['ddG_dir']
        
    out_loc = f'./data/analysis/{dataset}_analysis.csv'
    db_mod.to_csv(out_loc)