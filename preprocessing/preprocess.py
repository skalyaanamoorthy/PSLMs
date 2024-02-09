"""
This script is used to preprocess data from FireProtDB or S669. Since S461 is a
subset of S669, this script will preprocess all data needed for S461. The script 
pulls structure and sequence data from PDB and UniProt. At the end of 
preprocessing, the tabular data originating from the respective databases is 
ready for prediction by other tools.
"""

import os
import argparse
import utils
import glob
import re
import requests

import pandas as pd
import numpy as np

# Some methods want unusual residues mapped to their closest original residue.
# Others want them removed or replaced with an 'X'. This mapping will be used to
# find the appropriate replacement later

d = {'CYS': 'C', 'ASP': 'D', 'SER': 'S', 'GLN': 'Q', 'LYS': 'K', 'ILE': 'I', 
    'PRO': 'P', 'THR': 'T', 'PHE': 'F', 'ASN': 'N', 'GLY': 'G', 'HIS': 'H', 
    'LEU': 'L', 'ARG': 'R', 'TRP': 'W', 'ALA': 'A', 'VAL':'V', 'GLU': 'E', 
    'TYR': 'Y', 'MET': 'M','MSE': 'Z', 'UNK': '9', 'X': 'X'} 

#canonical = {'MSE': 'MET'}

def main(args):

    output_path = os.path.abspath(args.output_root)
    print('Full path to outputs:', output_path)

    # used when running preprocessing on a different computer than inference
    if args.internal_path is not None:
        internal_path = args.internal_path
    else:
        internal_path = output_path
    
    BIO_ASSEMBLIES_DIR = os.path.join(output_path, 'assemblies')
    STRUCTURES_DIR = os.path.join(output_path, 'structures')
    ALIGNMENTS_DIR = os.path.join(output_path, 'alignments')
    SEQUENCES_DIR = os.path.join(output_path, 'sequences')
    WINDOWS_DIR = os.path.join(output_path, 'windows')
    # this defines where results from Rosetta will be stored, for organization
    RESULTS_DIR = os.path.join(output_path, 'results')
    DATA_DIR = os.path.join(output_path, 'data', 'preprocessed')

    # first build a folder structure for organizing inputs and outputs.
    for folder in [BIO_ASSEMBLIES_DIR, STRUCTURES_DIR, ALIGNMENTS_DIR, 
                   SEQUENCES_DIR, WINDOWS_DIR, RESULTS_DIR, DATA_DIR,
                   os.path.join(STRUCTURES_DIR, 'single_chains'),
                   os.path.join(SEQUENCES_DIR, 'fasta_wt'), 
                   os.path.join(SEQUENCES_DIR, 'fasta_mut'),
                   os.path.join(SEQUENCES_DIR, 'fasta_up'),
                   os.path.join(output_path, 'DMS_Tranception'),
                   os.path.join(output_path, 'DMS_MSA')]:
        os.makedirs(folder, exist_ok=True)

    # chains listed in databases do not always correspond to the biological
    # assembly's naming convention.
    wrong_chains = {'1ACB': 'I', '1AON': 'O', '1AYF': 'B', '1ANT': 'L',
                    '1GUY': 'C', '1HK0': 'X', '1HYN': 'P', '1RN1': 'C',
                    '1RTP': '1', '2CI2': 'I', '5CRO': 'O',
                    '1IV7': 'A', '3L15': 'A', '4N6V': '2', '1YU5': 'X',
                    '2I5L': 'X', '2L7F': 'P', '2L7M': 'P', '2LVN': 'C',
                    '1A0N': 'B', '1BSA': 'C', '1BSB': 'C', '1NFI': 'F'}
    
    orig_chains = {'1IV7': 'B', '3L15': 'B', '4N6V': '0'}

    # original database needs to be at this location and can be obtained from
    # the FireProtDB website or from Pancotti et al.
    db = pd.read_csv(args.db_loc)
    dataset = args.dataset
    dataset_outname = args.dataset
    sym = False

    if 'fireprot' in args.db_loc.lower():
        dataset = 'fireprot'
        # some entries in FireProt do not have associated structures
        db = db.dropna(subset=['pdb_id'])
        # get the first PDB from the list (others might be alternate structures)
        db['code'] = db['pdb_id'].apply(lambda x: x.split('|')[0])
        # correct for using the 1LVE structure sequence rather than UniProt
        db.loc[db['code']=='1HTI', 'position'] -= 37
        db.loc[db['code']=='1LVE', 'position'] -= 20
        db.loc[db['code']=='1ZNJ', 'chain'] = 'B'
        #db.loc[~(db['code']=='1ZNJ') & (db['wild_type']=='T'), 'chain'] = 'B'
        db.loc[(db['code']=='1ZNJ') & (db['wild_type']=='T'), 'chain'] = 'A'
    elif 's669' in args.db_loc.lower():
        dataset = 's669'
        db['code'] = db['Protein'].str[0:4]
        db['chain'] = db['Protein'].str[-1]
        db['wild_type'] = db['PDB_Mut'].str[0]
        db['position'] = db['PDB_Mut'].str[1:-1].astype(int)
        db.loc[db['code']=='1IV7', 'position'] = db.loc[db['code']=='1IV7', \
             'Mut_seq'].str[1:-1].astype(int)
        db['mutation'] = db['PDB_Mut'].str[-1]
    elif 'ssym' in args.db_loc.lower():
        sym = True
        dataset = 'ssym'
        db = db.rename({'PDB': 'structureD', 'PDB.1': 'structureR',
                        'ddG_D': 'ddGD', 'ddG_R': 'ddGR',
                        'MUT_D': 'MUTD', 'MUT_R': 'MUTR'}, axis=1)
        db_dir = db[[c for c in db.columns if c[-1] == 'D']]
        db_dir.columns = [c[:-1] for c in db_dir.columns]
        db_rev = db[[c for c in db.columns if c[-1] == 'R']]
        db_rev.columns = [c[:-1] for c in db_rev.columns]
        db = pd.concat([db_dir, db_rev])
        db['code'] = pd.concat([db['structure'][:342], db['structure'][:342]])
        db['code'] = db['code'].str[:4]
        db['wild_type'] = db['MUT'].str[0]
        db['chain'] = db['MUT'].str[1]
        db['position'] = db['MUT'].str[2:-1].astype(int)
        db['mutation'] = db['MUT'].str[-1]
    elif 'q3421' in args.db_loc.lower():
        dataset = 'q3421'
        db = db.rename({'PDB_ID': 'code', 'Chain ': 'chain', 
            'Wildtype': 'wild_type', 'Pos(PDB)': 'position', 
            'mutant ': 'mutation'}, axis=1)
    elif 'korpm' in args.db_loc.lower():
        db.loc[db['code']=='1IV7', 'position'] -= 100
    elif 'proteingym' in args.db_loc.lower():
        db = db.loc[~db['mutant'].str.contains(':')]
        db['code'] = args.use_code if args.use_code else 'PG00'
        db['chain'] = 'A'
        db['wild_type'] = db['mutant'].str[0]
        db['position'] = db['mutant'].str[1:-1].astype(int)
        db['mutation'] = db['mutant'].str[-1]
    else:
        print('Running with a custom user-specified database\n' 
              'This is NOT desired behaviour for reproducing benchmarks')
        print(db)
        db['code'] = db['code'].str.upper()
        for c in ['code', 'chain', 'wild_type', 'position', 'mutation']:
            assert c in db.columns
    
    if sym:
        db['uid'] = db['structure'] + "_" + db['position'].astype(str) + db['mutation']
        grouper = ['code', 'structure', 'chain']
    else:
        db['uid'] = db['code']+'_'+db['position'].astype(str)+db['mutation']
        grouper = ['code', 'code', 'chain']
    hit = pd.DataFrame() # collection of successfully parsed mutations

    # mutations which were not successfully parsed
    miss = pd.DataFrame(columns=['code', 'wt', 'pos', 'mut'])
    missing_msas = []

    # iterate through one PDB code at a time, e.g. all sharing the wt structure
    for (code, struct, chain), group in db.groupby(grouper):
        # chains listed in database do not always correspond to the assembly
        if code in wrong_chains and dataset in ['fireprot', 's669']:
            chain = wrong_chains[code]
        elif code == '1RN1' and not sym:
            chain = wrong_chains[code]
        elif code in ['1IV7', '1NFI', '5CRO']:
            chain = wrong_chains[code]
        if struct in wrong_chains and sym:
            chain = wrong_chains[struct]

        print(code, struct, chain)
        
        print(f"Parsing {struct}_{chain},\
            {len(group['uid'].unique())} unique mutations")
            
        # directory which will be used to organize RESULTS structure-wise
        os.makedirs(
            os.path.join(RESULTS_DIR, f'{struct}_{chain}'), exist_ok=True
            )

        if not args.use_pdb:
            # get the biological assembly, which includes multivmeric structures
            prot_path, prot_file = utils.download_assembly(
                struct, chain, BIO_ASSEMBLIES_DIR
                )

            # get the pdb sequence corresponding to the entry
            mapping_df, is_nmr, multimer = utils.extract_structure(
                struct, chain, d, prot_path, prot_file, STRUCTURES_DIR)

        else:
            prot_file = os.path.basename(args.use_pdb)
            mapping_df, is_nmr, multimer = utils.extract_structure(
                struct, chain, d, args.use_pdb, prot_file,
                STRUCTURES_DIR, compressed=False)            

        pdb_ungapped = ''.join(list(mapping_df.loc[
            mapping_df['repaired_seq'] != '-', 'repaired_seq']))

        if dataset == 'fireprot':
            # in the FireProtDB, the UniProt sequence is provided
            uniprot_seq = group['sequence'].head(1).item()
            with open(
                os.path.join(SEQUENCES_DIR, 'fasta_up', f'{code}_{chain}.fa'), 
            'w') as f:
                f.write(f'>{code}_{chain}\n{uniprot_seq}')

        if args.use_uniprot:
            req2 = f'https://rest.uniprot.org/uniprotkb/{args.use_uniprot}'
            r2 = requests.get(req2).text
            r2 = r2.split('"sequence":{"value":')[-1].split(',')[0].strip('\""')
            uniprot_seq = r2

            with open(
                os.path.join(
                    SEQUENCES_DIR, 'fasta_up', f'{code}_{chain}.fa'), 'w'
                ) as f:
                    f.write(f'>{code}_{chain}\n{uniprot_seq}')
        # assume we need to get the uniprot sequence corresponding to the entry
        # UniProt comes from the wt for Ssym
        elif dataset != 'fireprot' or code == '1HTI':
            uniprot_seq, accession = utils.get_uniprot(
                code, chain, SEQUENCES_DIR
                )

        # align the pdb sequence to the uniprot sequence
        alignment_df, window_start, pdb_ungapped, uniprot_seq = \
            utils.align_sequence_structure(
                code, chain, pdb_ungapped, dataset, mapping_df,
                SEQUENCES_DIR, WINDOWS_DIR, ALIGNMENTS_DIR, 
                group['position'].min(), group['position'].max(), args.indexer,
                uniprot_seq)

        if not args.use_msa:
            # create a convenience link to the alignment file
            # note: this MSA (included in the repo) is already reduced
            # to the context of interest and has no more than 90% identity
            # between sequences and no less than 75% coverage per sequence
            matching_files = glob.glob(
                os.path.join(args.alignments, f'{code}_*.a3m')
                )
            assert len(matching_files) <= 1, \
                f"Expected one file, but found {len(matching_files)}"
            if len(matching_files) == 0:
                exp = "un" if code not in ["1DXX", "1JL9", "1TIT"] else ""
                print(f'Did not find an MSA for {code}. This is {exp}expected')
                missing_msas.append(code)
                new_msa = ''
            else:
                orig_msa = os.path.abspath(matching_files[0])
                new_msa = os.path.join(internal_path, args.alignments, 
                    os.path.basename(orig_msa))

            matching_weights = glob.glob(
                os.path.join('data', 'preprocessed', 'weights', f'{code}_*.npy')
                )
            assert len(matching_weights) <= 1, \
                f"Expected one file, but found {len(matching_weights)}"
            if len(matching_weights) == 0:
                print(f'Did not find sequence weights for MSA for {code}')
            else:
                orig_weights = os.path.abspath(matching_weights[0])
                msa_weights = os.path.join(internal_path, 'data', 
                    'preprocessed', 'weights', os.path.basename(orig_weights))            
        else:
            new_msa = args.use_msa
            msa_weights = os.path.join(internal_path, 'data', 
                    'preprocessed', 'weights', 
                    os.path.basename(new_msa).replace('.a3m', '.npy')) 

        # need the original chain to refer to predicted structures  
        chain_orig = orig_chains[code] \
            if code in orig_chains.keys() else chain

        # create a convenience link to the structure file
        if not args.use_pdb:
            pdb_file = os.path.join(STRUCTURES_DIR, f'{struct}_{chain}.pdb')
        else:
            pdb_file = os.path.join(STRUCTURES_DIR, prot_file)
        pdb_file = re.sub(output_path, internal_path, pdb_file)

        if sym:
            grouper2 = ['uid', 'wild_type', 'position', 'mutation']
        else:
            grouper2 = ['wild_type', 'position', 'mutation']
        
        flag = False
        for name, group2 in group.groupby(grouper2):
            uid = None
            if sym:
                uid, wt, pos, mut = name
            else:
                wt, pos, mut = name
                #pos -= inferred_offset

            # get offsets for interconversion between uniprot and pdb positions
            offset_up, seq_pos, mismatch = utils.get_offsets(
                wt, pos, dataset, alignment_df)          

            # '9' is unknown, but it needed to be distinct from residues
            # such as MSE which are only sometimes unknown
            pu = pdb_ungapped.replace('9', 'X')
            # ESM-IF canonicalizes by default  
            pu = pu.replace('Z', 'M')       

            # second validation that mutants are correct
            if not np.isnan(offset_up):
                if uniprot_seq[seq_pos -1 - offset_up] != wt:
                    p = seq_pos -1 - offset_up
                    print('UniProt mismatch detected.')
                #if dataset != 'fireprot':
                #    assert pu[seq_pos -1] == wt
                if pu[seq_pos -1] != wt:
                     print('Warning! PDB at mutation does not match wt')

                # format the mutant sequence as required by each method
                pu = utils.generate_mutant_sequences(
                    struct, chain, seq_pos, mut, pu, SEQUENCES_DIR
                    )
            else:
                pu = ''

            if not np.isnan(offset_up):
                mapper = mapping_df.loc[
                    mapping_df['sequential_id']==seq_pos].reset_index()
                assert len(mapper) == 1
                mapper = mapper.loc[0, :]
                #assert mapper['repaired_seq'] == wt
                pos_pdb = mapper['author_id']
            else:
                pos_pdb = np.nan

            try:
                new_hit = pd.DataFrame({0: {
                    'uid': (code + '_' + str(pos) + mut
                        if uid is None else uid),
                    'code':code, 'structure':struct, 'chain':chain, 
                    'wild_type':wt, 'position_orig':pos, 'position':seq_pos,
                    'mutation':mut,'pdb_ungapped': pu,
                    'position_pdb': pos_pdb,
                    'uniprot_seq': uniprot_seq,
                    'offset_up':offset_up, 'window_start': window_start, 
                    'is_nmr':is_nmr,'multimer': multimer,
                    'pdb_file': pdb_file, 'msa_file': new_msa, 
                    'msa_weights': msa_weights,
                    'tranception_dms': os.path.join(internal_path,
                    'DMS_Tranception', f'{struct}_{chain}_{dataset}.csv'),  
                    'mismatch': mismatch   
                }}).T          
                
                # ultimately turns into the output table used downstream
                hit = pd.concat([hit, new_hit])

            # there are exceptions in FireProt where the PDB sequence 
            # doesn't match UniProt at mutated positions
            # e.g. due to mutant structures
            except Exception as e:
                print(e, code, wt, pos, mut)
                miss = pd.concat([miss, pd.DataFrame({'0': {
                    'code':code, 'struct':struct,
                    'wt': wt, 'pos': pos, 'mut': mut, 'ou': offset_up  
                }}).T])

    if args.verbose:
        hit.to_csv(
            os.path.join(output_path, DATA_DIR, f'hit_{dataset}.csv')
            )
        miss.to_csv(
            os.path.join(output_path, DATA_DIR, f'miss_{dataset}.csv')
            )

    db = db.drop(
        ['code', 'chain', 'wild_type', 'position', 'mutation']\
            + (['structure'] if sym else []), axis=1
        )

    print(hit)
    # combine all the original mutation information from FireProtDB with hits
    out = db.merge(hit, on=['uid'])
    print(out)

    # check how many mutants could not be processed or validated
    lost = sorted(list(set(db['uid']).difference(set(out['uid']))))
    print('Unique mutants lost from original dataset:', len(lost))
    print('Lost:', lost)

    # at this point, we have all the information about the mapping between 
    # sequence and structure, and we have validated the mutant sequences.
    # Now we just need to prepare the input files for each predictor based on 
    # the format it expects. This usually includes sequence, the location of the 
    # mutation, and the structure file location for structural methods.

    # iterate back through the output dataframe based on wt structure
    for (code, struct, chain), group in out.groupby(grouper):
        # save the data in a method specific directory in the output_root 
        # e.g. DMS_MSA for MSA transformer
        utils.save_formatted_data(
            struct, chain, group, dataset, output_path)

    # change which structure gets used if it is Ssym
    if sym:
        out = out.rename({'code': 'wt_code'}, axis=1)
        out = out.rename({'structure': 'code'}, axis=1)
        
    out = out.set_index('uid')

    # put certain columns first for postprocessing
    aligned_cols = ['code', 'chain', 'wild_type', 'position', 'mutation', 
        'offset_up', 'uniprot_seq', 'msa_file']
    remaining_cols = list(out.columns.drop(aligned_cols))
    out = out[aligned_cols + remaining_cols]

    # this is the main input file for all PSLMs
    outloc = os.path.join(
        output_path, DATA_DIR, f'{dataset_outname}_mapped.csv')
    out.to_csv(outloc)
    print(f'Saved mapped database to {outloc}')

    if dataset_outname == 's669':

        db = out
        db['uid2'] = db['code'] + '_' + \
            db['position'].astype(int).astype(str) + db['mutation'].str[-1]
        db = db.reset_index().set_index(['uid', 'uid2'])

        # create and use a third index for matching with the S461 subset
        db_full = db.copy(deep=True)
        db_full['uid3'] = db['code'] + '_' + db['PDB_Mut'].str[1:]
        db_full = db_full.reset_index().set_index('uid3')

        # preprocess S461 to align with S669
        s461 = pd.read_csv(os.path.join
            (output_path, 'data', 'external_datasets','S461.csv'))
        s461['uid3'] = s461['PDB'] + '_' + s461['MUT_D'].str[2:]
        s461 = s461.set_index('uid3')
        s461['ddG_I'] = -s461['ddG_D']
        s461.columns = [s+'_dir' for s in s461.columns]
        s461 = s461.rename(
            {'ddG_D_dir': 'ddG_dir', 'ddG_I_dir': 'ddG_inv'}, axis=1)

        # merge S669 with S461 
        # (keeping predictions from both for comparison purposes)
        db = s461.join(db_full, how='left').reset_index(
            drop=True).set_index(['uid', 'uid2'])

        db.to_csv(os.path.join(output_path, DATA_DIR, 's461_mapped.csv'))

    grouped = out.reset_index(drop=True).reset_index()
    missing_indices = grouped.loc[grouped['code'].isin(missing_msas)].groupby(
        ['code', 'chain']).first()['index'].astype(str)

    print('Missing MSAs for', missing_msas)
    print('Missing MSA indices:')
    print(' '.join(missing_indices))

    inds = sorted(list(grouped.groupby(['code', 'chain']).first()['index']))
    print(len(inds))
    inds = ','.join([str(s) for s in inds])
    print(inds)

if __name__=='__main__':
    parser = argparse.ArgumentParser(
                    description = 'Preprocesses data to facilitate'
                        'downstream prediction')
    parser.add_argument('--dataset', help='name of database (s669/fireprot), '
                        +'assuming you are in the root of the repository',
                      default='q3421')
    parser.add_argument('--db_loc', help='location of database,'
                        'only specify if you are using a custom DB',
                       default=None)
    parser.add_argument('-o', '--output_root', 
                        help='root of folder to store outputs',
                        default='.')
    parser.add_argument('-i', '--internal_path', 
                        help='modified path to outputs at inference computer',
                        default='.')
    parser.add_argument('-a', '--alignments',
                        help='folder where redundancy-reduced alignments are',
                        default='./data/msas')
    parser.add_argument('-w', '--weights',
                        help='folder where saved sequence reweightings are',
                        default='./data/preprocessed')
    parser.add_argument('--verbose', action='store_true',
                        help='whether to save which mutations could not be ' 
                        +'parsed')
    parser.add_argument('--indexer', help='which type of index specifies '
                        +'the mutation locations', default='author_id')
    parser.add_argument('--use_pdb', help='specify a single PDB file rather '
                        +'than obtaining one automatically')
    parser.add_argument('--use_uniprot', help='specify a Uniprot ID rather '
                        +'than obtaining one automatically')
    parser.add_argument('--use_msa', help='specify an MSA location rather '
                        +'than obtaining one automatically')
    parser.add_argument('--use_code', help='specify a unique code rather '
                        +'than obtaining one automatically')

    args = parser.parse_args()
    if args.dataset.lower() in ['q3421']:
        args.db_loc = './data/external_datasets/Q3421.csv'
    elif args.dataset.lower() in ['fireprot', 'fireprotdb']:
        args.db_loc = './data/external_datasets/fireprotdb_results.csv'
        args.indexer = 'uniprot_id'
    elif args.dataset.lower() in ['s669', 's461']:
        args.db_loc = './data/external_datasets/Data_s669_with_predictions.csv'
        args.dataset = 's669'
    elif args.dataset.lower() in ['ssym']:
        args.db_loc = './data/external_datasets/ssym.csv'
    else:
        print('Inferred use of user-created database. Note: this must '
              'contain columns for code, wild_type, position, mutation. '
              'position must correspond to PDB sequence')
        assert args.dataset != 'fireprot'
        assert args.db_loc is not None

    main(args)