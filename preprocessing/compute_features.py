import os
import argparse
import numpy as np
import pandas as pd
import requests
import urllib

from tqdm import tqdm
from quantiprot.metrics import aaindex
from Bio import ExPASy, SwissProt, AlignIO, pairwise2
from Bio.PDB import *
from Bio.PDB.DSSP import DSSP, dssp_dict_from_pdb_file
from scipy.stats import entropy
from Bio.SeqUtils import seq1

path = '../'

d = {'CYS': 'C', 'ASP': 'D', 'SER': 'S', 'GLN': 'Q', 'LYS': 'K',
    'ILE': 'I', 'PRO': 'P', 'THR': 'T', 'PHE': 'F', 'ASN': 'N', 
    'GLY': 'G', 'HIS': 'H', 'LEU': 'L', 'ARG': 'R', 'TRP': 'W', 
    'ALA': 'A', 'VAL':'V', 'GLU': 'E', 'TYR': 'Y', 'MET': 'M',
    'MSE': 'Z', 'UNK': '9', 'X': 'X'} 


def get_sprot_raw_with_retry(uniprot_id, max_retries=5, delay=5):
    for i in range(max_retries):
        try:
            handle = ExPASy.get_sprot_raw(uniprot_id)
            return handle
        except urllib.request.HTTPError as err:
            if err.code == 500:  # If the error is an Internal Server Error
                print(f"Attempt {i+1} of {max_retries} failed with error 500. \
                    Retrying in {delay} seconds...")
                time.sleep(delay)  # Wait before retrying
            else:
                raise  # If the error is something else, raise it
    raise Exception("Max retries exceeded with HTTP 500 errors.")


def run_alistat(alignment_file, alistat_loc, output_loc):
    print(f'Running AliStat on {alignment_file}')
    print(output_loc)
    os.system(
        f'{os.path.join(alistat_loc, "alistat")} {alignment_file} 6 -t 2 \
            -o {output_loc}'
        )


def get_column_completeness(filename, column):
    """
    Derived using AliStat: https://github.com/thomaskf/AliStat
    Using (for each MSA): alistat $msa_name.a3m 6 -t 1,2,3 although
    technically only -t 2 (Table_2.csv) is needed for this function
    """
    df = pd.read_csv(filename)
    return df.at[column, 'Cc']


def get_alignment_summary(filename):
    """
    Derived using AliStat: https://github.com/thomaskf/AliStat
    Using (for each MSA): alistat $msa_name.a3m 6 -t 1,2,3 from Summary.txt
    """
    with open(filename, 'r') as f:
        for i, line in enumerate(f.readlines()):
            if i == 3:
                n_seqs = int(line[64:].strip())
            elif i == 6:
                completeness = float(line[64:].strip())
    return completeness, n_seqs


def get_conservation(alignment_file, residue_index, wild_type):
    """
    Obtains the entropy and (related) percent identity at a given position in
    the provided MSA
    """
    alignment = AlignIO.read(alignment_file, 'fasta')

    # obtain only the relevant column
    residues = [seq[residue_index] for seq in alignment]
    # the first sequence at this position is necessarily the wild-type
    #assert residues[0] == wild_type, print('Unexpected residue in alignment')

    # compute the entropy based on the amounts of each residue type
    count = {aa: residues.count(aa) for aa in set(residues)}
    probability = [count[aa]/len(residues) for aa in count]
    entropy_score = entropy(probability)

    # compute percent residue identity, a.k.a. percent conservation of wild-type
    pri_score = count[wild_type] / len(residues) * 100
    return entropy_score, pri_score


def get_residue_features(uniprot_id, chain_id, residue_id):
    """
    Uses UniProt ID to get residue features from SwissProt
    Returns a list of features
    """
    try:
        handle = get_sprot_raw_with_retry(uniprot_id)
        record = SwissProt.read(handle)
    except Exception as e:
        print(e)
        return []
    
    features = []
    # iterate over all features for the given UniProt ID
    for feature in record.features:
        site_start = feature.location.start
        site_end = feature.location.end
        try:
            # only use this feature if the residue_id occurs within its range
            if residue_id >= site_start and residue_id <= site_end and \
                feature.type not in ['CHAIN', 'DOMAIN', 'REGION']: 
                features.append(feature.type)
        except Exception as e:
            print('Exception in get_residue_features:', e)
    return features


def get_interface_residues(model, target_chain_id, distance_threshold=10.0):
    """
    Determine which residues of a given chain are part of an interface based on
    a distance threshold between CAs of each residue on a target chain and any 
    residue on another chain
    """

    # Get the target chain and the list of other chains
    target_chain = None
    other_chains = []
    for chain in model.get_chains():
        if chain.get_id() == target_chain_id:
            target_chain = chain
        else:
            other_chains.append(chain)

    # Check if the target chain exists
    if target_chain is None:
        raise ValueError(f"Chain {target_chain_id} not found")

    # Store the interface residues of the target chain
    interface_residues = []
    # Iterate over all residues, (other) chains, and other chains' residues
    for other_chain in other_chains:
        for target_residue in target_chain.get_residues():
            for other_residue in other_chain.get_residues():
                try:
                    target_atom = target_residue['CA']
                    other_atom = other_residue['CA']
                except KeyError:
                    continue
                # distance determination
                distance = np.linalg.norm(
                    np.array(target_atom.coord) - np.array(other_atom.coord)
                    )
                # only count as interface if within the threshold
                if distance < distance_threshold:
                    interface_residues.append(target_residue.get_id()[1])
                    break
    return interface_residues


def get_residue_accessibility(model, filename, target_chain):
    """
    Run DSSP to determine the absolute surface area of each residue
    """
    dssp_dict = dict(DSSP(model, filename, dssp='mkdssp'))
    # get the target chain only
    df = pd.DataFrame(dssp_dict).T.loc[target_chain, :]
    df.index = pd.Series(df.index).apply(lambda x: x[1])
    df = df.rename({1:'wild_type', 2: 'SS', 3: 'rel_ASA'}, axis=1)
    return df


def get_packing_density(model):
    """Not used (does not work)"""
    hse = HSExposureCA(model)
    asa_sum = sum(hse.get_exposed_total_rel())
    volume = hse.get_volume()
    print(volume)
    packing_density = asa_sum / volume
    return packing_density, asa_sum, volume


def get_h_angle(c_atom_coord, o_atom_coord, n_atom_coord):
    """
    Used in the determination of hydrogen bonds
    """
    a = np.array(c_atom_coord)
    b = np.array(o_atom_coord)
    c = np.array(n_atom_coord)
    
    ba = a - b
    bc = c - b
    
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(cosine_angle)
    
    return np.degrees(angle)


def get_residue_interactions(model, target_chain_id, residue_id):
    """
    Uses previous functions and internal logic to determine if a target residue
    is involved in hydrogen bonding or salt bridges (and also extracts the beta-
    factor)
    """

    # get the target chain object
    for chain in model.get_chains():
        if chain.get_id() == target_chain_id:
            target_chain = chain

    # get the target residue object
    target_residue = None
    for residue in target_chain.get_residues():
        if residue.get_id()[1] == residue_id:
            target_residue = residue
    if target_residue is None:
        return None, None, None, None
    
    h_bonds = 0
    salt_bridges = 0
    b_factor = []
    h_bond_identities = []

    # iterate through all atoms of the target residue
    for atom in target_residue.get_atoms():
        # extract beta factor
        b_factor.append(atom.get_bfactor())
        try:
            # detect all atom neighbors within 3.5 Angstrom
            neighbors = NeighborSearch(Selection.unfold_entities(
                model[target_chain_id], 'A'
                ), bucket_size=10).search(atom.coord, radius=3.5, level='A')
        except Exception as e:
            print('Exception in NeighborSearch,', e)
            continue
        # iterate through neighboring atoms within 3.5 Angstrom
        for neighbor in neighbors:
            # exclude niegbors from the same residue
            if neighbor.parent != atom.parent:
                # calculate the distance between the two
                dist = np.linalg.norm(atom.coord - neighbor.coord)
                try:
                    # hydrogen bonding logic: based on C/O/N adjacency
                    # hydrogens are implicit
                    h_angle = 0
                    if 'N' in atom.name and 'O' in neighbor.name:
                        # we want to find the C attached to O in neighbor
                        # so look for covalently bonded atoms
                        local = NeighborSearch(
                            list(neighbor.get_parent().get_atoms())
                            ).search(neighbor.coord, radius=1.5, level='A')
                        for l in local:
                            if 'C' in l.name:
                                target_atom = l
                        if target_atom is None:
                            print('did not find a closeby atom')
                            continue
                        # use the previous function to determine the angle
                        # and ultimately whether this matches h-bond constraints
                        h_angle = get_h_angle(
                            target_atom.coord, neighbor.coord, atom.coord
                            )
                    # same as above but with ids swapped
                    elif 'O' in atom.name and 'N' in neighbor.name:
                        # we want to find the C attached to O in atom
                        local = NeighborSearch(
                            list(atom.get_parent().get_atoms())
                            ).search(atom.coord, radius=1.5, level='A')
                        target_atom = None
                        for l in local:
                            if 'C' in l.name:
                                target_atom = l
                        if target_atom is None:
                            print('did not find a closeby atom')
                            continue
                        h_angle = get_h_angle(target_atom.coord, atom.coord, neighbor.coord)
                    # case where we have a sidechain O (e.g. OD, OE)
                    elif 'O' in atom.name and 'O' in neighbor.name and \
                        (atom.name != 'O' or neighbor.name != 'O'):
                        # neighbor is sidechain
                        if atom.name == 'O':
                            h_angle = get_h_angle(
                                atom.get_parent()['C'].coord, 
                                atom.coord, neighbor.coord
                                )
                        # atom (target) is sidechain O
                        elif neighbor.name == 'O':
                            h_angle = get_h_angle(
                                neighbor.get_parent()['C'].coord, 
                                atom.coord, neighbor.coord
                                )
                        # both are sidechains
                        else:
                            local = NeighborSearch(
                                list(neighbor.get_parent().get_atoms())
                                ).search(atom.coord, radius=1.5, level='A')
                            target_atom = None
                            for l in local:
                                if 'C' in l.name:
                                    target_atom = l
                            if target_atom is None:
                                print('did not find a closeby atom')
                                continue
                            h_angle = get_h_angle(target_atom.coord, 
                                atom.coord, neighbor.coord
                                )
                    else:
                        continue
                    # criteria for hydrogen bonding
                    if 120 < h_angle < 180:
                        h_bonds += 1
                        # for debugging
                        h_bond_identities.append((
                            f'{atom.name}-{atom.get_parent().id[1]}-H',
                            f'-{neighbor.name}-{neighbor.get_parent().id[1]}'
                        ))

                    # salt-bridge criteria
                    if atom.name in ['NH1', 'NH2', 'NZ'] and \
                        atom.parent.resname in ['LYS', 'ARG'] and \
                            neighbor.name in ['OE1', 'OE2', 'OD1', 'OD2'] and \
                                neighbor.parent.resname in ['ASP', 'GLU']:
                        salt_bridges += 1
                    elif neighbor.name in ['NH1', 'NH2', 'NZ'] and \
                        neighbor.parent.resname in ['LYS', 'ARG'] and \
                            atom.name in ['OE1', 'OE2', 'OD1', 'OD2'] and \
                                atom.parent.resname in ['ASP', 'GLU']:
                        salt_bridges += 1
                except Exception as e:
                    print('Exception inside residue interactions,', e)

    # take average over residue
    b_factor = sum(b_factor) / len(b_factor)
        
    return h_bonds, salt_bridges, b_factor, h_bond_identities


def extract_features(database_loc, path):
    """
    Generate all the features and add them to the dataset_mapped.csv
    """

    # list for storing DSSP outputs
    dfs_acc = []

    # get the first instance of a unique mutation only (avoid duplications)
    db = pd.read_csv(database_loc)
    db = db.groupby('uid').first()
    #db['offset_rosetta'] = db['offset_rosetta'].fillna(0)

    # construct the output dataframe
    df_out = pd.DataFrame(index=db.index, 
        columns=['on_interface', 'entropy', 'conservation', 
                 'column_completeness', 'completeness_score', 'n_seqs', 
                 'structure_length', 'features', 'hbonds', 'h_bond_ids', 
                 'saltbrs', 'b_factor']) #, 'residue_depth'])

    df_out['code'] = df_out.index.str[:4]
    df_out['on_interface'] = False

    # sequence-based features from QuantiProt
    vol = aaindex.get_aa2volume() # residue total volume
    kdh = aaindex.get_aa2hydropathy() # Kyte-Doolittle hydrophobicity
    chg = aaindex.get_aa2charge() # neutral-pH charge

    # compute the sequence-based features
    df_out['kdh_wt'] = db['wild_type'].apply(lambda x: kdh[x])
    df_out['kdh_mut'] = db['mutation'].apply(lambda x: kdh[x])
    df_out['vol_wt'] = db['wild_type'].apply(lambda x: vol[x])
    df_out['vol_mut'] = db['mutation'].apply(lambda x: vol[x])
    df_out['chg_wt'] = db['wild_type'].apply(lambda x: chg[x])
    df_out['chg_mut'] = db['mutation'].apply(lambda x: chg[x])
     
    # iterate through each unique wild-type PDB structure
    for code, group in tqdm(db.groupby('code')):
        out_loc = os.path.join(path, 'results', code)
        if not os.path.exists(out_loc):
            print('Storing features in new directory')
            os.makedirs(out_loc)
        print(code)

        # get the structure and target chain where the mutation is
        pdb_file = group['pdb_file'].head(1).item()
        target_chain_id = group['chain'].head(1).item()
        alignment_file = group['msa_file'].head(1).item()
        print(alignment_file)

        # parse the structure and get its high-level model object
        structure = PDBParser().get_structure('PDB_ID', pdb_file)
        model = structure[0]

        # according to Bio.PDB
        #rd = ResidueDepth(model)

        # create a mapping of chain to tuples of (resnum, resname)
        chains = {chain.id:
                    [(residue.id[1], residue.resname) for residue in chain] \
                  for chain in structure.get_chains()
                 }

        # get interface residues less than 7 Angstrom from another chain
        interface_residues = get_interface_residues(
            model, target_chain_id, distance_threshold=7
            )
        
        # get the sequence of the chain of interest
        pdb_seq = chains[target_chain_id]
        # indices of this sequence (1...N)
        indices = [val[0] for val in pdb_seq]
        # 1 letter code for residues in the sequence
        aas = [d[val[1]] for val in pdb_seq]

        # convert to 1-based indexing
        interface_residues_indices = [
            indices.index(v)+1 for v in interface_residues
            ]
        # indexing a list which is 0-based using 1-based indexing
        interface_residue_identities = [
            aas[v-1] for v in interface_residues_indices
            ]

        #try:
        # according to DSSP
        df_acc = get_residue_accessibility(
            model, pdb_file, target_chain=group['chain'].head(1).item()
            )
        # add information so that this dataframe can be joined later
        df_acc['code'] = code
        df_acc = df_acc.reset_index()
        # convert to 1-based
        df_acc['position'] = df_acc['index'].apply(
            lambda x: indices.index(x)+1
            )
        dfs_acc.append(df_acc)
        #except Exception as e:
        #    print('Exception inside residue accessibility,', e)
        
        # now iterate through each unique mutation

        run_alistat(
            alignment_file, args.alistat_loc,
            os.path.join(out_loc, '')
            )

        for uid, row in group.iterrows():
            df_out.at[uid, 'structure_length'] = len(pdb_seq)
            df_out.at[uid, 'sequence_length'] = len(row['uniprot_seq'])

            wt = row['wild_type']
            target_pos = row['position'] #+ \
                #row['offset_up'] * (0 if dataset == 's669' else 1)
            try:
                target_pos = int(target_pos)
            except ValueError as e:
                print(f'Position {row["position"]} does not exist in structure {code}')
                continue

            res0 = Selection.unfold_entities(model[target_chain_id], 'R')[0]
            # should always start as 1 following recent changes
            offset = res0.id[1]
            # handle unknowns not being parsed
            if res0.resname == 'UNK':
                offset += 1
            # corrected position
            target_pos += offset - 1

            try:
                #validation
                assert d[model[target_chain_id][target_pos].resname] == wt
                # only assign validated residues
                #df_out.at[uid, 'residue_depth'] = rd[
                #    target_chain_id, (' ', target_pos, ' ')][0]
            except:
                try:
                    # indicate what discrepancy occured
                    print('wt:', wt, 'obs', 
                        d[model[target_chain_id][target_pos].resname], 
                        'target_pos', target_pos)
                    print([e.get_name() for e in Selection.unfold_entities(
                            model[target_chain_id], 'R'
                            )])
                except:
                    print('Could not find', target_pos)
                    continue

            # next section uses sequence-features, which are indexed differently

            # -1 for 0-based indexing
            if 'fireprot' not in args.db_loc.lower():
                target_pos_up = row['position'] + \
                    -row['offset_up'] - 1
            else:
                target_pos_up = row['position_orig'] - 1
            
            try:
                entropy, pri = get_conservation(
                    alignment_file, target_pos_up, wild_type=row['wild_type']
                    )
                df_out.at[uid, 'entropy'] = entropy
                df_out.at[uid, 'conservation'] = pri

                # assumes predictions and msa stats from AliStats are saved
                # in folders within results directory
                df_out.at[uid, 'column_completeness'] = get_column_completeness(
                        os.path.join(out_loc, '.Table_2.csv'
                        ), target_pos_up + 1)
                
                completeness, n_seqs = get_alignment_summary(
                    os.path.join(out_loc, '.Summary.txt'))
                #print(completeness, n_seqs, 1)
                df_out.at[uid, 'completeness_score'] = completeness
                df_out.at[uid, 'n_seqs'] = n_seqs

            except Exception as e:
                print('Exception inside alignment', e)

            if target_pos in interface_residues_indices:
                # validate again
                assert wt == interface_residue_identities[
                    interface_residues_indices.index(target_pos)]
                df_out.at[uid, 'on_interface'] = True
                df_out.at[uid, 'target_position'] = target_pos

            hbonds, saltbrs, b_factor, h_bond_identities = \
                get_residue_interactions(model, target_chain_id, target_pos)
            df_out.at[uid, 'hbonds'] = hbonds
            df_out.at[uid, 'h_bond_ids'] = h_bond_identities
            df_out.at[uid, 'saltbrs'] = saltbrs
            df_out.at[uid, 'b_factor'] = b_factor

            #if row['uniprot_id'] is not None:
                # undo zero-based indexing
                #try:
            #    df_out.at[uid, 'features'] = get_residue_features(
            #        row['uniprot_id'], None, target_pos_up + 1
            #        )
                #except Exception:
                #    print(uid, 'Timed out')
                #    df_out.at[uid, 'features'] = 'TIMEOUT'
            #else:
            #    df_out.at[uid, 'features'] = None

    return df_out.drop('code', axis=1), pd.concat(dfs_acc)


if __name__=='__main__':
    parser = argparse.ArgumentParser(
                    description = 'Processes features from either s669 or \
                                   FireProtDB for downstream analysis'
                    )
    parser.add_argument('--db_loc', help='location of the MAPPED database csv \
                            file, generated by preprocess.py. Must contain the \
                            name of the database (s669/fireprot)',
                        default='data/fireprot_mapped.csv')
    parser.add_argument('--alistat_loc', help='location of the Alistat repo. \
                            Do not use the relative path, ~/software... is ok',
                        required=True)
    parser.add_argument('-o', '--output_root', 
                        help='root of folder to store outputs',
                        default='.')

    args = parser.parse_args()

    db = pd.read_csv(args.db_loc)
    db = db.groupby('uid').first()
    db['offset_rosetta'] = 0
    
    # preserve original columns but compute for completeness
    db = db.rename({
        'conservation': 'conservation_fp', 'b_factor': 'b_factor_fp'
        }, axis=1)

    feat, dssp = extract_features(
        args.db_loc, args.output_root)

    feat_2 = db.join(feat, how='left')
    # sequence will be offset for FireProt because DSSP uses structures
    #if 'fireprot' in args.db_loc.lower():
    #    mask = feat_2['offset_up'].isna()
    #    feat_2.loc[~mask, 'position'] = feat_2.loc[~mask, 'position'] - feat_2.loc[~mask, 'offset_up']

    # cysteines in disulfide bonds have unusual names
    dssp.loc[dssp['wild_type'].isin(['a','b']), 'wild_type'] = 'C'
    dssp = dssp[['code', 'wild_type', 'SS', 'rel_ASA', 'position']]

    # combine with DSSP information (SASA and secondary structure)
    out = feat_2.merge(dssp, on=['code', 'wild_type', 'position'], how='left')

    # undo offset
    #if 'fireprot' in args.db_loc.lower():
    #    # First, select rows where 'offset_up' is not NaN
    #    mask = ~out['offset_up'].isna()
    #    # Then, update 'position' for these rows
    #    out.loc[mask, 'position'] = out.loc[mask].apply(lambda row: row['position'] - row['offset_up'], axis=1)

    out.to_csv(args.db_loc.replace('_mapped.csv', '_mapped_feats.csv')) 