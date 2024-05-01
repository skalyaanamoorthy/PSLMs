import urllib
import os
import gzip
import requests
import re
import shutil

from Bio import pairwise2, SeqIO
from Bio.PDB import PDBParser, PDBIO, Select
from Bio.SeqUtils import seq1

from modeller import *
from modeller.scripts import complete_pdb

import pandas as pd
import numpy as np

d = {'CYS': 'C', 'ASP': 'D', 'SER': 'S', 'GLN': 'Q', 'LYS': 'K', 'ILE': 'I', 
    'PRO': 'P', 'THR': 'T', 'PHE': 'F', 'ASN': 'N', 'GLY': 'G', 'HIS': 'H', 
    'LEU': 'L', 'ARG': 'R', 'TRP': 'W', 'ALA': 'A', 'VAL':'V', 'GLU': 'E', 
    'TYR': 'Y', 'MET': 'M', 'MSE': 'Z', 'UNK': '9', 'X': 'X'} 

class ChainSelect(Select):
    def __init__(self, chain):
        self.chain = chain

    def accept_chain(self, chain):
        return chain.id == self.chain

def download_assembly(code, chain, BIO_ASSEMBLIES_DIR):
    """
    Downloads the full (multimeric) biological assembly associated with a given
    PDB code as a gzip file.
    """

    # don't re-download (time-consuming)
    prot_file = f'{code}_{chain}.pdb1.gz'
    prot_path = os.path.join(BIO_ASSEMBLIES_DIR, prot_file)    
    if not os.path.exists(prot_path):
        print(f'Downloading {code} biological assembly')
        try:
            #these bio assemblies don't exist or cause issues
            # use monomer instead
            if code in ['1W4E', '1E0L', '1GYZ', '1H92', '1QLY', '1QM0',
                        '1URF', '1V1C', '1W4F', '1W4G', '1W4H', '2WNM']:
                urllib.request.urlretrieve(
                    f'http://files.rcsb.org/download/{code}.pdb.gz',
                    prot_path
                )
            else:
                urllib.request.urlretrieve(
                    (f'https://files.wwpdb.org/pub/pdb/data/biounit/PDB/all/'
                    f'{code.lower()}.pdb1.gz'), 
                    prot_path
                )
        # only happens due to connection issues
        except Exception as e:
            print(e)
            print(f'Downloading {code} failed')
    
    # convert the file with a different structure
    if code == '1W4E':
        lines = gzip.open(prot_path, 'rt').readlines()
        lines = lines[:954]
        with gzip.open(prot_path, 'wt') as g:
            g.writelines(lines)

    return prot_path, prot_file


def get_uniprot(code, chain, SEQUENCES_DIR, uniprot_id=None):
    """
    Gets the UniProt sequence (and accession code, for computing features)
    for specifically S669 / S461 proteins, since these are not provided with
    the database. Uses the PDB sequence if this cannot be found.
    """
    origin = None

    if uniprot_id is None:
        # uniprot entries corresponding to multichain PDBs may need to be specified   
        if code in ['1ACB', '1AON', '1GUA', '1GLU', '1OTR', '2CLR', '3MON']:
            entity = 2
        elif code in ['1HCQ', '1NFI', '1TUP', '3DV0']:
            entity = 3
        else:
            entity = 1

        # get the uniprotkb data associated with the PDB code if it existss
        req = (
            f'https://www.ebi.ac.uk/pdbe/graph-api/pdbe_pages/uniprot_mapping/'
            f'{code.lower()}/{entity}'
        )

        # convert json to Python
        r = requests.get(
            req).text.replace('true','True').replace('false','False')
        try:
            r = eval(r)
        except Exception as e:
            print(e)
            print('It looks like the PDB e-KB might be temporarily down.')
            print('Please try again later.')

    # get specifically the sequence related to the target structure
    try:
        if uniprot_id is None:
            data = r[code.lower()]
            # get the uniprotkb accession (skip interpro entries which have _)
            num = -1
            accession = '_'
            while '_' in accession:
                num += 1
                accession = data['data'][num]['accession']
        else:
            accession = uniprot_id

        # query uniprotkb for the accession to get the FULL sequence 
        # (used for alignment searching as it gives improved performance)
        req2 = f'https://rest.uniprot.org/uniprotkb/{accession}'
        up_info = requests.get(req2).text
        uniprot_seq = up_info.split(
            '"sequence":{"value":')[-1].split(',')[0].strip('\""')

        # reduce the length of the titin sequence to a relevant window
        if code == '1TIT':
            uniprot_seq = uniprot_seq[11742:11742+1024]
            #print(uniprot_seq)

        origin = up_info.split(
            '"lineage":[')[-1].split(']')[0].strip('\""').split('\"')[0]

        with open(
            os.path.join(SEQUENCES_DIR, 'fasta_up', f'{code}_{chain}.fa'), 'w'
            ) as f:
                f.write(f'>{code}_{chain}\n{uniprot_seq}')
               
    # e.g. 1FH5, which is an FAB fragment
    except KeyError:
        print(f'No UP for {code}')
        uniprot_seq = None
        accession = None

    # UniProt sequence has incorrect residues for second half of protein
    # so just use the PDB sequence for searching
    # for 1TIT and 1WIT, the UniProt sequences are too long
    if code in ['1IV7', '1IV9', '1TIT', '1WIT']:
        uniprot_seq = None

    return uniprot_seq, accession, origin


def renumber_pdb(pdb_file, output_file):
    """
    Renumbers the residues in the PDB file sequentially
    """

    # Parse the structure
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('structure', pdb_file)

    # Temporarily renumber the residues with a large offset to avoid conflicts
    # (where two residues share the same identity)
    offset = 10000
    for model in structure:
        for chain in model:
            for i, residue in enumerate(chain.get_list(), start=1):
                residue.id = (' ', i + offset, ' ')

    # Sequentially renumber the residues (starting from 1)
    for model in structure:
        for chain in model:
            residues = sorted(chain.get_list(), key=lambda res: res.get_id()[1])
            for i, residue in enumerate(residues, start=1):
                residue.id = (' ', i, ' ')

    # Write the output file
    io = PDBIO()
    io.set_structure(structure)
    io.save(output_file)


def repair_pdb(pdb_file, output_file):
    """
    Repairs missing atoms / residues using Modeller. Requires Modeller and
    the LIB environment variable to be set to the appropriate directory
    """

    # setup Modeller
    env = Environ()
    env.libs.topology.read(file='$(LIB)top_heav.lib')
    env.libs.parameters.read(file='$(LIB)par.lib')
    # get the PDB code
    filename = os.path.basename(pdb_file)

    # repairing these structures causes as numbering conflict with UniProt
    if filename[:4] in ['1G3P', '1IR3', '4HE7']:
        return
    # the other chain in these structures is DNA, causing errors
    #if filename[:4] in ['1AZP', '1BNZ', '1C8C']:
    #    mdl = complete_pdb(env, pdb_file, model_segment=('1:A', 'LAST:A'))
    #if filename[:4] == '1R2Y':
    #    mdl = complete_pdb(env, pdb_file, model_segment=('2:A', 'LAST:A'))
    # usually nothing is missing, so the structure is unchanged
    else:
        mdl = complete_pdb(env, pdb_file)
    mdl.write(file=output_file)


def extract_structure(code, chain, d, prot_path, prot_file,
                        STRUCTURES_DIR, compressed=True):
    """
    Using the gzip assembly from the PDB, parse the file to get the sequence of 
    interest from the structure
    """

    pdbparser = PDBParser()

    # the chains will end up getting renamed, as sometimes in the assembly
    # two chains will share a name, causing errors
    chain_names = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    
    if compressed:
        lines = gzip.open(prot_path, 'rt').readlines()
    else:
        lines = open(prot_path, 'r').readlines()

    # chain whose sequence we want (becomes mutated)
    target_chain = chain
    target_chain_found = False
    new_idx = -1
    skip_ter = False
    skip_hets = False
    is_nmr = False
     
    # this section puts all structures in a consistent format
    # first, specify the output location used for inference
    target_structure = re.sub(
        '.pdb1.gz', '.pdb', os.path.join(STRUCTURES_DIR, prot_file)
        )
    with open(target_structure, 'w') as f:
        # lines from gzip
        for line in lines:
            # erase model lines / replace with TER and replace chain as needed 
            # unless it is target. partly to ensure chain name uniqueness

            # multiple models usually implies NMR structures, but all bio-
            # assemblies have at least one model
            if 'SOLUTION NMR' in line:
                print('Treating NMR structure.')
                is_nmr = True
            if line[0:5] == 'MODEL':
                new_idx += 1
                # select a new unique chain name
                chn = chain_names[new_idx]
                # if this is the next model, terminate the previous chain
                if int(new_idx) > 0 and not skip_ter:
                    f.write('TER\n')
                elif skip_ter:
                    skip_ter = False
                if chn == target_chain:
                    new_idx += 1
                    chn = chain_names[new_idx]
                continue
            # remove DNA (deoxynucleotides)
            if line[18:20] in ['DA', 'DT', 'DC', 'DG']:
                continue
            # don't output this
            elif line[0:6] == 'ENDMDL':
                if is_nmr:
                    break
                continue
            # rewrite lines with atom records according to the new designation
            elif line[0:4] == 'ATOM':
                # by default, include heteroatoms which occur within a chain
                # since they are probably associated with usual residues
                skip_hets = False
                # reassign the chain, unless it is the target chain, in which
                # case there are checks to ensure the designation is correct
                if line[21]==target_chain and not target_chain_found:
                    target_chain_found = True
                    chn = target_chain
                f.write(line[:21]+chn+line[22:])
                continue
            # remove heteroatoms which are not presumed residues 
            # (usually ligands and waters)
            elif line[0:6] == 'HETATM' and skip_hets:
                continue
            elif line[0:6] == 'HETATM':
                # it is possible for the target chain to start with HETS
                if line[21]==target_chain and not target_chain_found:
                    target_chain_found = True
                    chn = target_chain
                if line[17:20] not in d.keys():
                    # original residue      
                    old = line[17:20]
                    line = list(line)
                    # exclude sepharose, aminosuccinimide, acetyl
                    if ''.join(line[17:20]) in ['SEP', 'SNN', 'ACE']:
                        print(
                            f'Omitting residue {"".join(line[17:20])} in {code}'
                            )
                        continue
                    else:
                        # ESM-IF uses a 3-1 letter encoding that cannot handle 
                        # nonstandard residues except 'UNK'
                        line[17:20] = 'UNK'
                    line = ''.join(line)
                    print(f'Converted {old} in {code} to {line[17:20]}')
                f.write(line[:21]+chn+line[22:])
                continue               
            elif line[0:3] == 'TER':
                f.write(line[:21]+chn+line[22:])
                new_idx += 1
                # when moving on to a new chain, chose a new name which is not 
                # the name of the target
                chn = chain_names[new_idx]
                if chn == target_chain:
                    new_idx += 1
                    chn = chain_names[new_idx]
                # deletes waters and ligands by default
                skip_hets = True
                # ensures additional TER won't be written
                skip_ter = True
                continue
            f.write(line)
    
    assert target_chain_found

    structure = pdbparser.get_structure(code, target_structure)

    # create a mapping like: (chain) A: [(1, MET), (2, PHE), ...]
    chains_orig = {chain.id:[
        (residue.id[1], residue.resname) for residue in chain
        ] for chain in structure.get_chains()}

    # repair missing atoms / residues
    repair_pdb(target_structure, target_structure)
    # renumber sequentially
    renumber_pdb(target_structure, target_structure)

    # add in a CRYST1 line so that DSSP will accept the file
    text_to_insert = \
     'CRYST1    1.000    1.000    1.000  90.00  90.00  90.00 P 1           1 '

    with open(target_structure, 'r') as original_file:
        lines = original_file.readlines()

    if 'MODELLER' in lines[0]:
        lines.insert(1, text_to_insert + '\n')
    else:
        lines.insert(0, text_to_insert + '\n')

    with open(target_structure, 'w') as modified_file:
        modified_file.writelines(lines)

    io = PDBIO()
    
    # the remainder can be handled by an existing parser
    structure = pdbparser.get_structure(code, target_structure)
    io.set_structure(structure)
    new_filename = os.path.join(
    STRUCTURES_DIR, 'single_chains', f'{code}_{chain}.pdb'
    )
    io.save(new_filename, ChainSelect(target_chain))

    # create a mapping like: (chain) A: [(1, MET), (2, PHE), ...]
    chains = {chain.id:[(residue.id[1], residue.resname) for residue in chain]\
        for chain in structure.get_chains()}

    mapping = pd.DataFrame(
        columns=['author_id', 'sequential_id', 'pdb_seq', 'repaired_seq'])
    seq_orig = ''.join([d[res[1]] for res in chains_orig[target_chain]])
    new_seq =  ''.join([d[res[1]] for res in chains[target_chain]])
    if seq_orig != new_seq:
        print('Modelled residues added!')
        aln = pairwise2.align.globalms(seq_orig, new_seq, 2, 0.5, -1, -0.1)[0]
        mapping['pdb_seq'] = list(aln.seqA)
        mapping['repaired_seq'] = list(aln.seqB)
        mapping.loc[mapping['pdb_seq']!='-', 'author_id'] = \
            [res[0] for res in chains_orig[target_chain]]
        mapping.loc[mapping['repaired_seq']!='-', 'sequential_id'] = \
            [res[0] for res in chains[target_chain]]
        if len(mapping.loc[mapping['repaired_seq']!='-']) != len(mapping):
            print('Deleted one or more residues!')
    else:
        mapping['pdb_seq'] = list(seq_orig)
        mapping['repaired_seq'] = list(new_seq)
        mapping.loc[mapping['pdb_seq']!='-', 'author_id'] = \
            [res[0] for res in chains_orig[target_chain]]
        mapping.loc[mapping['repaired_seq']!='-', 'sequential_id'] = \
            [res[0] for res in chains[target_chain]]

    multimer = len(chains.keys())

    return mapping, is_nmr, multimer


def align_sequence_structure(code, chain, pdb_ungapped, dataset, mapping_df,
                             SEQUENCES_DIR, WINDOWS_DIR, ALIGNMENTS_DIR, 
                             min_pos, max_pos, indexer, uniprot_seq=None):
    """
    In this critical preprocessing step, the mutations from the database are 
    mapped to the structures that will be used by inverse folding methods 
    downstream. Since the predictive methods can be finicky about data format, 
    some manual corrections are made. This function also writes the alignments 
    for validation, and selects a window of the full UniProt sequence which best 
    encompasses the structure while maximimizing length.
    """

    # hand-adjusted alignments
    if code == '1LVM':
        pdb_gapped = ['-']*2029 + list(pdb_ungapped)
        pdb_gapped = ''.join(pdb_gapped)
        uniprot_gapped = uniprot_seq
    elif code == '1GLU':
        pdb_gapped = ['-']*433 + list(pdb_ungapped)
        pdb_gapped = ''.join(pdb_gapped)
        uniprot_gapped = uniprot_seq
    #elif code == '1TIT':
    #    pdb_gapped = ['-']*12676 + list(pdb_ungapped)
    #    pdb_gapped = ''.join(pdb_gapped)
    #    uniprot_gapped = uniprot_seq

    # in most cases, it suffices to do an automatic alignment
    # code 1LVE because UniProt seq does not cover mutations
    elif uniprot_seq is None or code == '1LVE':
        pdb_gapped = pdb_ungapped
        uniprot_gapped = pdb_ungapped
        uniprot_seq = pdb_ungapped
        # create a "fake" UniProt sequence from the PDB sequence if 
        # no UniProt is associated
        fake_up_seq = pdb_ungapped
        # would normally be saved when it is found 
        with open(
            os.path.join(SEQUENCES_DIR, 'fasta_up', f'{code}_{chain}.fa'), 
        'w') as f:
            f.write(f'>{code}_{chain}\n{fake_up_seq}')
        
    else:
        #print('up ',uniprot_seq)
        #print('pdb',pdb_ungapped)
        # get the highest-scoring candidate alignment
        aln = pairwise2.align.globalms(
            uniprot_seq, pdb_ungapped, 2, 0.5, -1, -0.1
            )[0]
        uniprot_gapped = aln.seqA
        # pdb_gapped is the PDB sequence, with added gaps (-) to match up 
        # with  FireProtDB (UniProt) sequences.
        pdb_gapped = aln.seqB

    # hand-adjusted alignment
    if code == '1AAR':
        pdb_gapped = ['-']*608 + list(pdb_ungapped)
        pdb_gapped = ''.join(pdb_gapped)

    # dataframe which shows how the sequence-structure alignment was created
    alignment_df = pd.DataFrame(index=range(100000))
    # paste in the sequences
    alignment_df['uniprot_gapped'] = pd.Series(list(uniprot_gapped))
    alignment_df['pdb_gapped'] = pd.Series(list(pdb_gapped))
    # drop extra rows
    alignment_df = alignment_df.dropna()

    alignment_df.loc[alignment_df['uniprot_gapped']!='-', 'uniprot_id'] = \
        range(1, len(alignment_df.loc[alignment_df['uniprot_gapped']!='-'])+1)

    alignment_df.loc[alignment_df['pdb_gapped']!='-', 'sequential_id'] = \
        range(1, len(alignment_df.loc[alignment_df['pdb_gapped']!='-'])+1)

    mapping_df = mapping_df.rename({'repaired_seq': 'pdb_gapped'}, axis=1)
    alignment_df = alignment_df.merge(
        mapping_df, on=['sequential_id', 'pdb_gapped'], how='outer')
        #).drop_duplicates(subset='uniprot_id', keep='first')

    if alignment_df.at[len(alignment_df)-1, 'pdb_seq'] == '9':
        print('Removing terminal non-canonical residue')
        alignment_df = alignment_df.loc[:len(alignment_df)-2, :]

    #alignment_df['uniprot_id'] = alignment_df['uniprot_id'].astype(int)
    #alignment_df['sequential_id'] = alignment_df['sequential_id'].astype(int)
    alignment_df['chosen_index'] = alignment_df[indexer]

    alignment_df.to_csv(
        os.path.join(ALIGNMENTS_DIR, f'{code}_uniprot_aligned.csv')
        )

    # for the MSA transformer, we can only have 1023 characters in the sequence
    # this block extracts the most relevant residues up to 1023 from the UniProt
    # sequence: the first 1023 residues which also fully cover the structure, 
    # extending past the N- and then C- terminus if there is space

    # assume mutants are always within the structure range
    if dataset != 'fireprot':
        window_start = alignment_df.set_index(
            'chosen_index').at[min_pos, 'sequential_id']
        if code != '2MWA':
            window_end = alignment_df.set_index(
                'chosen_index').at[max_pos, 'sequential_id']
        else:
            window_end = 34
        #print(window_start)
        # we now have the start and end of the mutant range, extend it to include 
        # the whole structure
        while window_start > 1 and \
            window_end - window_start < alignment_df['sequential_id'].max():
            window_start -= 1
        while window_end < alignment_df['sequential_id'].max() and \
            window_end - window_start < 1022:
            window_end += 1
        # now switch into the UniProt sequence frame which is usually longer
        window_start = alignment_df.set_index(
            'sequential_id').at[window_start, 'uniprot_id']
        window_end = alignment_df.set_index(
            'sequential_id').at[window_end, 'uniprot_id']
    # mutants are always within the UniProt range
    else:
        window_start = min_pos
        window_end = max_pos

    #print(window_start, min_pos, max_pos)
    # if there is no Uniprot at the start of the sequence, just use it all
    if np.isnan(window_start):
        window_start = 1
    if np.isnan(window_end):
        window_end = 1022

    # try to extend it backward until we reach the start of the UniProt
    while window_start > 1 and window_end - window_start < 1022:
        window_start -= 1
    # ran out of sequence at N-term, so add some on the C-term
    while window_end - window_start < 1022:
        window_end += 1

    window_start, window_end = int(window_start)-1, int(window_end)-1
    
    if dataset == 'fireprot':
        if not (min_pos-1 >= window_start and max_pos-1 <= window_end):
            print(min_pos-1, window_start, max_pos-1, window_end)

    # this is the sequence used by MSA-Transformer and ESM-1V
    # but not Tranception
    uniprot_seq_reduced = uniprot_seq[window_start:window_end]
    with open(os.path.join(WINDOWS_DIR, f'{code}_{chain}'), 'w') as f:
        f.write(f'{window_start},{window_end}')
    with open(os.path.join(WINDOWS_DIR, f'{code}_{chain}.fa'), 'w') as f:
        f.write(f'>{code}_{chain}\n{uniprot_seq_reduced}')
    alignment_df.to_csv(
        os.path.join(ALIGNMENTS_DIR, f'{code}_uniprot_aligned.csv')
        )

    return alignment_df, window_start, pdb_ungapped, uniprot_seq


def get_offsets(wt, pos, dataset, alignment_df):
    """
    Count the number of gaps in the uniprot sequences (caused by insertions in 
    the PDB structure), ensuring that all gaps prior to the target mutation 
    position are counted is preserved
    """
    mismatch = False
    no_match = False
    
    seq1 = ''.join(list(alignment_df['uniprot_gapped']))
    seq2 = ''.join(list(alignment_df['pdb_gapped']))
    seq1_ = seq1
    seq1_ = list(seq1)
    seq2_ = seq2
    seq2_ = list(seq2)

    # case where the PDB is mutated relative to UniProt (or has gap)
    pdb_res = alignment_df.set_index('chosen_index').loc[pos, 'pdb_gapped']
    up_res = alignment_df.set_index('chosen_index').loc[pos, 'uniprot_gapped']
    up = alignment_df.set_index('chosen_index').at[pos, 'uniprot_id']
    try:
        up = int(up)
    except ValueError:
        return np.nan, np.nan, np.nan
    except TypeError:
        print('Unknown position', wt, pos)
        return np.nan, np.nan, np.nan

    sid = alignment_df.set_index('chosen_index').at[pos, 'sequential_id']
    idx_mut = alignment_df.reset_index().set_index(
        'chosen_index').at[pos, 'index']
    offset_up = sid - up
    seq1_.insert(idx_mut, '[')
    seq1_.insert(idx_mut+2, ']')
    seq1_ = ''.join(seq1_)
    seq2_.insert(idx_mut, '[')
    seq2_.insert(idx_mut+2, ']')
    seq2_ = ''.join(seq2_) 
    if pdb_res == '-':
        print(seq1_)
        print(seq2_)
        no_match = True
        if not dataset == 'fireprot':
            raise AssertionError('Mutant position does not exist in structure!')
        else:
            print('Mutant position does not exist in structure!')
    elif up_res == '-':
        print(seq1_)
        print(seq2_)
        no_match = True
        raise AssertionError('Mutant is an insertion relative to UniProt!')
    elif pdb_res != wt:
        print(seq1_)
        print(seq2_) 
        print(f'PDB is {pdb_res} but should be {wt} at {pos}')
        mismatch = True
    elif up_res != wt and dataset != 'ssym':
        print(seq1_)
        print(seq2_) 
        print(f'UniProt is {up_res} but should be {wt} at {up}')
        mismatch = True      
        
    if not no_match:
        offset_up = int(offset_up)
        sid = int(sid)

    return offset_up, sid, mismatch


def generate_mutant_sequences(code, chain, pos, mut, pdb_ungapped,
                              SEQUENCES_DIR):
    """
    Save the sequences of the wild-type and mutant proteins, returning the 
    latter. Mainly for record-keeping.
    """

    with open(
        os.path.join(SEQUENCES_DIR, 'fasta_wt', f'{code}_{chain}_PDB.fa'), 
        'w') as f:
        f.write(f'>{code}_{chain}\n{pdb_ungapped}')

    # modify the string in the target position
    mutseq = list(pdb_ungapped)
    mutseq[int(pos) - 1] = mut
    mutseq = ''.join(mutseq)

    with open(
        os.path.join(SEQUENCES_DIR, 'fasta_mut', f'{code}_{chain}_PDB.fa'),
        'w') as f:
        f.write(f'>{code}_{chain}\n{mutseq}')

    return mutseq


def save_formatted_data(code, chain, group, dataset, output_root):
    """
    Save information about the mutants in the formats expected for each method.
    """

    # open the Tranception file
    with open(os.path.join(
        output_root, 'DMS_Tranception', f'{code}_{chain}_{dataset}.csv'
        ), 'w') as trance:
        trance.write('mutated_sequence,mutant\n')

        # Open the MSA-Transformer file (also used by ESM-1V)
        with open(os.path.join(
            output_root, 'DMS_MSA', f'{code}_{chain}_{dataset}.csv'),
             'w') as msa:
            msa.write(',mutation\n')

            # iterate through the mutations, writing each one after validation
            for (wt, pos, mut, ou, seq, ws), _ in group.groupby(
                ['wild_type', 'position', 'mutation', 
                'offset_up', 'uniprot_seq', 'window_start']):

                uniprot_seq = list(seq)
                try:
                    assert uniprot_seq[pos - 1 -ou] == wt,\
                        ('Wrote a mutation whose wt disagrees with uniprot_seq')
                except Exception as e:
                    print(e, code, wt, pos, mut)

                # generation of the mutant sequence
                uniprot_seq[pos-1-ou] = mut
                mutated_uniprot_seq = ''.join(uniprot_seq)

                # write to the Tranception file
                trance.write(f'{mutated_uniprot_seq},{wt}{pos - ou}{mut}\n')

                # write to MSA-Transformer file
                new_pos = pos - ou - ws
                msa.write(f',{wt}{new_pos}{mut}\n')
