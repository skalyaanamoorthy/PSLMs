import os
import pandas as pd
import numpy as np
import glob
import warnings
import random
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn import metrics
import matplotlib
import re
from tqdm.notebook import tqdm
from matplotlib.patches import Patch
from scipy.stats import spearmanr, pearsonr
from scipy import stats
from seaborn import diverging_palette
from matplotlib.colors import Normalize, ListedColormap
from matplotlib.lines import Line2D
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from copy import deepcopy
from sklearn.feature_selection import RFE
import matplotlib.pyplot as plt
import numpy as np
from math import pi
import matplotlib.font_manager as font_manager
import sys
from contextlib import contextmanager, redirect_stdout, redirect_stderr
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.base import clone
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec


@contextmanager
def suppress_output():
    """
    Context manager to suppress stdout and stderr.
    """
    with open(os.devnull, 'w') as devnull:
        with redirect_stdout(devnull), redirect_stderr(devnull):
            yield

# convert names used in inference outputs to those used in figures
remap_names = {
    'esmif_monomer': 'ESM-IF(M)', 
    'esmif_multimer': 'ESM-IF', 
    'mpnn_mean': 'ProteinMPNN mean', 
    'esm2_mean': 'ESM-2 mean',
    'esmif_mean': 'ESM-IF mean',
    'mif_mean': 'MIF mean',
    'msa_transformer_mean': 'MSA-T mean',
    'msa_transformer_median': 'MSA-T median',
    'esm1v_mean': 'ESM-1V mean',
    'esm1v_median': 'ESM-1V median',
    'esm2_150M': 'ESM-2 150M',
    'esm2_650M': 'ESM-2 650M',
    'esm2_3B': 'ESM-2 3B',
    'esm2_15B_half': 'ESM-2 15B',
    'mif': 'MIF', 
    'mifst': 'MIF-ST', 
    'monomer_ddg': 'Ros_ddG_monomer', 
    'cartesian_ddg': 'Rosetta CartDDG', 
    'mpnn_10_00': 'ProteinMPNN 0.1', 
    'mpnn_20_00': 'ProteinMPNN 0.2', 
    'mpnn_30_00': 'ProteinMPNN 0.3', 
    'mpnn_sol_10_00': 'ProteinMPNN_sol 0.1', 
    'mpnn_sol_20_00': 'ProteinMPNN_sol 0.2', 
    'mpnn_sol_30_00': 'ProteinMPNN_sol 0.3', 
    'tranception': 'Tranception (reduced)', 
    'tranception_weights': 'Tranception',
    'tranception_original': 'Tranception_original',
    'tranception_reproduced': 'Tranception_reproduced',
    'tranception_target': 'Tranception_target',
    'esm1v_2': 'ESM-1V 2', 
    'msa_1': 'MSA-T 1', 
    'korpm': 'KORPM',
    'Evo': 'EvoEF',
    'msa_transformer_median': 'MSA-T median',
    'ankh': 'Ankh',
    'saprot_pdb': 'SaProt PDB',
    'saprot_af2': 'SaProt AF2',
    'structural': 'Structural',
    'evolutionary': 'Evolutionary',
    'supervised': 'Supervised',
    'clustered_ensemble': 'Clustered Ensemble',
    'mpnn_rosetta': 'Rosetta/ProtMPNN',
    'mutcomputex': 'MutComputeX',
    'stability-oracle': 'Stability Oracle',
    'delta_kdh': 'Δ hydrophobicity', 
    'delta_vol': 'Δ volume', 
    'delta_chg': 'Δ charge',
    'rel_ASA': 'relative ASA',
    'q3421_pslm_rfa_2': 'Ensemble* 2 Feats',
    'q3421_pslm_rfa_3': 'Ensemble* 3 Feats',
    'q3421_pslm_rfa_4': 'Ensemble* 4 Feats',
    'q3421_pslm_rfa_5': 'Ensemble* 5 Feats',
    'q3421_pslm_rfa_6': 'Ensemble* 6 Feats',
    'q3421_pslm_rfa_7': 'Ensemble* 7 Feats',
    'q3421_pslm_rfa_8': 'Ensemble* 8 Feats',
    'K1566_pslm_rfa_2': 'Ensemble 2 Feats',
    'K1566_pslm_rfa_3': 'Ensemble 3 Feats',
    'K1566_pslm_rfa_4': 'Ensemble 4 Feats',
    'K1566_pslm_rfa_5': 'Ensemble 5 Feats',
    'K1566_pslm_rfa_6': 'Ensemble 6 Feats',
    'K1566_pslm_rfa_7': 'Ensemble 7 Feats',
    'K1566_pslm_rfa_8': 'Ensemble 8 Feats',
    'random': 'Gaussian Noise',
    'ddG': 'ΔΔG label', 
    'dTm': 'ΔTm label',
    'upper_bound': 'Theoretical Max'
    }
#    'random_1': 'Gaussian Noise',

# predictions will have dir in their name to specify direct mutation
remap_names_2 = {f"{key}_dir": value for key, value in remap_names.items()}

remap_cols = {  'auprc': 'AUPRC', 
                'spearman': 'Spearman\'s ρ', 
                'auppc': 'mean PPC', 
                'aumsc': 'mean MSC', 
                'weighted_ndcg': 'wNDCG', 
                'ndcg': 'NDCG',
                'weighted_spearman': 'wρ', 
                'weighted_auprc': 'wAUPRC', 
                'tp': 'True Positives', 
                'sensitivity': 'Sensitivity', 
                'mean_stabilization': 'Mean Stabilization',
                'net_stabilization': 'Net Stabilization',
                'mean_squared_error': 'MSE',
                'accuracy': 'Accuracy', 
                'mean_reciprocal_rank': 'MRR', 
                'n': 'n', 
                'MCC': 'MCC', 
                'recall@k1.0': 'Recall @ k1',
                'recall@k0.0': 'Recall @ k0'
                }

# check if these substrings are in the name of a model in order to assign colors
evolutionary = ['tranception', 'msa_transformer', 'esm1v', 'msa', 'esm2', 'ankh']
structural = ['mpnn', 'mif', 'mifst', 'esmif', 'mutcomputex', 'sapro']
supervised = ['MAESTRO', 'ThermoNet', 'INPS', 'PremPS', 'mCSM', 'DUET', 'I-Mutant3.0', 'SAAFEC', 'MUpro', 'MuPro']
untrained = ['DDGun']
transfer = ['stability-oracle', 'ACDC']
potential = ['KORPM', 'PopMusic', 'SDM', 'korpm', 'PoPMuSiC']
biophysical = ['cartesian_ddg', 'FoldX', 'Evo', 'CartDDG']
ensemble = ['ens', 'mpnn_rosetta', 'rfa', ' + ']
unknown = ['ddG', 'dTm', 'random', 'delta', 'ASA', 'Dynamut', 'upper_bound']

categories = tuple(['struc. PSLM', 'seq. PSLM', 'transfer', 'biophysical', 'potential', 'untrained', 'supervised', 'unknown', 'unused', 'ensemble'])
colors = tuple(list(sns.color_palette('tab10'))[:len(categories)])
custom_colors = dict(zip(categories, colors)) 

mapping_categories = {  'ensemble': ensemble,
                        'unknown': unknown,
                        'struc. PSLM': structural,
                        'seq. PSLM': evolutionary,
                        'supervised': supervised,
                        'untrained': untrained,
                        'transfer': transfer,
                        'potential': potential,
                        'biophysical': biophysical,
                      }

def determine_category(model):
    for k,v in mapping_categories.items():
        if any(v_ in str(model) for v_ in v):
            category = k
            return category

def determine_base_color(model):
    return custom_colors[determine_category(model)]

def generate_palette(base_color):
    # Generate the base palette
    palette = [sns.light_palette(base_color, n_colors=4, reverse=True)[0]]
    for p in range(1, 4):
        palette.append(sns.light_palette(base_color, n_colors=7, reverse=True)[::2][p])
        palette.append(sns.dark_palette(base_color, n_colors=7, reverse=True)[::2][p])

    # Predefined offsets to create variation
    # Ensure these offsets keep the colors within the [0, 1] range after application
    offsets = [
        (0, 0, 0),  # First color unchanged
        (0.07, 0.07, 0),  # Slightly increase contrast for the second color
        (-0.08, 0.08, -0.),  # Increase contrast for the third color
        (0.07, -0.07, 0.07),  # Significantly alter the fourth color for more distinction
        (-0.06, 0.6, -0.06),  # Minor adjustments for the fifth to balance the palette
        (0.04, -0.04, 0.04),  # Continue with subtle changes
        (-0.04, 0.04, -0.04)  # And further subtle changes
    ]

    # Apply deterministic offsets to each color in the palette
    deterministic_palette_hex = []
    for color, offset in zip(palette, offsets):
        # Adjust each color component within the clipping bounds
        adjusted_color = np.clip(np.array([c + o for c, o in zip(color, offset)]), 0, 1)
        deterministic_palette_hex.append(tuple(adjusted_color))

    return deterministic_palette_hex

# Function to stochastically select a color
def select_color_from_palette(palette, used_colors):
    i = 0
    color = palette[i]
    while color in used_colors:
        i += 1
        color = palette[i]
        #print(i)
    return color

# Function to assign color
def assign_color(model, used_colors, palette):
    selected_color = select_color_from_palette(palette, used_colors)
    return sns.color_palette([selected_color])[0]  # Convert to RGB

def get_color_mapping(data, column='variable'):
    used_colors = set()
    palettes = {}
    color_mapping = {}
    for var in data[column].unique():
        print(var)
        base_color = determine_base_color(var)
        if base_color in palettes.keys():
            palette = palettes[base_color]
        else:
            palette = generate_palette(base_color)
            palettes[base_color] = palette

        color_mapping[var] = assign_color(var, used_colors, palette)
        used_colors.add(color_mapping[var])
    return color_mapping

def format_fixed_total_digits(number, total_digits=4):
    try:
        integer_part = int(number)  # Convert to integer to get the integer part
        integer_digits = len(str(abs(integer_part)))  # Count digits in the integer part, ignore the minus sign if negative
        decimal_places = total_digits - integer_digits  # Calculate the remaining digits for the decimal part

        # Adjust for the decimal point itself
        decimal_places = max(0, decimal_places - 1) if total_digits > integer_digits else 0
    
        # Create format string dynamically based on calculated decimal places
        format_string = f"{{:.{decimal_places}f}}"
        return format_string.format(number)
    except (TypeError, ValueError):
        return number  # Return the original number if there's an error


def bootstrap_by_grouper(dbf, n_bootstraps, grouper='code', drop=True, noise=0, target='ddG', duplicates=True):
    if grouper == 'code' and not 'code' in dbf.columns:
        dbf['code'] = dbf.index.str[:4]
    if grouper is not None:
        groups = list(dbf[grouper].unique())
    else:
        groups = list(set(dbf.index))
    out = []
    for i in range(n_bootstraps):
        redraw = []
        if grouper is not None:
            while len(redraw) < len(groups):
                group = random.choice(groups)
                new_db = dbf.loc[dbf[grouper]==group]
                if drop:
                    new_db = new_db.drop(grouper, axis=1)
                redraw.append(new_db)
            df_bs = pd.concat(redraw, axis=0)
        else:
            df_bs = dbf.sample(frac=1, replace=True)
        if noise > 0:
            df_bs[target] += np.random.normal(scale=noise, size=len(df_bs))
        if not duplicates:
            df_bs = df_bs.drop_duplicates()
        out.append(df_bs)
    return out


def compute_ndcg(dbf, pred_col, true_col, pos=True):
    
    df = dbf.copy(deep=True)

    if pos:
        df.loc[df[true_col]<0, true_col] = 0
        if all(df[true_col]==0):
            return np.nan
    else:
        # Shift scores to be non-negative
        min_score = df[true_col].min()
        shift = 0
        if min_score < 0:
            shift = -min_score

        df[true_col] += shift
    
    # Sort dataframe by ground truth labels
    df_sorted = df.sort_values(by=pred_col, ascending=False)
    #print(df_sorted)
    
    # Get the sorted predictions
    sorted_preds = df_sorted[pred_col].values
    
    # Use the ground truth labels as relevance scores
    relevance = df_sorted[true_col].values
    
    # Reshape data as ndcg_score expects a 2D array
    sorted_preds = sorted_preds.reshape(1, -1)
    relevance = relevance.reshape(1, -1)
    
    # Compute and return NDCG
    try:
        ndcg = metrics.ndcg_score(relevance, sorted_preds)
        #if pred_col == 'ddG_dir':
        #    print(ndcg)
        #    print(sorted_preds)
        return ndcg
    except Exception as e:
        print(e)
        print(pred_col)
        print(sorted_preds)
        print(true_col)
        print(relevance)


# definition of statistics used for model combination heatmaps
def compute_weighted_ndcg(dbf, pred_col, true_col, grouper='code'):
    cur_dbf = dbf.copy(deep=True)
    #cur_dbf['code'] = cur_dbf.index.str[:4]
    cur_dbf = cur_dbf[[pred_col, true_col, grouper]]

    w_cum_ndcg = 0
    w_cum_d = 0
    for _, group in cur_dbf.groupby(grouper): 
        if len(group.loc[group[true_col]>0]) > 1 and not all(group[true_col]==group[true_col][0]):
            cur_ndcg = compute_ndcg(group, pred_col, true_col)
            w_cum_ndcg += cur_ndcg * np.log(len(group.loc[group[true_col]>0]))
            w_cum_d += np.log(len(group.loc[group[true_col]>0]))
    return w_cum_ndcg / (w_cum_d if w_cum_d > 0 else 1)


def compute_weighted_spearman(dbf, pred_col, true_col, grouper='code'):
    cur_dbf = dbf.copy(deep=True)
    #cur_dbf['code'] = cur_dbf.index.str[:4]
    cur_dbf = cur_dbf[[pred_col, true_col, grouper]]
    
    w_cum_p = 0
    w_cum_d = 0
    for _, group in cur_dbf.groupby(grouper): 
        if len(group) > 1 and not all(group[true_col]==group[true_col].head(1).item()):
            spearman, _ = spearmanr(group[pred_col], group[true_col])
            if np.isnan(spearman):
                spearman=0
            w_cum_p += spearman * np.log(len(group))
            w_cum_d += np.log(len(group))
    return w_cum_p / (w_cum_d if w_cum_d > 0 else 1)


def compute_auprc(dbf, pred_col, true_col, grouper='code'):
    return metrics.average_precision_score(dbf[true_col] > 0, dbf[pred_col])


def compute_t1s(dbf, pred_col, true_col, grouper='code'):
    cur_dbf = dbf.copy(deep=True)
    cur_dbf = cur_dbf[[pred_col, true_col]]
    #cur_dbf['code'] = cur_dbf.index.str[:4]

    t1s = 0
    for _, group in cur_dbf.groupby(grouper): 
        t1s += group.sort_values(pred_col, ascending=False)[true_col].head(1).item()

    mean_t1s = t1s / len(cur_dbf[grouper].unique())
    return mean_t1s


def compute_net_stab(dbf, pred_col, true_col, grouper='code'):
    cur_dbf = dbf.copy(deep=True)
    cur_dbf = cur_dbf[[pred_col, true_col]]
    return cur_dbf.loc[cur_dbf[pred_col] > 0, true_col].sum()


def compute_mean_stab(dbf, pred_col, true_col, grouper='code'):
    cur_dbf = dbf.copy(deep=True)
    cur_dbf = cur_dbf[[pred_col, true_col]]
    return cur_dbf.loc[cur_dbf[pred_col] > 0, true_col].mean()

def compute_recall_k0(dbf, pred_col, true_col, grouper='code'):
    k = 0
    pred_df_discrete_k = dbf.copy(deep=True).drop_duplicates()
    pred_df_discrete_k[true_col] = pred_df_discrete_k[true_col].apply(lambda x: 1 if x > k else 0)
    stable_ct = pred_df_discrete_k[true_col].sum()
    sorted_preds = pred_df_discrete_k.sort_values(pred_col, ascending=False).index
    return pred_df_discrete_k.loc[sorted_preds[:stable_ct], true_col].sum() / stable_ct

def compute_recall_k1(dbf, pred_col, true_col, grouper='code'):
    k = 0
    pred_df_discrete_k = dbf.copy(deep=True).drop_duplicates()
    pred_df_discrete_k[true_col] = pred_df_discrete_k[true_col].apply(lambda x: 1 if x > k else 0)
    stable_ct = pred_df_discrete_k[true_col].sum()
    sorted_preds = pred_df_discrete_k.sort_values(pred_col, ascending=False).index
    return pred_df_discrete_k.loc[sorted_preds[:stable_ct], true_col].sum() / stable_ct

def antisymmetry(fwd, rvs):
    try:
        arr = pd.concat([fwd, rvs], axis=1).dropna()
    except:
        print(fwd)
        print(rvs)
    #if len(arr) != 669:
        #print(len(arr))
    cov = np.cov(arr.T)
    std1 = fwd.dropna().std()
    std2 = rvs.dropna().std()
    return (cov/(std1*std2))[0,1]


def bias(fwd, rvs):
    arr = pd.concat([fwd, rvs], axis=1).dropna()
    #if len(arr) != 669:
        #print(len(arr))
    s = arr.sum(axis=1)
    s2 = s.sum()
    return s2 / (2*len(arr))


def is_combined_model(column_name):
    pattern = r'(\w|-|\.)+ \+ (\w|-|\.)+ \* -?[\d\.]+'
    return bool(re.match(pattern, column_name))


def process_index(index_str):
    # Split the index string by ' + '
    components = index_str.split(' + ')

    # Initialize the model and weight columns
    model1 = None
    weight1 = 1
    model2 = None
    weight2 = 0

    # Process the components
    for component in components:
        model_weight = component.split(' * ')

        if len(model_weight) == 1:
            # Only one model with an implicit weight of 1
            model1 = model_weight[0].strip()
            model2 = model1
        elif len(model_weight) == 2:
            model, weight = model_weight

            if model1 is None:
                model1 = model.strip()
                weight1 = float(weight.strip())
            else:
                model2 = model.strip()
                weight2 = float(weight.strip())

    return model1, weight1, model2, weight2


def parse_cartesian(filename, reduce='mean'):
    # read the file into a dataframe
    try:
        df = pd.read_csv(filename, delim_whitespace=True, header=None)
    except:
        return np.nan
    # we used 3 conformational samples for both the wild-type and mutant
    # taking the lowest-energy structure generally leads to worse results
    
    # take the average of the fourth field for the first 3 lines
    try:
        if reduce == 'mean':
            reduced = df.groupby(2)[3].mean()
        elif reduce == 'min':
            reduced = df.groupby(2)[3].min()
    except:
        print(filename)
        print(df)

    # group means/mins
    wt_red = reduced.loc['WT:']
    try:
        mut_red = reduced.drop('WT:').item()
    except:
        print(reduced)
        print(filename)
        return np.nan

    return float(wt_red - mut_red)


def parse_rosetta_predictions(df, root_name, inverse=False, reduce='mean', kind='cartesian', runtime=False):
    """
    loads results from Rosetta, which are included in the repo
    """
    df_rosetta = pd.DataFrame(columns=[f"cartesian_ddg{'_inv' if inverse else '_dir'}"] if kind=='cartesian' else [f"monomer_ddg{'_inv' if inverse else '_dir'}"])
    df_rosetta_runtimes = pd.DataFrame(columns=['runtime_'+df_rosetta.columns[0]])
    missed = []

    # including in the repo are predictions and runtimes associated with each uid
    for uid in sorted(df.reset_index()['uid'].unique()):
        
        if not inverse:
            pred_path = os.path.join(root_name, uid + '.ddg')
            if runtime:
                rt_path = os.path.join(root_name, 'runtime_' + uid + '.txt')
        else:
            pred_path = os.path.join(root_name, uid + '_inv.ddg')
            if runtime:
                rt_path = os.path.join(root_name, 'runtime_' + uid + '_inv.txt')
        
        if not os.path.exists(pred_path):
            print('Could not find predictions for', uid)
            continue

        df_rosetta.at[uid, f"cartesian_ddg_{'inv' if inverse else 'dir'}"] = parse_cartesian(pred_path, reduce=reduce)
        if runtime:
            df_rosetta_runtimes.at[uid, f"runtime_cartesian_ddg_{'inv' if inverse else 'dir'}"] = int(open(rt_path, 'r').read().strip())          

    return df_rosetta, df_rosetta_runtimes


def make_bar_chart(df, models, title, figsize=(12, 12), xlim=(-1, 1)):
    df['model'] = df['model'].str.replace('_dir', '', regex=False)
    df = df[df['model'].isin(models)]

    mean_cols = [col for col in df.columns if col.endswith('mean')]
    std_cols = [col for col in df.columns if (col.endswith('_std') or col.endswith('stdev'))]
    has_std = len(std_cols) > 0

    n_stats = len(mean_cols)

    bar_width = 0.2
    present_models = [model for model in models if model in df['model'].unique()]
    n_models = len(present_models)
    group_width = bar_width * n_models + bar_width
    positions = np.arange(n_stats) * group_width

    plt.figure(figsize=figsize, dpi=300)
    ax = plt.gca()

    color_dict = get_color_mapping(df, 'model')

    for i, model in enumerate(reversed(present_models)):
        model_means = df[df['model'] == model][mean_cols].values.flatten()
        bar_positions = positions + i * bar_width
        ax.barh(bar_positions, model_means, height=bar_width, label=model, color=color_dict[model])

        if has_std:
            model_stds = df[df['model'] == model][std_cols].values.flatten()
            ax.errorbar(model_means, bar_positions, xerr=model_stds, fmt='none', color='black', capsize=5, linestyle='None')

    plt.yticks(positions + bar_width * (n_models / 2), [c[:-5] for c in mean_cols], fontsize=18)
    plt.xticks(fontsize=18)
    plt.xlabel('Value', fontsize=20)
    plt.title(f'Performance on {title}', fontsize=24)

    handles = [plt.Rectangle((0, 0), 1, 1, color=color_dict[model]) for model in present_models]
    plt.legend(handles, [remap_names.get(c, c) for c in present_models], title='Model', loc='upper left', bbox_to_anchor=(1, 1), fontsize=16)

    plt.xlim(xlim)

    plt.show()


def make_scatter_chart(df, models, title, figsize=(12, 8), ylim=(-1, 1), use_dual_y_axis=False, right_y_lim=(-1, 1), sw=1, scale_y2=False):
    df['model'] = df['model'].str.replace('_dir', '', regex=False)
    df = df[df['model'].isin(models)]
    
    mean_cols = [col for col in df.columns if col.endswith('mean')]
    std_cols = [col for col in df.columns if col.endswith('stdev')]
    
    symbols = ['o', '^', 's', 'D', '*']  # Example symbols for different models

    color_dict = get_color_mapping(df, 'model')  # Assuming this function exists
    
    fig1 = plt.figure(figsize=figsize, dpi=300)
    axes_dimensions = (0.1, 0.1, 0.8, .9)
    ax = fig1.add_axes(axes_dimensions)
    #ax = plt.gca()
    
    if use_dual_y_axis:
        ax2 = ax.twinx()
        ax2.set_ylim(right_y_lim)

    if scale_y2:    
        ax2.set_ylabel('ΔΔG (kcal/mol)', rotation=270, labelpad=20, fontsize=18)# + mean_cols[-1][:-5]
    #else:
    #    ax2.set_ylabel(

    n_stats = len(mean_cols)
    width_per_stat = 1.0 / n_stats  # Calculate the width per statistic
    
    # Define offsets to avoid overlap
    offsets = np.linspace(-width_per_stat * sw, width_per_stat * sw, num=len(models))

    for i, model in enumerate(models):
        model_data = df[df['model'] == model]
        if model_data.empty:
            continue
        
        symbol = symbols[i % len(symbols)]  # Use same symbol for each model across all statistics
        for j, mean_col in enumerate(mean_cols):
            std_col = std_cols[j]  # Corresponding std dev column
            mean_values = model_data[mean_col].values
            std_values = model_data[std_col].values
            x_positions = j + offsets[i]  # Adjust position to avoid overlap
            
            if use_dual_y_axis and j == len(mean_cols) - 1:
                # Plot on secondary axis for the last column
                ax2.errorbar(x_positions, mean_values, yerr=std_values, fmt=symbol,
                             capsize=3, color=color_dict.get(model, 'black'), label=remap_names.get(model, model) if j == 0 else "_nolegend_")
                labels = ax2.get_yticklabels()
                ax2.set_yticklabels(labels, fontsize=18)
            else:
                # Plot on primary axis
                ax.errorbar(x_positions, mean_values, yerr=std_values, fmt=symbol,
                            capsize=3, color=color_dict.get(model, 'black'), label=remap_names.get(model, model) if j == 0 else "_nolegend_")
    
    labels = ax.get_yticklabels()
    ax.set_yticklabels(labels, fontsize=12)
    
    # Correct the ticks and dividers
    ax.set_xticks(np.arange(n_stats))
    ax.set_xticklabels([col[:-5] for col in mean_cols], rotation=45, ha="right", fontsize=18)
    ax.set_xlim(-0.5, n_stats-0.5)
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=18)
    ax.set_ylim(ylim)
    ax.legend(title='', bbox_to_anchor=(1.3, 1), loc='upper left', fontsize=18)
    ax.set_ylabel('Score', fontsize=18)
    
    # Draw zero lines correctly confined to each column
    for j in range(n_stats):
        if use_dual_y_axis and j == n_stats - 1:
            ax2.axhline(y=0, xmin=j/n_stats, xmax=(j+1)/n_stats, linestyle='--', color='red', zorder=1)  # Secondary axis zero line
        else:
            ax.axvline(x=j + 0.5, linestyle='--', color='grey', zorder=0)  # Draw dividers
            ax.axhline(y=0, xmin=j/n_stats, xmax=(j+1)/n_stats, linestyle='--', color='red', zorder=1)  # Red line within each column

    #ax.xlabel('Statistic')
    #ax.ylabel('Score', rotation=270, color='black')
    plt.title(f'Performance on {title}', fontsize=22)

    # Draw the canvas to get the renderer
    #fig1.canvas.draw()

    # Get the bounding box of the axes including tick labels but excluding axes labels and titles
    #ax_bbox = ax.get_tightbbox(fig1.canvas.get_renderer()).transformed(fig1.transFigure.inverted())
    
    # Set a new axes rectangle that is always at the same place regardless of the text size
    #ax.set_position([0.1, 0.1, ax_bbox.width, ax_bbox.height])
    return fig1



def recovery_curves(rcv, models=['cartesian_ddg_dir', 'ddG_dir', 'dTm_dir', 'random_dir'], measurements=('dTm', 'ddG'), plots=('auppc', 'aumsc'), 
    points=[10], left_spacing=0.02, right_spacing=0.02, left_text_offset=(20, 0.07), right_text_offset=(20, 0.07), title='Dataset'):

    def annotate_points(ax, data, x_col, y_col, hue_col, x_values, text_offset=(0, 0), spacing=0.02):
        line_colors = {}
        for line in ax.lines:
            label = line.get_label()
            color = line.get_color()
            line_colors[label] = color

        for x_val in x_values:
            models_and_points = []
            for model, model_data in data.groupby(hue_col):
                value_row = model_data.loc[model_data[x_col] == x_val]
                if not value_row.empty:
                    if len(value_row) > 1:
                        x, y = value_row[x_col].values[0], value_row[y_col].values.mean()
                    else:
                        x, y = value_row[x_col].values[0], value_row[y_col].values[0]
                    models_and_points.append((model, x, y))

            # Sort models_and_points by y values to space them evenly
            models_and_points = sorted(models_and_points, key=lambda x: x[2], reverse=True)

            # Calculate annotation positions and add annotations
            y_annot = max(y for _, _, y in models_and_points) + text_offset[1]
            for model, x, y in models_and_points:
                ax.annotate(f"{y:.2f}", (x, y),
                            xytext=(x + text_offset[0], y_annot),
                            arrowprops=dict(arrowstyle='-', lw=1, color='gray'),
                            fontsize=12, color=line_colors[model])
                y_annot -= spacing
                ax.axvline(x=x, color='r', linestyle='dashed')

    font = {'size'   : 12}
    matplotlib.rc('font', **font)

    if len(plots) == 1:
        if len(measurements) == 1:
            fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(6, 6), dpi=300)
            ax_list = [axes]  
        else:
            fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 6), dpi=300)
            ax_list = [axes[0], axes[1]]        
    elif len(plots)==2:
        if len(measurements) == 1:
            fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 6), dpi=300)
            ax_list = [axes[0], axes[1]]
        else:
            fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 12), dpi=300)
            ax_list = [axes[0, 0], axes[1, 0], axes[0, 1], axes[1, 1]]

    d5 = rcv.reset_index()
    d5 = d5.loc[d5['model'].isin(models)].set_index(['measurement', 'model_type', 'model', 'class'])
    d5 = d5.drop([c for c in d5.columns if 'stab_' in c], axis=1)

    i = 0

    if 'ddG' in measurements:
        # for plotting recovery over thresholds

        if 'auppc' in plots:
            recov = d5[[c for c in d5.columns if '%' in c]].reset_index()
            recov = recov.loc[recov['model']!='dTm_dir']
            #recov = recov.loc[recov['model'].isin(['cartesian_ddg_dir', 'mpnn_20_00_dir', 'msa_transformer_mean_dir', 'esm1v_mean_dir'])]
            recov = recov.loc[recov['measurement']=='ddG']
            recov = recov.drop(['measurement', 'model_type', 'class'], axis=1)
            melted_1 = recov.melt(id_vars='model', value_vars=recov.columns, var_name="variable", value_name="value")
            recov = melted_1
            recov['variable'] = recov['variable'].str.strip('%').astype(float)
            # Sort the DataFrame to move 'random_dir' to the end
            # Create a temporary sorting helper column
            recov['sort_helper'] = recov['model'] == 'random_dir'
            # Sort by the helper column (False entries first, True entries last)
            recov = recov.sort_values(by='sort_helper').drop('sort_helper', axis=1)
            cmap = get_color_mapping(recov, 'model')
            
            for model in models:
                subset = recov[recov['model']==model]
                color = cmap[model]
                ax_ = sns.lineplot(data=subset, x='variable', y='value', ax=ax_list[i], label=model, color=color)
            if model == 'random_dir':
                ax_.lines[-1].set_linestyle('--')

            #ax_ = sns.lineplot(data=recov, x='variable', y='value', hue='model', ax=ax_list[i])
            if len(measurements) > 1:
                ax_list[i].set_xlabel('')
            else:
                ax_list[i].set_xlabel('top x% of ranked predictions')
            #axes[0, 1].set_ylabel('fraction of top mutants identified')
            ax_list[i].set_ylabel('fraction stabilizing (ΔΔG > 1 kcal/mol)')
            #ax_list[i].set_title('ΔΔG')
            annotate_points(ax_list[i], recov, 'variable', 'value', 'model', points, text_offset=left_text_offset, spacing=left_spacing/2)
            i += 1

        if 'aumsc' in plots:
            recov = d5[[c for c in d5.columns if '$' in c]].reset_index()
            recov = recov.loc[recov['model']!='dTm_dir']
            #recov = recov.loc[recov['model'].isin(['cartesian_ddg_dir', 'mpnn_20_00_dir', 'msa_transformer_mean_dir', 'esm1v_mean_dir'])]
            recov = recov.loc[recov['measurement']=='ddG']
            recov = recov.drop(['measurement', 'model_type', 'class'], axis=1)
            recov = recov.melt(id_vars='model')
            recov['variable'] = recov['variable'].str.strip('$').astype(float)
            # Sort the DataFrame to move 'random_dir' to the end
            # Create a temporary sorting helper column
            recov['sort_helper'] = recov['model'] == 'random_dir'
            # Sort by the helper column (False entries first, True entries last)
            recov = recov.sort_values(by='sort_helper').drop('sort_helper', axis=1)
            cmap = get_color_mapping(recov, 'model')
            if not cmap:
                cmap = get_color_mapping(recov, 'model')
                
            for model in models:
                subset = recov[recov['model']==model]
                color = cmap[model]
                ax_ = sns.lineplot(data=subset, x='variable', y='value', ax=ax_list[i], label=model, color=color)
            if model == 'random_dir':
                ax_.lines[-1].set_linestyle('--')
            #ax_ = sns.lineplot(data=recov, x='variable', y='value', hue='model', ax=ax_list[i])
            ax_list[i].set_xlabel('top x% of ranked predictions')
            ax_list[i].set_ylabel('mean stabilizition (kcal/mol)')
            annotate_points(ax_list[i], recov, 'variable', 'value', 'model', points, text_offset=right_text_offset, spacing=right_spacing*3)
            i += 1

    if 'dTm' in measurements:
        if len(ax_list) == 4:
            i = 2
        else:
            i = 0

        if 'auppc' in plots:
            recov = d5[[c for c in d5.columns if '%' in c]].reset_index()
            recov = recov.loc[recov['model']!='ddG_dir']
            #recov = recov.loc[recov['model'].isin(['cartesian_ddg_dir', 'mpnn_20_00_dir', 'msa_transformer_mean_dir', 'esm1v_mean_dir'])]
            recov = recov.loc[recov['measurement']=='dTm']
            recov = recov.drop(['measurement', 'model_type', 'class'], axis=1)
            recov = recov.melt(id_vars='model')
            recov['variable'] = recov['variable'].str.strip('%').astype(float)
            sns.lineplot(data=recov, x='variable', y='value', hue='model', ax=ax_list[i])
            ax_list[i].set_ylabel('fraction stabilizing (ΔTm > 1K)')
            #axes[0, 1].set_ylabel('fraction of top mutants identified')
            ax_list[i].set_title('ΔTm') #measurement_ = {'ddG': 'ΔΔG', 'dTm': 'ΔTm'}[measurement]
            annotate_points(ax_list[i], recov, 'variable', 'value', 'model', points, text_offset=left_text_offset, spacing=left_spacing/2)
            i += 1

        if 'aumsc' in plots:
            recov = d5[[c for c in d5.columns if '$' in c]].reset_index()
            recov = recov.loc[recov['model']!='ddG_dir']
            #recov = recov.loc[recov['model'].isin(['cartesian_ddg_dir', 'mpnn_20_00_dir', 'msa_transformer_mean_dir', 'esm1v_mean_dir'])]
            recov = recov.loc[recov['measurement']=='dTm']
            recov = recov.drop(['measurement', 'model_type', 'class'], axis=1)
            recov = recov.melt(id_vars='model')
            recov['variable'] = recov['variable'].str.strip('$y').astype(float)
            sns.lineplot(data=recov, x='variable', y='value', hue='model', ax=ax_list[i])
            ax_list[i].set_xlabel('top x% of ranked predictions')
            #axes[1, 1].set_ylabel('fraction of stablizing mutants recovered')
            ax_list[i].set_ylabel('mean stabilizition (deg. K)')
            annotate_points(ax_list[i], recov, 'variable', 'value', 'model', points, text_offset=right_text_offset, spacing=right_spacing*12)
            i += 1

    handles, labels = ax_list[0].get_legend_handles_labels()
    if len(ax_list) > 1:
        for ax in axes.flat:
            ax.get_legend().remove()
            ax.set_title(title)
    else:
        ax_list[0].get_legend().remove()

    #labels[labels.index('ddG_dir')] = 'Ground truth label'
    labels = [remap_names_2[name] if name in remap_names_2.keys() else name for name in labels]

    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, -0.), ncol=2)
    plt.tight_layout()

    #sp_labels = ['(a)', '(b)', '(c)', '(d)']

    #if len(ax_list) > 1:
    #    for i, ax in enumerate(axes.flat):
    #        ax.text(-0.05, 1.05, sp_labels[i], transform=ax.transAxes, fontsize=16, va='top')

    plt.show()
    return fig


def recovery_curves_2(rcv, models, percentile_labels, directions=['dir', 'inv']):

    def annotate_points(ax, data, x_col, y_col, hue_col, x_values, text_offset=(0, 0), spacing=0.02):
        line_colors = {}

        for line in ax.lines:
            label = line.get_label()
            color = line.get_color()
            line_colors[label] = color

        for x_val in x_values:
            models_and_points = []
            for model, model_data in data.groupby(hue_col):
                value_row = model_data.loc[model_data[x_col] == x_val]
                if not value_row.empty:
                    x, y = value_row[x_col].values[0], value_row[y_col].values[0]
                    models_and_points.append((model, x, y))

            # Sort models_and_points by y values to space them evenly
            models_and_points = sorted(models_and_points, key=lambda x: x[2], reverse=True)

            # Calculate annotation positions and add annotations
            y_annot = max(y for _, _, y in models_and_points) - text_offset[1]
            for model, x, y in models_and_points:
                ax.annotate(f"{y:.2f}", (x, y),
                            xytext=(x - text_offset[0], y_annot),
                            arrowprops=dict(arrowstyle='-', lw=1, color='gray'),
                            fontsize=12, color=line_colors[model])
                y_annot -= spacing
                ax.axvline(x=x, color='r', linestyle='dashed')

    font = {'size'   : 15}
    matplotlib.rc('font', **font)

    if len(directions) == 1:
        nrows = 1
        ncols = 2
    else:
        nrows = 2
        ncols = 3
        directions += ['combined']

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(20, 12))

    d5 = rcv.reset_index()
    d5 = d5.loc[d5['model'].isin([m+'_dir' for m in models] + [m+'_inv' for m in models] + models)].set_index(['direction', 'model_type', 'model', 'class'])

    if len(directions) == 1:
        plot_locations = [0, 1]
    else:
        plot_locations = [(0, 0), (1, 0), (0, 1), (1, 1), (0, 2), (1, 2)]

    plot_no = 0

    dirs = {'dir': 'Direct', 'inv': 'Inverse', 'combined': 'Both Directions'}

    for direction in directions:

        recov = d5[[c for c in d5.columns if '%' in c]].reset_index()
        recov = recov.loc[recov['direction']==direction]
        recov = recov.drop(['direction', 'model_type', 'class'], axis=1)
        recov = recov.melt(id_vars='model')

        recov = recov.loc[~recov['variable'].str.contains('pos_')]
        recov = recov.loc[~recov['variable'].str.contains('stab_')]
        recov['variable'] = recov['variable'].str.strip('%').astype(float)

        sns.lineplot(data=recov, x='variable', y='value', hue='model', ax=axes[plot_locations[plot_no]])
        axes[plot_locations[plot_no]].set_xlabel('percentile of predictions assessed')
        axes[plot_locations[plot_no]].set_ylabel('fraction stabilizing (ΔΔG > 0)')
        axes[plot_locations[plot_no]].set_title(dirs[direction])

        if direction == 'dir':
            text_offset = (20, 0)
            spacing = 0.06
        elif direction == 'inv':
            text_offset = (40, 0.12)
            spacing = 0.01
        else:
            text_offset = (20, 0.2)
            spacing = 0.04
        annotate_points(axes[plot_locations[plot_no]], recov, 'variable', 'value', 'model', percentile_labels, text_offset=text_offset, spacing=spacing)

        plot_no += 1

        recov = d5[[c for c in d5.columns if '$' in c]].reset_index()
        recov = recov.loc[recov['direction']==direction]
        recov = recov.drop(['direction', 'model_type', 'class'], axis=1)
        recov = recov.melt(id_vars='model')

        recov = recov.loc[~recov['variable'].str.contains('pos_$_')]
        recov['variable'] = recov['variable'].str.strip('$').astype(float)

        sns.lineplot(data=recov, x='variable', y='value', hue='model', ax=axes[plot_locations[plot_no]])
        axes[plot_locations[plot_no]].set_xlabel('percentile of predictions assessed')
        axes[plot_locations[plot_no]].set_ylabel('mean stabilization (kcal/mol)')
        axes[plot_locations[plot_no]].set_title(dirs[direction])

        if direction == 'dir':
            text_offset = (30, -0.65)
            spacing = 0.15
        elif direction == 'inv':
            text_offset = (25, -1.4)
            spacing = 0.25
        else:
            text_offset = (25, -1.5)
            spacing = 0.25
        annotate_points(axes[plot_locations[plot_no]], recov, 'variable', 'value', 'model', percentile_labels, text_offset=text_offset, spacing=spacing)
        
        if len(directions)==1:
            axes[plot_locations[plot_no]].set_title(dirs[direction])

        plot_no += 1

    handles, labels = axes[plot_locations[0]].get_legend_handles_labels()
    if 'ddG' in labels:
        labels[labels.index('ddG')] = 'Ground truth label'
    if 'random' in labels:
        labels[labels.index('random')] = 'Random Noise'
    if 'cartesian_ddg + mpnn_mean * 0.5' in labels:
        labels[labels.index('cartesian_ddg + mpnn_mean * 0.5')] = 'CartDDG + ProteinMPNN_mean * 0.5'
    labels = [remap_names[name] if name in remap_names.keys() else name for name in labels]

    for ax in axes.flat:
        ax.get_legend().remove()

    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.2), ncol=2)
    plt.tight_layout()
    plt.show()
    return


def correlations(db_gt_preds, dbr, score_name, score_name_2=None, min_obs=5, bins=20, stat='spearman', runtime=False, 
                 group=True, valid=False, out=True, plot=False, coeff2=0.2, meas='ddG', measurements=['ddG'], marker_name=None,
                 saveloc_comp='../data/extended/figure_data/data_comp.csv'):

    font = {'size'   : 12}
    matplotlib.rc('font', **font)
    
    # the first section is for testing combinations of models on the spot by making a custom score name
    df = db_gt_preds.copy(deep=True).reset_index()
    pattern = r"^(\w+) \+ (\w+) \* ([\d\.]+)$"
    # Use regex to match the pattern in the sample string
    match = re.match(pattern, score_name)
    
    if match and create:
        # Extract the parsed values from the regex match
        item_1 = match.group(1)
        item_2 = match.group(2)
        weight = float(match.group(3))
        combo = True

        assert item_1 in df.columns
        assert item_2 in df.columns
        if match.group(0) not in df.columns:
            df[score_name] = df[item_1] + df[item_2] * weight
            dbr[f'runtime_{score_name}'] = dbr[f'runtime_{item_1}'] + dbr[f'runtime_{item_2}']

    if score_name_2==None:
        score_name_2 = 'tmp'
        df[score_name_2]=0
        dbr[f'runtime_{score_name_2}']=0

    melted = df.melt(id_vars=['uid'], value_vars=measurements).dropna().rename({'variable':'type', 'value':'measurement'}, axis=1)
    df = melted.set_index('uid').join(df[['uid', 'code', marker_name, score_name, score_name_2]].set_index('uid'))
    if runtime and score_name not in measurements + ['random_dir']:
        dbr = melted.set_index('uid').join(dbr[[f'runtime_{score_name}', f'runtime_{score_name_2}', 'code']])

    #todo: match colours between plots, add legends, add measurement distribution
    #if plot:
    #    fig, axs = plt.subplots(2, 2, figsize=(15, 15))
    #    fig.suptitle(score_name)

    g = df[['code', 'type', marker_name, score_name, score_name_2, 'measurement']].dropna()
    g[f'{score_name} + {coeff2} * {score_name_2}'] = g[score_name] + coeff2 * g[score_name_2]

    if group:   
        if stat == 'spearman':
            i = pd.DataFrame()
            for (code, t, p), group in g.groupby(['code', 'type', marker_name]):
                if len(group) > 1 and not all(group['measurement']==group['measurement'][0]):
                    ndcg1, _ = spearmanr(group['measurement'], group[score_name])
                    ndcg2, _ = spearmanr(group['measurement'], group[score_name_2])
                    tmp = pd.DataFrame([code, len(group), t, ndcg1, ndcg2, p]).T
                    i = pd.concat([i, tmp])
            i.columns=['code', 'n mutants', 'type', score_name, score_name_2, marker_name]
            i = i.set_index(['code', 'n mutants', 'type'])
            ungrouped = pd.DataFrame()
            for t, group in g.groupby('type'):
                ug1, _ = spearmanr(group['measurement'], group[score_name])
                ug2, _ = spearmanr(group['measurement'], group[score_name_2])
                tmp = pd.DataFrame([len(group), t, ug1, ug2]).T
                ungrouped = pd.concat([ungrouped, tmp])
            ungrouped.columns=['n mutants', 'type', score_name, score_name_2]
            ungrouped = ungrouped.set_index('type')
        if stat == 'pearson':
            i = pd.DataFrame()
            for (code, t, p), group in g.groupby(['code', 'type', marker_name]):
                if len(group) > 1 and not all(group['measurement']==group['measurement'][0]):
                    ndcg1, _ = pearsonr(group['measurement'], group[score_name])
                    ndcg2, _ = pearsonr(group['measurement'], group[score_name_2])
                    tmp = pd.DataFrame([code, len(group), t, ndcg1, ndcg2, p]).T
                    i = pd.concat([i, tmp])
            i.columns=['code', 'n mutants', 'type', score_name, score_name_2, marker_name]
            i = i.set_index(['code', 'n mutants', 'type'])
            ungrouped = pd.DataFrame()
            for t, group in g.groupby('type'):
                ug1, _ = pearsonr(group['measurement'], group[score_name])
                ug2, _ = pearsonr(group['measurement'], group[score_name_2])
                tmp = pd.DataFrame([len(group), t, ug1, ug2]).T
                ungrouped = pd.concat([ungrouped, tmp])
            ungrouped.columns=['n mutants', 'type', score_name, score_name_2]
            ungrouped = ungrouped.set_index('type')
        elif stat == 'ndcg':
            i = pd.DataFrame()
            for (code, t, p), group in g.groupby(['code', 'type', marker_name]):
                if len(group) > 1 and not all(group['measurement']==group['measurement'][0]):
                    ndcg1 = compute_ndcg(group, score_name, 'measurement')
                    ndcg2 = compute_ndcg(group, score_name_2, 'measurement')
                    tmp = pd.DataFrame([code, len(group), t, ndcg1, ndcg2, p]).T
                    i = pd.concat([i, tmp])
            i.columns=['code', 'n mutants', 'type', score_name, score_name_2, marker_name]
            i = i.set_index(['code', 'n mutants', 'type'])
            ungrouped = pd.DataFrame()
            for t, group in g.groupby('type'):
                ug1 = compute_ndcg(group, score_name, 'measurement')
                ug2 = compute_ndcg(group, score_name_2, 'measurement')
                tmp = pd.DataFrame([len(group), t, ug1, ug2]).T
                ungrouped = pd.concat([ungrouped, tmp])
            ungrouped.columns=['n mutants', 'type', score_name, score_name_2]
            ungrouped = ungrouped.set_index('type')

        #if plot:
        #    axs[0,0].set_title(f'{stat} to ground truth')
        #    axs[0,1].set_title(f'distribution of predictions')
        #    sns.histplot(ax=axs[0, 1], x=score_name, data=g[[score_name, 'type']].reset_index(drop=True), alpha=0.3, hue='type', bins=bins)
        #    sns.histplot(ax=axs[1, 0], x='measurement', data=g[['measurement', 'type']].reset_index(drop=True), alpha=0.3, hue='type', bins=bins)
        if runtime and score_name not in measurements + ['random_dir']:
            runs = dbr[[f'runtime_{score_name}', f'runtime_{score_name_2}', 'code', 'type']].groupby(['code', 'type']).sum().reset_index()

    #else:
    #    f = g.loc[g['type']=='ddG', [score_name, score_name_2, 'measurement']]
    #    n = len(f)
    #    i = f.corr('spearman')[['measurement', score_name]].drop('measurement').T
    #    i['n mutants'] = n
    #    if score_name_2 == 'tmp':
    #        f = f.drop('tmp', axis=1)
    #    if plot:
    #        sns.scatterplot(ax=axs[0,0], data=f, x=score_name, y=score_name_2, hue='measurement', alpha=0.3, palette='coolwarm_r')
    #        sns.scatterplot(ax=axs[0,1], data=f, x=score_name_2, y='measurement', hue=score_name, alpha=0.3, palette='coolwarm_r')
    #        sns.scatterplot(ax=axs[1,0], data=f, x=score_name, y='measurement', hue=score_name_2, alpha=0.3, palette='coolwarm_r')
    #        sns.scatterplot(ax=axs[1,1], data=f, x=f'{score_name} + {coeff2} * {score_name_2}', y='measurement', alpha=0.3, legend=None)
    #        plt.tight_layout(pad=1, w_pad=0.5, h_pad=1.0)
    #        
    #    return i
        
    if plot:
        data=i.reset_index()
        data_out = data.copy(deep=True)
        data_out.columns = [remap_names_2.get(c,c) for c in data_out.columns]
        data_out = data_out.drop('type', axis=1)
        data_out.to_csv(saveloc_comp)

        if stat == 'ndcg':
            data[score_name] = 100**data[score_name].astype(float)
            data[score_name_2] = 100**data[score_name_2].astype(float)
        #print(data.loc[data[score_name]!=data[score_name_2]])
        
        #sns.histplot(ax=axs[0,0], x=score_name, data=data[[score_name, 'type']].sort_values('type'), alpha=0.3, hue='type', bins=bins, stat='count', kde=True)
        #axs[0,0].set_xlim((-1, 1))
        #sns.scatterplot(ax=axs[1,1], data=g, x=score_name, y='measurement', hue='type', alpha=0.3)
        g = sns.jointplot(data=data, x=score_name_2, y=score_name, hue='type', kind='hist', marginal_kws=dict(bins=20), joint_kws=dict(alpha=0), height=8)
        g.fig.set_dpi(300)
        min_size = data['n mutants'].min() * 5
        max_size = data['n mutants'].max() * 5
        for code, row in data.reset_index().iterrows():
            if (row['n mutants'] > min_obs):# and (row['code'] not in ('1RTB', '1BVC', '1RN1', '1BNI', '1BPI', '1HZ6', '1OTR', '2O9P', '1AJ3', '3VUB', '1LZ1')):# \
            #if row['code'] in ['4E5K', '3D2A', '1ZNJ', '1WQ5', '1UHG', '1TUP', '1STN', '1QLP', '1PGA']:
                g.ax_joint.text(row[score_name_2]-0.01, row[score_name]-0.01, f"{row['code']}:{row['n mutants']}", size=6)
        ax = sns.scatterplot(data=data, x=score_name_2, y=score_name, hue=marker_name, size='n mutants', style=marker_name, sizes=(min_size,max_size), ax=g.ax_joint, alpha=0.6)#,
                            #markers={True: "s", False: "o"})
        small = np.array(i[[score_name_2, score_name]].dropna()).min().min()
        big = np.array(data[[score_name_2, score_name]].dropna()).max().max()
        sns.lineplot(data=pd.DataFrame({'x': np.arange(small, big, 0.01), 'y': np.arange(small, big, 0.01)}), x='x', y='y', ax=g.ax_joint, color='red')
        plt.tight_layout(pad=1, w_pad=0.5, h_pad=1.0)

        if stat == 'spearman':
            ax.set_xlabel(f'Spearman of {remap_names[ax.get_xlabel()[:-4]]}')
            ax.set_ylabel(f'Spearman of {remap_names[ax.get_ylabel()[:-4]]}')
        elif stat == 'ndcg':
            ax.set_xlabel(f'100^(NDCG) of {remap_names[ax.get_xlabel()[:-4]]}')
            ax.set_ylabel(f'100^(NDCG) of {remap_names[ax.get_ylabel()[:-4]]}')
        handles, labels = g.ax_joint.get_legend_handles_labels()

        for pos in [4,6,8,10,12]:
            # Assuming you want to add space after the first entry
            space_handle = mpatches.Patch(color='none', label='')  # Create an invisible handle
            handles.insert(pos, space_handle)  # Insert at the position after which you want extra space
            labels.insert(pos, '')  # Corresponding empty label


        legend = g.ax_joint.legend(handles, labels, loc='upper left', ncol=1, prop={'size': 10}, bbox_to_anchor=(-0.5, 1), labelspacing=1.2)
        #legend.get_frame().set_alpha(0.) 

        plt.show()
        
    if out:   

        df_out = pd.DataFrame(index=pd.MultiIndex.from_product([measurements, ['n_total', f'ungrouped_{stat}', 'n_proteins', f'n_proteins_{stat}', f'avg_{stat}', f'weighted_{stat}', 'runtime (s)']]),
                               columns=[score_name, score_name_2] if score_name_2 != 'tmp' else [score_name], dtype=object) 
        
        for t in measurements:
            
            reduced = i.reset_index()
            reduced = reduced.loc[reduced['type']==t]
            reduced['n mutants'] = reduced['n mutants'].astype(int)
            if runtime and score_name not in measurements + ['random_dir']:
                runs_reduced = runs.loc[runs['type']==t]

            for score in [score_name, score_name_2]:
                if score != 'tmp':
                    df_out.at[(t, 'n_total'), score] = ungrouped.at[t, 'n mutants']
                    df_out.at[(t, f'ungrouped_{stat}'), score] = ungrouped.at[t, score]
                    df_out.at[(t, f'n_proteins'), score] = len(db_gt_preds[['code', score_name, t]].dropna().groupby('code').first())
                    df_out.at[(t, f'n_proteins_{stat}'), score] = int(len(reduced.loc[reduced['n mutants']>=min_obs]))
                    df_out.at[(t, f'avg_{stat}'), score] = reduced[score].mean()
                    df_out.at[(t, f'weighted_{stat}'), score] = np.average(reduced[score], weights=np.log(reduced['n mutants']))
                    if runtime and score_name not in measurements + ['random_dir']:
                        df_out.at[(t, 'runtime (s)'), score] = runs_reduced[f'runtime_{score}'].sum()
        return df_out, g

    return i, g


def correlations_2(db_complete, score_name, score_name_2=None, min_obs=5, bins=20, corr='spearman', out=True, plot=False, direction='dir', highlight=[], annotate=True, color_col=None, scale=1, th=0.0):
    print(score_name)
    dbf = db_complete.copy(deep=True)

    if score_name_2==None:
        score_name_2 = 'tmp'
        dbf[score_name_2]=0

    if direction == 'combined':
        ddg = 'ddG'
        cols = ['code', score_name, score_name_2, ddg]
        cols = list(set(cols))
        g = dbf[cols].dropna()

    else:
        ddg = f'ddG_{direction}'
        cols = ['code', score_name, score_name_2, ddg]
        cols = list(set(cols))
        g = dbf[cols].dropna()
        new_score_name = score_name.replace('_dir', '').replace('_inv', '')
        g = g.rename({score_name: new_score_name, ddg: 'ddG'}, axis=1)
        dbf = dbf.rename({score_name: new_score_name, ddg: 'ddG'}, axis=1)
        score_name = new_score_name
        ddg = 'ddG'

        if score_name_2 != 'tmp':
            g = g.rename({score_name_2: score_name_2[:-4]}, axis=1)
            dbf = dbf.rename({score_name_2: score_name_2[:-4]}, axis=1)
            score_name_2 = score_name_2[:-4]          

    if plot:
        fig, axs = plt.subplots(1, 2, figsize=(40, 15))
        fig.suptitle(score_name)
        axs[1].set_title(f'distribution of {score_name} predictions')

        sns.histplot(ax=axs[1], data=dbf[score_name], alpha=0.3, color='red', bins=bins)

    g[score_name] = g[score_name].astype(float)
    g[score_name_2] = g[score_name_2].astype(float)

    # get the correlation of each set of predictions, for each protein, to the ground truth
    f = g.groupby('code').corr(corr)[[ddg]]
    h = g.groupby('code').count().iloc[:, -1]
    h.name = 'n_muts'
    f = f.join(h).reset_index()
    f.columns = ['code', 'model', 'corr', 'n_muts']
    if score_name != ddg:
        f = f.loc[f['model']!=ddg]
    i = f.loc[f['n_muts']>=min_obs].pivot_table(index=['code', 'n_muts'], columns=['model'], values='corr')

    if color_col is not None:
        to_join = dbf[['code', color_col]]
        i = i.reset_index().set_index('code').join(to_join.set_index('code')).reset_index().set_index(['code', 'n_muts']).drop_duplicates()
        print(i)

    ungrouped = g.corr(corr)[[ddg]]#.drop(ddg)
    ungrouped['n_total'] = len(g)
    if score_name_2 == 'tmp':
        ungrouped = ungrouped.drop('tmp')

    #if plot:
    #    axs[0].set_title(f'{score_name} to ground truth')
    #    sns.scatterplot(ax=axs[0], data=g, x=score_name, y='ddG')
    #return(f)  
    
    label_shift = 0.1
    if plot:
        axs[0].set_title(f'distribution of {corr} correlations to ground truth')
        sns.histplot(ax=axs[0],data=i[score_name], alpha=0.3, color='orange', bins=bins, stat='count', kde=True)
        axs[0].set_xlim((-1, 1))
        plt.show()
        data=i
        g = sns.jointplot(data=data, x=score_name_2, y=score_name, kind='kde', # marginal_kws=dict(bins=20),
                          joint_kws=dict(alpha=0), height=30*scale, hue=color_col)
        
        if annotate:
            threshold = 0.005
            for idx, row in data.reset_index().iterrows():
                tmp_data = data.reset_index().drop(idx).loc[(data.reset_index().drop(idx)[score_name]-row[score_name])**2 < threshold]
                tmp_data_2 = tmp_data.loc[(tmp_data[score_name_2]-row[score_name_2])**2 < threshold]
                dx = random.uniform(-label_shift, label_shift)
                dx += 0.01
                dy = random.uniform(-label_shift, label_shift)
                dy += 0.01
                if len(tmp_data_2) > 0:
                    g.ax_joint.text(row[score_name_2]+dx, row[score_name]+dy, f"{row['code']}:{row['n_muts']}", size=4*5, color='r' if row['code'] in highlight else 'black')
                    g.ax_joint.plot([row[score_name_2], row[score_name_2]+dx], [row[score_name], row[score_name]+dy])
                else:
                    g.ax_joint.text(row[score_name_2]+0.01, row[score_name]+0.01, f"{row['code']}:{row['n_muts']}", size=4*5, color='r' if row['code'] in highlight else 'black')
                    g.ax_joint.plot([row[score_name_2], row[score_name_2]+dx], [row[score_name], row[score_name]+dy])

        if th > 0:
            for idx, row in data.reset_index().iterrows():
                if np.abs(row[score_name_2]-row[score_name]) > th:
                    flag = True
                else:
                    flag = False
                if th > 0 and flag:
                    dx = random.uniform(-label_shift, label_shift)
                    dx += 0.01
                    dy = random.uniform(-label_shift, label_shift)
                    dy += 0.01
                    g.ax_joint.plot([row[score_name_2], row[score_name_2]+dx], [row[score_name], row[score_name]+dy])
                    g.ax_joint.text(row[score_name_2]+dx, row[score_name]+dy, f"{row['code']}:{row['n_muts']}", size=4*5, color='g')

        sns.scatterplot(data=data, x=score_name_2, y=score_name, size='n_muts', sizes=(2*100*scale,70*100*scale), ax=g.ax_joint, alpha=0.4, hue=color_col)
        small = np.array(i[[score_name_2, score_name]].dropna()).min().min()
        sns.lineplot(data=pd.DataFrame({'x': np.arange(small, 1, 0.01), 'y': np.arange(small, 1, 0.01)}), 
                        x='x', y='y', ax=g.ax_joint, color='red')
        plt.show()
        
    if out:
        df_out = pd.DataFrame(columns=[score_name, score_name_2] if score_name_2 != 'tmp' else [score_name], dtype=float) 
        
        reduced = i.reset_index()

        for score in [score_name] + ([score_name_2] if score_name_2 != 'tmp' else []):
            df_out.at[f'n_total', score] = ungrouped.at[score, 'n_total']
            df_out.at[f'n_proteins', score] = len(reduced.loc[reduced['n_muts']>=min_obs])
            df_out.at[f'avg_{corr}', score] = reduced[score].mean()
            df_out.at[f'weighted_{corr}', score] = np.average(reduced[score], weights=np.log(reduced['n_muts']))
            df_out.at[f'ungrouped_{corr}', score] = ungrouped.at[score, ddg]

        #for col in [d for d in df_out.columns if '_inv' in d]:
        #    df_out.loc[[f'avg_{corr}', f'weighted_{corr}', f'ungrouped_{corr}'], col]
        df_out = df_out.T.reset_index().rename({'index': 'model'}, axis=1)
        df_out['direction'] = direction
        df_out = df_out.set_index(['direction', 'model'])

        return df_out
    
    return i

def calculate_ppc(group, pred_col, percentile_values, meas='ddG', threshold=1):
    result = {}
    ground_truth = set(group.loc[group[meas] > threshold].index)
    sorted_predictions = group.sort_values(pred_col, ascending=False)
    
    for p in percentile_values:
        k = (p - 100) / 100
        l = int(len(group) * k)
        kth_prediction = set(sorted_predictions.head(l).index)
        result[f"{p}%"] = len(ground_truth.intersection(kth_prediction))
        result[f"pos_{p}%"] = len(kth_prediction)
    
    return pd.Series(result)

def calculate_msc(group, pred_col, percentile_values, meas='ddG'):
    result = {}
    sorted_predictions = group.sort_values(pred_col, ascending=False)
    
    for p in percentile_values:
        k = (p - 100) / 100
        l = int(len(group) * k)
        kth_prediction = list(set(sorted_predictions.head(l).index))
        result[f"{p}$"] = group.loc[kth_prediction, meas].sum()
        result[f"pos_{p}$"] = len(kth_prediction)
    
    return pd.Series(result)

def compute_stats(
    df, 
    split_col=None, split_val=None, split_col_2=None, split_val_2=None, 
    measurements=('ddG', 'dTm'), stats=(), n_classes=2, quiet=False, 
    grouper=('code'), n_bootstraps=-1, split_first=True, split_last=True,
    ):
    """
    Computes all per-protein and per-dataset stats, including when splitting
    into more than one feature-based scaffold. Splitting is done by specifying
    split_cols (the feature names) and split_vals (the threshold for splitting
    on the respective features). Specifying only split_col and split_val will
    create two scaffolds. Specifying only split_col with split_val > 
    split_val_2 will create 3 scaffolds, with high, intermediate and low values.
    Specifying different split_col and split_col_2 will create 4 scaffolds
    based on high and low values of 2 features. You can pass in a tuple of stats
    to only calculate a subset of the possible stats. You can use n_classes=3
    to eliminate the near-neutral mutations.
    """
    assert (split_first or split_last)
    if n_bootstraps > 0:
        dbs_bs = bootstrap_by_grouper(df, n_bootstraps, grouper=grouper, drop=False)
    else:
        dbs_bs = [df]
    dfs_out = []

    for db_gt_preds in tqdm(dbs_bs) if not quiet else dbs_bs:
        #db_gt_preds.to_csv('test.csv')
        split_col_ = split_col
        split_col_2_ = split_col_2

        # make sure to not accidentally modify the input
        db_internal = db_gt_preds.copy(deep=True)
        if grouper is not None:
            index_names = db_internal.index.names
            if index_names == [None]:
                db_internal.index.name = 'uid_sym'
                index_names = ['uid_sym']
            db_grouper = db_internal[grouper].reset_index().drop_duplicates()
            db_grouper = db_grouper.set_index(index_names)
            db_internal = db_internal.drop(grouper, axis=1)
        # currently, grouper cant be None
        else:
            db_grouper = db_internal[[]]

        # eliminate the neutral mutations
        if n_classes == 3:
            db_internal = db_internal.loc[
                ~((db_internal['ddG'] > -1) & (db_internal['ddG'] < 1))
                ]
            if 'dTm' in db_internal.columns:
                db_internal = db_internal.loc[
                    ~((db_internal['dTm'] > -2) & (db_internal['dTm'] < 2))
                    ]

        # case where there are two split_vals on the same column
        if split_col_2_ is None and split_val_2 is not None:
            split_col_2_ = split_col_
        # case where there is no split (default)
        if (split_col_ is None) or (split_val is None):
            split_col_ = 'tmp'
            split_val = 0
            db_internal['tmp'] = -1
        # case where there is only one split (2 scaffolds)
        if (split_col_2_ is None) or (split_val_2 is None):
            split_col_2_ = 'tmp2'
            split_val_2 = 0
            db_internal['tmp2'] = -1

        #print(db_internal)
        # there may be missing features for some entries
        db_internal = db_internal.dropna(subset=[split_col_, split_col_2_])

        # db_discrete will change the continuous measurements into binary labels
        db_discrete = db_internal.copy(deep=True)
        
        # default case
        # stability threshold is defined exactly at 0 kcal/mol or deg. K
        if n_classes == 2:
            if 'ddG' in measurements:
                db_discrete.loc[db_discrete['ddG'] > 0, 'ddG'] = 1
                db_discrete.loc[db_discrete['ddG'] < 0, 'ddG'] = 0
            if 'dTm' in measurements:
                db_discrete.loc[db_discrete['dTm'] > 0, 'dTm'] = 1
                db_discrete.loc[db_discrete['dTm'] < 0, 'dTm'] = 0

        # stabilizing mutations now need to be >= 1 kcal/mol or deg. K
        elif n_classes == 3:
            if 'ddG' in measurements:
                db_discrete.loc[db_discrete['ddG'] > 1, 'ddG'] = 1
                db_discrete.loc[db_discrete['ddG'] < -1, 'ddG'] = -1
            if 'dTm' in measurements:
                db_discrete.loc[db_discrete['dTm'] >= 2, 'dTm'] = 1
                db_discrete.loc[db_discrete['dTm'] <= -2, 'dTm'] = -1

        # for creating a multi-index later
        cols = db_discrete.columns.drop(measurements + [split_col_, split_col_2_])
        
        # db_discrete_bin has discrete labels and binarized (discrete) predictions
        # drop the split_col_s so they do not get binarized
        db_discrete_bin = db_discrete.copy(deep=True).drop(
            [split_col_, split_col_2_], axis=1).astype(float)

        # binarize predictions (>0 stabilizing, assigned positive prediction)
        db_discrete_bin[db_discrete_bin > 0] = 1
        db_discrete_bin[db_discrete_bin < 0] = 0

        # retrieve the original split_col_(s)
        db_discrete_new = db_discrete[
            [split_col_] + ([split_col_2_] if split_col_2_ != split_col_ else [])]
        # make sure the indices align
        assert all(db_discrete_new.index == db_discrete_bin.index)
        # reunite with split_col_s
        db_discrete_bin = pd.concat([db_discrete_bin, db_discrete_new], axis=1)

        # create labels to assign to different scaffolds
        # case no split
        if split_col_ == 'tmp' and split_col_2_ == 'tmp2':
            split = ['']
        # case only one split col
        elif split_col_2_ == 'tmp2':
            split = [f'{split_col_} <= {split_val}', f'{split_col_} > {split_val}',]
        # case 2 splits on same col
        elif split_col_ == split_col_2_:
            split = [f'{split_col_} <= {split_val_2}',
                     f'{split_val} >= {split_col_} > {split_val_2}', 
                     f'{split_col_} > {split_val}']
        # case 3 splits total
        elif split_last == False:
            split = [f'{split_col_} <= {split_val} & {split_col_2_} <= {split_val_2}',
                     f'{split_col_} > {split_val} & {split_col_2_} <= {split_val_2}',
                     f'{split_col_2_} > {split_val_2}']
        # case 3 splits total
        elif split_first == False:
            split = [f'{split_col_} <= {split_val} & {split_col_2_} <= {split_val_2}',
                     f'{split_col_} <= {split_val} & {split_col_2_} > {split_val_2}',
                     f'{split_col_} > {split_val}']
        # case 2 splits on 2 cols
        else:
            split = [f'{split_col_} <= {split_val} & {split_col_2_} <= {split_val_2}',
                     f'{split_col_} <= {split_val} & {split_col_2_} > {split_val_2}',
                     f'{split_col_} > {split_val} & {split_col_2_} <= {split_val_2}', 
                     f'{split_col_} > {split_val} & {split_col_2_} > {split_val_2}']
            #s2 = []
            #for keep, scaffold in zip(keep_scaffolds, split):
            #    if keep:
            #        s2.append(scaffold)
            #split = s2
                
        # separate statistics by measurement, feature scaffold, prediction
        idx = pd.MultiIndex.from_product([['dTm', 'ddG'], split, cols])
        df_out = pd.DataFrame(index=idx)

        # iterate through measurements and splits
        for meas in measurements:
            for sp in split:

                # get new copies that get reduced per scaffold / measurement
                cur_df_bin = db_discrete_bin.copy(deep=True)
                cur_df_discrete = db_discrete.copy(deep=True)
                cur_df_cont = db_internal.copy(deep=True)

                # the following section contains the logic for splitting based on
                # which scaffold is being considered and is self-explanatory
                # there is no logic needed if there is no split requested

                if split_col_ != 'tmp' and split_col_2_ != 'tmp2' and split_col_ != split_col_2_:
                    # case where there are 4 scaffolds
                    if len(sp.split('&')) > 1:
                        if '>' in sp.split('&')[0]:
                            cur_df_bin = cur_df_bin.loc[cur_df_bin[split_col_] > split_val]
                            cur_df_discrete = cur_df_discrete.loc[cur_df_discrete[split_col_] > split_val]
                            cur_df_cont = cur_df_cont.loc[cur_df_cont[split_col_] > split_val]
                        elif '<=' in sp.split('&')[0]:
                            cur_df_bin = cur_df_bin.loc[cur_df_bin[split_col_] <= split_val]
                            cur_df_discrete = cur_df_discrete.loc[cur_df_discrete[split_col_] <= split_val]
                            cur_df_cont = cur_df_cont.loc[cur_df_cont[split_col_] <= split_val]

                        if '>' in sp.split('&')[1]:
                            cur_df_bin = cur_df_bin.loc[cur_df_bin[split_col_2_] > split_val_2]
                            cur_df_discrete = cur_df_discrete.loc[cur_df_discrete[split_col_2_] > split_val_2]
                            cur_df_cont = cur_df_cont.loc[cur_df_cont[split_col_2_] > split_val_2]
                        elif '<=' in sp.split('&')[1]:
                            cur_df_bin = cur_df_bin.loc[cur_df_bin[split_col_2_] <= split_val_2]
                            cur_df_discrete = cur_df_discrete.loc[cur_df_discrete[split_col_2_] <= split_val_2]
                            cur_df_cont = cur_df_cont.loc[cur_df_cont[split_col_2_] <= split_val_2]

                    # case where there are 3 scaffolds
                    elif len(sp.split('&')) == 1:
                        if not split_first:
                            if '>' in sp:
                                cur_df_bin = cur_df_bin.loc[cur_df_bin[split_col_] > split_val]
                                cur_df_discrete = cur_df_discrete.loc[cur_df_discrete[split_col_] > split_val]
                                cur_df_cont = cur_df_cont.loc[cur_df_cont[split_col_] > split_val]
                            elif '<=' in sp:
                                cur_df_bin = cur_df_bin.loc[cur_df_bin[split_col_] <= split_val]
                                cur_df_discrete = cur_df_discrete.loc[cur_df_discrete[split_col_] <= split_val]
                                cur_df_cont = cur_df_cont.loc[cur_df_cont[split_col_] <= split_val]

                        elif not split_last:
                            if '>' in sp:
                                cur_df_bin = cur_df_bin.loc[cur_df_bin[split_col_2_] > split_val_2]
                                cur_df_discrete = cur_df_discrete.loc[cur_df_discrete[split_col_2_] > split_val_2]
                                cur_df_cont = cur_df_cont.loc[cur_df_cont[split_col_2_] > split_val_2]
                            elif '<=' in sp:
                                cur_df_bin = cur_df_bin.loc[cur_df_bin[split_col_2_] <= split_val_2]
                                cur_df_discrete = cur_df_discrete.loc[cur_df_discrete[split_col_2_] <= split_val_2]
                                cur_df_cont = cur_df_cont.loc[cur_df_cont[split_col_2_] <= split_val_2]   

                # case where there are 3 scaffolds (on the same feature)
                elif split_col_ == split_col_2_:

                    if ('>' in sp and not '>=' in sp):
                        cur_df_bin = cur_df_bin.loc[cur_df_bin[split_col_] > split_val]
                        cur_df_discrete = cur_df_discrete.loc[cur_df_discrete[split_col_] > split_val]
                        cur_df_cont = cur_df_cont.loc[cur_df_cont[split_col_] > split_val]
                    elif '<=' in sp:
                        cur_df_bin = cur_df_bin.loc[cur_df_bin[split_col_] <= split_val_2]
                        cur_df_discrete = cur_df_discrete.loc[cur_df_discrete[split_col_] <= split_val_2]
                        cur_df_cont = cur_df_cont.loc[cur_df_cont[split_col_] <= split_val_2]
                    else:
                        cur_df_bin = cur_df_bin.loc[(cur_df_bin[split_col_] > split_val_2) & (cur_df_bin[split_col_] <= split_val)]
                        cur_df_discrete = cur_df_discrete.loc[(cur_df_discrete[split_col_] > split_val_2) & (cur_df_discrete[split_col_] <= split_val)]
                        cur_df_cont = cur_df_cont.loc[(cur_df_cont[split_col_] > split_val_2) & (cur_df_cont[split_col_] <= split_val)]
                        
                # case where there are two scaffolds on one feature
                elif split_col_2_ == 'tmp2' and split_col_ != 'tmp':

                    if '>' in sp:
                        cur_df_bin = cur_df_bin.loc[cur_df_bin[split_col_] > split_val]
                        cur_df_discrete = cur_df_discrete.loc[cur_df_discrete[split_col_] > split_val]
                        cur_df_cont = cur_df_cont.loc[cur_df_cont[split_col_] > split_val]
                    else:
                        cur_df_bin = cur_df_bin.loc[cur_df_bin[split_col_] <= split_val]
                        cur_df_discrete = cur_df_discrete.loc[cur_df_discrete[split_col_] <= split_val]                  
                        cur_df_cont = cur_df_cont.loc[cur_df_cont[split_col_] <= split_val] 
                
                # in this next section we compute the statistics one model at a time
                # all predictions should have the suffix _dir to designate direction mutations
                for col in (tqdm([col for col in cols if ('_dir' in col and not 'runtime' in col)]) \
                    if not quiet else [col for col in cols if ('_dir' in col and not 'runtime' in col)]):
                    
                    # get a reduced version of cur_df_cont for the relevant model
                    try:
                        pred_df_cont = cur_df_cont[[col,meas,f'runtime_{col}']].dropna()
                        # we only care about the total runtime for this function
                        df_out.loc[(meas,sp,col), 'runtime'] = pred_df_cont[f'runtime_{col}'].sum()
                        pred_df_cont = pred_df_cont.drop(f'runtime_{col}', axis=1)
                    except KeyError:
                        #if not quiet:
                        #    print('e', col)
                        pred_df_cont = cur_df_cont[[col,meas]].dropna()
                        df_out.loc[(meas,sp,col), 'runtime'] = np.nan    

                    # get a reduced version of the classification-task predictions and labels
                    pred_df_bin = cur_df_bin[[col,meas]].dropna()
                    #print(pred_df_bin)

                    if 'n' in stats or stats == ():
                        df_out.loc[(meas,sp,col), 'n'] = len(pred_df_bin)
                        saved_n = len(pred_df_bin)
                    if len(pred_df_bin) == 0:
                        raise AssertionError(f'There are no {col} predictions in this scaffold ({sp})!')
                    
                    # compute the 'easy' whole-dataset statistics
                    try:
                        tn, fp, fn, tp = metrics.confusion_matrix(pred_df_bin[meas], pred_df_bin[col]).ravel()
                    except:
                        tn, fp, fn, tp = 1,1,1,1
                    # compute each statistic by default (when stats==())
                    if 'tp' in stats or stats == ():
                        df_out.loc[(meas,sp,col), 'tp'] = tp
                    if 'fp' in stats or stats == ():
                        df_out.loc[(meas,sp,col), 'fp'] = fp
                    if 'tn' in stats or stats == ():
                        df_out.loc[(meas,sp,col), 'tn'] = tn 
                    if 'fn' in stats or stats == ():  
                        df_out.loc[(meas,sp,col), 'fn'] = fn   
                    if 'sensitivity' in stats or stats == (): 
                        df_out.loc[(meas,sp,col), 'sensitivity'] = tp/(tp+fn)
                    if 'specificity' in stats or stats == ():         
                        df_out.loc[(meas,sp,col), 'specificity'] = tn/(tn+fp)
                    if 'PPV' in stats or stats == (): 
                        df_out.loc[(meas,sp,col), 'PPV'] = tp/(tp+fp)
                    if 'pred_positives' in stats or stats == ():
                        df_out.loc[(meas,sp,col), 'pred_positives'] = tp+fp
                    if 'accuracy' in stats or stats == (): 
                        df_out.loc[(meas,sp,col), 'accuracy'] = metrics.accuracy_score(pred_df_bin[meas], pred_df_bin[col])
                    if 'f1_score' in stats or stats == (): 
                        df_out.loc[(meas,sp,col), 'f1_score'] = metrics.f1_score(pred_df_bin[meas], pred_df_bin[col])
                    if 'MCC' in stats or stats == ():
                        df_out.loc[(meas,sp,col), 'MCC'] = metrics.matthews_corrcoef(pred_df_bin[meas], pred_df_bin[col])

                    # get a reduced version of the model's predictions with discrete ground truth labels
                    pred_df_discrete = cur_df_discrete[[col,meas]].dropna()
                    # discrete labels allow testing different thresholds of continuous predictions
                    # e.g. for area-under-curve methods
                    try:
                        pred_df_discrete[meas] = pred_df_discrete[meas].astype(int)
                        auroc = metrics.roc_auc_score(pred_df_discrete[meas], pred_df_discrete[col])
                        auprc = metrics.average_precision_score(pred_df_discrete[meas], pred_df_discrete[col])
                        if 'auroc' in stats or stats == (): 
                            df_out.loc[(meas,sp,col), 'auroc'] = auroc
                        if 'auprc' in stats or stats == (): 
                            df_out.loc[(meas,sp,col), 'auprc'] = auprc
                    # might fail for small scaffolds
                    except Exception as e:
                        if not quiet:
                            print('Couldn\'t compute AUC:', e)

                    # using the full (continous) predictions and labels now
                    pred_df_cont = cur_df_cont[[col,meas]].dropna().join(db_grouper)

                    # recall of the top-k predicted-most-stable proteins across the whole slice of data
                    for stat in [s for s in stats if 'recall@' in s] if stats != () else ['recall@k0.0', 'recall@k1.0']:
                        k = stat.split('@')[-1].strip('k')
                        if k == '':
                            k = 0.
                        else:
                            k = float(k)
                        
                        pred_df_discrete_k = pred_df_cont.copy(deep=True).drop_duplicates()
                        pred_df_discrete_k[meas] = pred_df_discrete_k[meas].apply(lambda x: 1 if x > k else 0)
                        stable_ct = pred_df_discrete_k[meas].sum()

                        gain = pred_df_cont.loc[pred_df_cont[meas] > k, meas].sum()
                        #print(stable_ct)
                        #print(stable_ct)
                        df_out.loc[(meas,sp,col), f'{k}_n_stable'] = stable_ct
                    
                        sorted_preds = pred_df_discrete_k.sort_values(col, ascending=False).index
                        df_out.loc[(meas,sp,col), f'recall@k{k}'] = pred_df_discrete_k.loc[sorted_preds[:stable_ct], meas].sum() / stable_ct
                        df_out.loc[(meas,sp,col), f'gain@k{k}'] = pred_df_cont.drop_duplicates().loc[(sorted_preds[:stable_ct]), meas].sum() / gain

                    # average experimental stabilization of predicted positives
                    if 'mean_stabilization' in stats or stats == ():
                        df_out.loc[(meas,sp,col), 'mean_stabilization'] = pred_df_cont.loc[pred_df_cont[col]>0, meas].mean()
                    # average experimental stabilization of predicted positives
                    if 'net_stabilization' in stats or stats == ():
                        df_out.loc[(meas,sp,col), 'net_stabilization'] = pred_df_cont.loc[pred_df_cont[col]>0, meas].sum()
                    # average predicted score for experimentally stabilizing mutants
                    if 'mean_stable_pred' in stats or stats == ():
                        df_out.loc[(meas,sp,col), 'mean_stable_pred'] = pred_df_cont.loc[pred_df_cont[meas]>0, col].mean()
                    # mean squared error
                    if 'mse' in stats or stats == ():
                        df_out.loc[(meas,sp,col), 'mean_squared_error'] = metrics.mean_squared_error(pred_df_cont[meas], pred_df_cont[col])

                    # top-1 score, e.g. the experimental stabilization achieved on 
                    # average for the top-scoring mutant of each protein
                    if ('mean_t1s' in stats) or (stats == ()): 
                        top_1_stab = 0
                        for code, group in pred_df_cont.groupby(grouper):
                            top_1_stab += group.sort_values(col, ascending=False)[meas].head(1).item()
                        df_out.loc[(meas,sp,col), 'mean_t1s'] = top_1_stab / len(pred_df_cont[grouper].unique())

                    # inverse of the assigned rank of the number one most stable protein per group
                    #if ('mean_reciprocal_rank' in stats) or (stats == ()): 
                    #    reciprocal_rank_sum = 0
                    #    unique_groups = pred_df_cont[grouper].unique()
                    #    for code, group in pred_df_cont.groupby(grouper):
                    #        group = group.drop_duplicates()
                    #        sorted_group = group.sort_values(col, ascending=False)
                    #        highest_meas_rank = sorted_group[meas].idxmax()

                    #        rank_of_highest_meas = sorted_group.index.get_loc(highest_meas_rank)
                    #        if type(rank_of_highest_meas) in [slice, list, bool]:
                    #            print('Something went wrong with MRR for', col, code)
                    #            continue
                    #        try:
                    #            rank_of_highest_meas += 1
                    #        except:
                    #            print('Something went wrong with MRR for', col, code)
                    #            continue

                    #        reciprocal_rank_sum += 1 / rank_of_highest_meas

                    #    mean_reciprocal_rank = reciprocal_rank_sum / len(unique_groups)
                    #    df_out.loc[(meas, sp, col), 'mean_reciprocal_rank'] = mean_reciprocal_rank
                    
                    # normalized discounted cumulative gain, a measure of information retrieval ability
                    if ('ndcg' in stats) or (stats == ()):
                        # whole-dataset version (not presented in study)
                        df_out.loc[(meas,sp,col), 'ndcg'] = compute_ndcg(pred_df_cont, col, meas)
                        cum_ndcg = 0
                        w_cum_ndcg = 0
                        cum_d = 0
                        w_cum_d = 0
                        cum_muts = 0
                        # iterate over unique proteins (wild-type structures)
                        for code, group in pred_df_cont.groupby(grouper): 
                            # must be more than one to retrieve, and their stabilities should be different
                            if len(group.loc[group[meas]>0]) > 1 and not all(group[meas]==group[meas][0]):
                                cur_ndcg = compute_ndcg(group, col, meas)
                                # can happen if there are no stable mutants
                                if np.isnan(cur_ndcg):
                                    continue
                                # running-total (cumulative)
                                cum_ndcg += cur_ndcg
                                cum_d += 1
                                # weighted running-total (by log(num mutants))
                                w_cum_ndcg += cur_ndcg * np.log(len(group.loc[group[meas]>0]))
                                w_cum_d += np.log(len(group.loc[group[meas]>0]))
                                cum_muts += len(group.loc[group[meas]>0])
                        df_out.loc[(meas,sp,col), 'mean_ndcg'] = cum_ndcg / (cum_d if cum_d > 0 else 1)
                        df_out.loc[(meas,sp,col), 'weighted_ndcg'] = w_cum_ndcg / (w_cum_d if w_cum_d > 0 else 1)
                        # may be less than the number of proteins in the dataset based on the if statement               
                        df_out.loc[(meas,sp,col), 'n_proteins_ndcg'] = cum_d
                        # may be less than the number of mutants based on the if statement
                        df_out.loc[(meas,sp,col), 'n_muts_ndcg'] = cum_muts
                    
                    if ('pearson' in stats) or (stats == ()):
                        whole_r, _ = pearsonr(pred_df_cont[col], pred_df_cont[meas])
                        df_out.loc[(meas,sp,col), 'pearson'] = whole_r

                    # Spearman's rho, rank-order version of Pearson's r
                    # follows same logic as above
                    if ('spearman' in stats) or (stats == ()):
                        whole_p, _ = spearmanr(pred_df_cont[col], pred_df_cont[meas])
                        df_out.loc[(meas,sp,col), 'spearman'] = whole_p
                        cum_p = 0
                        w_cum_p = 0
                        cum_d = 0
                        w_cum_d = 0
                        cum_muts = 0
                        for code, group in pred_df_cont.groupby(grouper):
                            if len(group) > 1 and not all(group[meas]==group[meas][0]):
                                spearman, _ = spearmanr(group[col], group[meas])
                                # can happen if all predictions are the same
                                # in which case ranking ability is poor since we 
                                # already checked that the measurements are different
                                if np.isnan(spearman):
                                    spearman=0
                                cum_p += spearman
                                cum_d += 1
                                w_cum_p += spearman * np.log(len(group))
                                w_cum_d += np.log(len(group))
                                cum_muts += len(group)
                        df_out.loc[(meas,sp,col), 'mean_spearman'] = cum_p / (cum_d if cum_d > 0 else 1)
                        df_out.loc[(meas,sp,col), 'weighted_spearman'] = w_cum_p / (w_cum_d if w_cum_d > 0 else 1)
                        df_out.loc[(meas,sp,col), 'n_proteins_spearman'] = cum_d
                        df_out.loc[(meas,sp,col), 'n_muts_spearman'] = cum_muts
                        if cum_muts > saved_n:
                            print(cum_muts, saved_n, sp, col)

                    # refresh the discrete dataframe
                    pred_df_discrete = cur_df_discrete[[col,meas]].dropna().join(db_grouper)
                    #pred_df_discrete['code'] = pred_df_discrete.index.str[:4] 
                    
                    # calculate area under the precision recall curve per protein as with the above stats
                    if ('auprc' in stats) or (stats == ()):
                        #df_out.loc[(meas,sp,col), 'auprc'] = metrics.average_precision_score(pred_df_discrete[meas], pred_df_discrete[col])
                        cum_ps = 0
                        w_cum_ps = 0
                        cum_d = 0
                        w_cum_d = 0
                        cum_muts = 0
                        for _, group in pred_df_discrete.groupby(grouper): 
                            if len(group) > 1:
                                #group[meas] = group[meas].astype(int)
                                cur_ps = metrics.average_precision_score(group[meas], group[col])
                                # NaN if there is only one class in this scaffold for this protein
                                if np.isnan(cur_ps):
                                    continue
                                cum_ps += cur_ps
                                cum_d += 1
                                w_cum_ps += cur_ps * np.log(len(group))
                                w_cum_d += np.log(len(group))
                                cum_muts += len(group)
                        df_out.loc[(meas,sp,col), 'mean_auprc'] = cum_ps / (cum_d if cum_d > 0 else 1)
                        df_out.loc[(meas,sp,col), 'weighted_auprc'] = w_cum_ps / (w_cum_d if cum_d > 0 else 1)
                        df_out.loc[(meas,sp,col), 'n_proteins_auprc'] = cum_d
                        df_out.loc[(meas,sp,col), 'n_muts_auprc'] = cum_muts

                    # these are the expensive statistics (calculated at 100 thresholds)
                    # it would take too long to compute them per-scaffold
                    if split_col_ == 'tmp':
                        if ('auppc' in stats) or (stats == ()):
                            percentiles = [str(int(s))+'%' for s in range(1, 100)]
                            percentile_values = [int(s.split('%')[0]) for s in percentiles]
                        else:
                            percentiles = [s for s in stats if '%' in s]
                            percentile_values = [float(s.split('%')[0]) for s in percentiles]

                        if grouper is not None:
                            # Apply the function to each group and reset the index
                            results_df = pred_df_cont.groupby(grouper).apply(
                                calculate_ppc, pred_col=col, meas=meas, percentile_values=percentile_values
                                ).reset_index()
                        else:
                            results_df = calculate_ppc(pred_df_cont, pred_col=col, meas=meas, percentile_values=percentile_values)
                        stat_dict = {}
                        # Aggregate results
                        for stat in percentiles:
                            try:
                                stat_dict[stat] = results_df[stat].sum() / results_df[f"pos_{stat}"].sum()
                            except ZeroDivisionError:
                                stat_dict[stat] = 0

                        # Assign to df_out
                        df_out.loc[(meas, sp, col), percentiles] = list(stat_dict.values())
                        df_out.loc[(meas, sp, col), 'auppc'] = df_out.loc[(meas, sp, col), percentiles].mean()

                        # mean stability vs prediction percentile curve
                        if ('aumsc' in stats) or (stats == ()):
                            percentiles = [str(int(s))+'$' for s in range(1, 100)]
                            percentile_values = [int(s.split('$')[0]) for s in percentiles]
                        else:
                            percentiles = [s for s in stats if '$' in s]
                            percentile_values = [float(s.split('$')[0]) for s in percentiles]

                        if grouper is not None:
                            # Apply the function to each group and reset the index
                            results_df = pred_df_cont.groupby(grouper).apply(
                                calculate_msc, pred_col=col, meas=meas, percentile_values=percentile_values
                                ).reset_index()
                        else:
                            results_df = calculate_msc(pred_df_cont,
                                pred_col=col, meas=meas, percentile_values=percentile_values
                                )             

                        stat_dict = {}
                        # Aggregate results
                        for stat in percentiles:
                            try:
                                stat_dict[stat] = results_df[stat].sum() / results_df[f"pos_{stat}"].sum()
                            except ZeroDivisionError:
                                stat_dict[stat] = 0

                        # Assign to df_out
                        df_out.loc[(meas, sp, col), percentiles] = list(stat_dict.values())
                        df_out.loc[(meas, sp, col), 'aumsc'] = df_out.loc[(meas, sp, col), percentiles].mean()
        dfs_out.append(df_out)

    if n_bootstraps > 0:

        concatenated_df = pd.concat(dfs_out, axis=0)

        # Reset the index to a simple range index, then set it back to a multi-index
        concatenated_df.reset_index(inplace=True)
        concatenated_df.set_index(['level_0', 'level_1', 'level_2'], inplace=True)

        # Now perform the groupby operation and compute mean and std
        mean_df = concatenated_df.groupby(level=['level_0', 'level_1', 'level_2']).mean()
        std_df = concatenated_df.groupby(level=['level_0', 'level_1', 'level_2']).std()

        # Create new DataFrame with _mean and _std columns
        result_df = pd.DataFrame(index=mean_df.index)

        for col in mean_df.columns:
            result_df[f"{col}_mean"] = mean_df[col]
            result_df[f"{col}_std"] = std_df[col]

        df_out = result_df

    else:
        df_out = dfs_out[0]

    df_out = df_out.reset_index()
    
    # add labels for the input information used by the model
    #df_out['model_type'] = 'structural'
    for k,v in mapping_categories.items():
        for m in v:
            # there are many variants of the models so just check if their base name matches
            df_out.loc[df_out['level_2'].str.contains(m), 'model_type'] = k
    df_out = df_out.rename({'level_0': 'measurement', 'level_1': 'class', 'level_2': 'model'}, axis=1)

    df_out = df_out.set_index(['measurement', 'model_type', 'model', 'class'])
    # sort by measurement type, and then model type within each measurement type
    # class is the scaffold
    df_out = df_out.sort_index(level=1).sort_index(level=0)

    return df_out.dropna(how='all')


def calculate_p_values(df, ground_truth_col, statistic, n_bootstraps=100, grouper='code', noise=0):
    compute = {'auprc': compute_auprc, 'weighted_ndcg': compute_weighted_ndcg, 'weighted_spearman': compute_weighted_spearman, 
        'spearman': lambda x, y, z, grouper: stats.spearmanr(x[y], x[z])[0], 'recall@k0.0': compute_recall_k0, 'recall@k1.0': compute_recall_k1,
        'mean_t1s': compute_t1s, 'mean_stabilization': compute_mean_stab, 'net_stabilization': compute_net_stab}[statistic]

    #print(df)
    df_out = pd.DataFrame()
    model_columns = [col for col in df.columns if col != ground_truth_col and col != grouper]
    n_models = len(model_columns)

     # Initialize DataFrame of p-values with NA
    p_values = pd.DataFrame(index=[col for col in model_columns if not is_combined_model(col)], 
                            columns=[col for col in model_columns if not is_combined_model(col)]) 
    p_values = p_values.fillna(np.inf)
    mean_values = pd.DataFrame(index=[col for col in model_columns if not is_combined_model(col)], 
                        columns=[col for col in model_columns if not is_combined_model(col)])
    std_values = pd.DataFrame(index=[col for col in model_columns if not is_combined_model(col)], 
                        columns=[col for col in model_columns if not is_combined_model(col)])
    mean_values = mean_values.fillna(-np.inf)
    std_values = std_values.fillna(-np.inf)

    #if n_bootstraps < 1:
    #    bootstrap_statistics = np.zeros((1, n_models))
    #    for j, model_col in enumerate(model_columns):
    #        stat = compute(df, model_col, ground_truth_col, grouper=grouper)
    #        bootstrap_statistics[0, j] = stat
    #else:
    # Bootstrap
    bootstrap_statistics = np.zeros((n_bootstraps, n_models))
    dfs_resampled = bootstrap_by_grouper(df, n_bootstraps, grouper=grouper, drop=False, noise=noise)
    for i in tqdm(range(n_bootstraps)):
        #df_resampled = df.sample(n=len(df), replace=True)
        for j, model_col in enumerate(model_columns):
            stat = compute(dfs_resampled[i], model_col, ground_truth_col, grouper=grouper)
            bootstrap_statistics[i, j] = stat

    # Compute p-values for each pair of models
    for j in range(n_models):
        #print(model_columns[j])
        better_combination = False
        model_j_statistic = bootstrap_statistics[:, j]
        if not is_combined_model(model_columns[j]):
            mean_values.loc[model_columns[j], model_columns[j]] = np.mean(model_j_statistic)
            std_values.loc[model_columns[j], model_columns[j]] = np.std(model_j_statistic)
            continue
        #print(mean_values)
        constituent_models = model_columns[j].split(" * ")[0].split(" + ")

        # if the first constituent model has a higher mean value than the second
        if np.mean(bootstrap_statistics[:, model_columns.index(constituent_models[0])]) > np.mean(bootstrap_statistics[:, model_columns.index(constituent_models[1])]):
            model_k_statistic = bootstrap_statistics[:, model_columns.index(constituent_models[0])]
            best_constituent = constituent_models[0]
        # if the second constituent model has a higher mean value than the first
        else:
            model_k_statistic = bootstrap_statistics[:, model_columns.index(constituent_models[1])]
            best_constituent = constituent_models[1]
        t_stat, p_value = stats.ttest_rel(model_j_statistic, model_k_statistic)
        assert all([c in p_values.columns for c in constituent_models])
        df_out = pd.concat([df_out, pd.DataFrame({0:  [constituent_models[0], 1, constituent_models[1], model_columns[j].split(" * ")[1], np.mean(model_j_statistic), p_value]}).T])

        # replace the mean bootstrap value if it is higher than the current value
        if np.mean(model_j_statistic) > mean_values.loc[constituent_models[0], constituent_models[1]]:
            better_combination = True
            mean_values.loc[constituent_models[0], constituent_models[1]] = np.mean(model_j_statistic)
            std_values.loc[constituent_models[0], constituent_models[1]] = np.std(model_j_statistic)
            assert np.mean(model_j_statistic) > mean_values.loc[constituent_models[1], constituent_models[0]]
            mean_values.loc[constituent_models[1], constituent_models[0]] = np.mean(model_j_statistic)
            std_values.loc[constituent_models[1], constituent_models[0]] = np.std(model_j_statistic)

        #print(model_columns[j], np.mean(model_j_statistic), best_constituent, np.mean(model_k_statistic), np.log10(p_value)) 
        # if the mean value for combined model j is greater than its best constituent, and this combination is better than the previous best, record the p_value
        if (np.mean(model_j_statistic) > np.mean(model_k_statistic)) and better_combination:
            #print('better combination')
            # record the p_value in the upper left triangle
            if p_value < p_values.loc[constituent_models[0], constituent_models[1]]:
                #print(p_value, p_values.loc[constituent_models[0], constituent_models[1]]) 
                p_values.loc[constituent_models[0], constituent_models[1]] = p_value
                assert p_value < p_values.loc[constituent_models[1], constituent_models[0]]
                p_values.loc[constituent_models[1], constituent_models[0]] = p_value
            #else:
                #print('no better p-value')

    df_out.columns = ['model1', 'weight1', 'model2', 'weight2', f'mean_{statistic}', 'p_value']
    return p_values, mean_values, std_values, df_out


def model_combinations_heatmap(df, dfm, db_measurements, statistic, measurement, n_bootstraps=100, threshold=None, subset=None, sigdigs=3, upper=None, grouper='cluster', noise=0, annot=True, title=None, saveloc_table='../data/extended/figure_data/data_heatmap.csv'):

    font = {'size'   : 14}
    matplotlib.rc('font', **font)

    df_slice = df.xs(measurement, level=0).reset_index()
    #df_slice = df_slice.loc[~df_slice['model'].isin(['ddG_dir', 'dTm_dir'])]
    df_slice[['model1', 'weight1', 'model2', 'weight2']] = df_slice['model'].apply(process_index).apply(pd.Series)

    #df_slice = df_slice.loc[df_slice['weight2'] > 0]

    if subset is not None:
        df = df_slice.loc[df_slice['model1'].isin(subset) & df_slice['model2'].isin(subset)]

    df = df[['model1', 'weight1', 'model2', 'weight2', statistic, 'runtime']]

    fig, ax = plt.subplots(figsize=(20,12), dpi=300)

    pdf = df.copy(deep=True).reset_index(drop=True).sort_values(statistic, ascending=False).reset_index(drop=True)

    pdf['orig_col'] = pdf['model1'] + ' + ' + pdf['model2'] + ' * ' + pdf['weight2'].astype(str)
    pdf.loc[pdf['weight2']==1, 'orig_col'] = pdf['model1'] + ' + ' + pdf['model2'] + ' * ' + '1'
    pdf.loc[pdf['weight2']==0, 'orig_col'] = pdf['model1']

    if grouper is not None:
        pdf = dfm[list(pdf['orig_col'].values) + [grouper]]
    else:
        pdf = dfm[list(pdf['orig_col'].values)]
    
    pdf = pdf.join(db_measurements[measurement]).dropna(subset=measurement).dropna(how='any')

    print(len(pdf))
    if n_bootstraps > 0:
        pvals, performances, std_values, stat_df = calculate_p_values(
            pdf, ground_truth_col=measurement, statistic=statistic, n_bootstraps=n_bootstraps, grouper=grouper, noise=noise)
        #print(performances)
        diagonal_indices = np.argsort(-performances.values.diagonal())
        performances = performances.iloc[diagonal_indices, diagonal_indices]
        #print(performances)
        log_pvals = pvals.astype(float).applymap(np.log10)
        log_pvals = log_pvals.iloc[diagonal_indices, diagonal_indices]
        std_values = std_values.iloc[diagonal_indices, diagonal_indices]
    else:
        stat_df = get_stat_df(df, statistic, '', dfm)
        performances = stat_df.drop(['weight1','weight2','corr'],axis=1).pivot_table(values=statistic, index='model1', columns='model2')
        diagonal_indices = np.argsort(-performances.values.diagonal())
        performances = performances.iloc[diagonal_indices, diagonal_indices]
        log_pvals = performances.copy(deep=True)
        std_values = performances.copy(deep=True)

    #print(performances)
    factor = 500
    #runtimes = runtimes.where(np.triu(np.ones(performances.shape)).astype(bool)).applymap(np.log10) * factor #.applymap(lambda x: '{:.2e}'.format(x)).to_numpy()

    white_cmap = ListedColormap(['white'])
    #print(performances)
    
    fstring = '{:.'+str(sigdigs)+'f}'
    # Combine mean performance and standard deviation with fewer significant digits
    if n_bootstraps > 1:
        combined_annotations = performances.applymap(fstring.format) + '\n± ' + std_values.applymap('{:.2f}'.format)
    else:
        combined_annotations = performances.applymap(fstring.format)

    if upper == 'delta':
        delta_df = (performances - performances.values.diagonal()).T
        pivot_table_2 = delta_df.where(np.triu(np.ones(delta_df.shape), k=1).astype(bool))
        vmin2 = np.nanmin(pivot_table_2, axis=(0,1))
        vmax2 = np.nanmax(pivot_table_2, axis=(0,1))
        v = max(np.abs(vmax2), np.abs(vmin2))
        print(v)

    sns.heatmap(performances, annot=combined_annotations if annot else None, cmap='viridis', cbar=False, fmt='', vmin=threshold, #fmt='.2e' if threshold is not None else '.3f',
        mask=np.triu(np.ones(log_pvals.shape, dtype=bool), k=1), ax=ax, annot_kws={"size": 13})
    cbar = plt.colorbar(ax.collections[0], ax=ax, location="right", use_gridspec=False, pad=0.03)
    cbar.ax.tick_params(labelsize=30)

    if upper is not None:
        sns.heatmap(pivot_table_2, annot=annot, cmap='coolwarm', cbar=False, vmin=-v, vmax=v, fmt=f'.{sigdigs}f', ax=ax)
        cbar1 = plt.colorbar(ax.collections[1], ax=ax, location="right", use_gridspec=False, pad=0.1)
        cbar1.ax.tick_params(labelsize=30)

    table_out = performances * np.tril(np.ones(log_pvals.shape, dtype=bool), k=0) + pivot_table_2.fillna(0)
    table_out.index.name = 'model1'
    table_out = table_out.reset_index() 
    table_out['model1'] = table_out['model1'].apply(lambda x: remap_names_2.get(x, x.replace('_dir', '')))
    table_out = table_out.set_index('model1')
    table_out.columns = table_out.index
    table_out.to_csv(saveloc_table)

    flattened_values = log_pvals[(log_pvals > -10000) & (log_pvals < 10000)].values.ravel()
    flattened_list = [value for value in flattened_values if not np.isnan(value)]

    for tick_label in ax.get_xticklabels():
        tick_label.set_color(determine_base_color(tick_label))
    for tick_label in ax.get_yticklabels():
        tick_label.set_color(determine_base_color(tick_label))
    remapped_x = [remap_names_2[tick.get_text()] if tick.get_text() in remap_names_2.keys() else tick.get_text() for tick in ax.get_xticklabels()]
    print(remapped_x)
    remapped_x = [n[:-4] if n.endswith('_dir') or n.endswith('_inv') else n for n in remapped_x]

    ax.set_xticklabels(remapped_x)
    ax.set_yticklabels(remapped_x)

    measurement_ = {'ddG': 'ΔΔG', 'dTm': 'ΔTm'}[measurement]
    try:
        statistic_ = {'weighted_ndcg': 'wNDCG', 'weighted_spearman': 'wρ', 'spearman': 'ρ', 'weighted_auprc': 'wAUPRC', 'auprc': 'AUPRC', 'mean_t1s': 'mean top-1 score', 'mean_stabilization': 'Mean Stabilization', 'net_stabilization': 'Net Stabilization'}[statistic]
    except:
        statistic_ = statistic
    try:
        upper_ = {'corr': 'Rank Correlation', 'antisymmetry': 'Antisymmetry', 'bias': 'Bias'}[upper]
    except:
        upper_ = upper

    plt.title(title, fontsize=50, y=1.03)
    if upper == None:
        plt.title(f'Prediction {statistic} of Models', fontsize=50, fontweight='bold')
    else:
        plt.text(1.05, 0.5, 'Improvement Over Best Constituent', va='center', ha='center', fontsize=36, rotation=270, rotation_mode='anchor', transform=plt.gca().transAxes)
        plt.text(-0.4, 0.5, f'Max {statistic_} of Model Combinations', va='center', ha='center', fontsize=36, rotation=90, rotation_mode='anchor', transform=plt.gca().transAxes)

    plt.ylabel(None)
    plt.xlabel(None)
    plt.xticks(fontsize=26, rotation=90)
    plt.yticks(fontsize=26, rotation=0)
    plt.show()

    return stat_df, log_pvals, fig

def get_stat_df(df, statistic, new_dir, preds=None):
    # Extract unique models from the DataFrame
    unique_models = pd.unique(df[['model1', 'model2']].values.ravel('K'))

    # Initialize the dictionary to store model combinations and their statistics
    combinations = {'model1': [], 'weight1': [], 'model2': [], 'weight2': [], statistic: []}
    if preds is not None:
        combinations.update({'corr': [], 'runtime_cpu': [], 'runtime_gpu': []}) #'runtime': [],
    #print(df.set_index('model').loc['ACDC-NN + I-Mutant3.0 * 0.2'])
    # Iterate through all unique pairs of models (upper-triangular)
    for i, model1 in enumerate(unique_models):
        for j, model2 in enumerate(unique_models):
            if j >= i:
                # Calculate the statistic for the current pair of models
                temp_df = df[((df['model1'] == model1) & (df['model2'] == model2)) | 
                            ((df['model1'] == model2) & (df['model2'] == model1))]

                stat_row = temp_df.loc[temp_df[statistic]==temp_df[statistic].max()].head(1)
                if len(stat_row) == 0:
                    print(model1, model2, 'e')
                    continue

                # Store the model pair and statistic value in the dictionary
                combinations['model1'].append(stat_row['model1'].item())
                combinations['weight1'].append(stat_row['weight1'].item())
                combinations['model2'].append(stat_row['model2'].item())
                combinations['weight2'].append(stat_row['weight2'].item())
                combinations[statistic].append(stat_row[statistic].item())
                runtime_cpu = 0
                runtime_gpu = 0

                if preds is not None:
                    runtime_cpu = -0.0001
                    runtime_gpu = -0.0001

                    if model1 != model2:
                        combinations['corr'].append(preds[[model1+new_dir, model2+new_dir]].corr('spearman').iloc[0,1])
                        combinations['corr'].append(preds[[model1+new_dir, model2+new_dir]].corr('spearman').iloc[0,1])                
                    else:
                        combinations['corr'].append(1)
                        combinations['corr'].append(1)
                    #print(model1, model2)
                    #print(new_dir) 

                    if 'runtime_'+model1+new_dir in preds.columns:
                        if 'cartesian' in model1 or 'korpm' in model1:
                            runtime_cpu += preds['runtime_'+model1+new_dir].sum()
                        else:
                            runtime_gpu += preds['runtime_'+model1+new_dir].sum() 
                    if 'runtime_'+model2+new_dir in preds.columns and model1 != model2:
                        if 'cartesian' in model2 or 'korpm' in model2:
                            runtime_cpu += preds['runtime_'+model2+new_dir].sum()
                        else:
                            runtime_gpu += preds['runtime_'+model2+new_dir].sum()

                    combinations['runtime_cpu'].append(runtime_cpu)
                    combinations['runtime_gpu'].append(runtime_gpu)
                    combinations['runtime_cpu'].append(runtime_cpu)
                    combinations['runtime_gpu'].append(runtime_gpu)

                    combinations['model1'].append(stat_row['model2'].item())
                    combinations['weight1'].append(stat_row['weight2'].item())
                    combinations['model2'].append(stat_row['model1'].item())
                    combinations['weight2'].append(stat_row['weight1'].item())
                    combinations[statistic].append(stat_row[statistic].item())

    # Create a new DataFrame with the calculated statistics
    #print(combinations)
    stat_df = pd.DataFrame(combinations)
    return stat_df


def model_combinations_heatmap_2(df, preds, statistic, direction, upper='corr', threshold=None, subset=None, annot=True, title=None, saveloc_table='../data/extended/figure_data/data_heatmap.csv'):

    font = {'size'   : 10}
    matplotlib.rc('font', **font)

    remap_direction = {'dir': 'Direct Mutations', 'inv': 'Inverse Mutations', 'combined': 'Both Directions'}

    df_slice = df.xs(direction, level=0).reset_index().drop('model_type', axis=1)
    df_slice = df_slice.loc[~df_slice['model'].isin(['ddG_dir', 'dTm_dir'])]
    df_slice[['model1', 'weight1', 'model2', 'weight2']] = df_slice['model'].apply(process_index).apply(pd.Series)

    #df_slice = df_slice.loc[df_slice['weight2'] > 0]
    if subset is not None:
        df = df_slice.loc[df_slice['model1'].isin(subset) & df_slice['model2'].isin(subset)]

    fig, ax = plt.subplots(figsize=(20,12), dpi=300)

    if direction == 'combined':
        new_dir = ''
        preds = stack_frames(preds.copy(deep=True))
    else:
        new_dir = '_' + direction

    stat_df = get_stat_df(df, statistic, new_dir, preds)

    # Create a pivot table to use for the heatmap
    pivot_table = stat_df.drop(['weight1','weight2','corr'],axis=1).pivot_table(values=statistic, index='model1', columns='model2') 
    pivot_table_2 = stat_df.drop(['weight1','weight2',statistic],axis=1).pivot_table(values='corr', index='model1', columns='model2') 

    # Sort the pivot table and delta_df by the diagonal entries
    diagonal_indices = np.argsort(-pivot_table.values.diagonal())
    pivot_table = pivot_table.iloc[diagonal_indices, diagonal_indices]
    pivot_table_2 = pivot_table_2.iloc[diagonal_indices, diagonal_indices]

    pivot_table = pivot_table.where(np.tril(np.ones(pivot_table.shape)).astype(bool))
    pivot_table_2 = pivot_table_2.where(np.triu(np.ones(pivot_table_2.shape), k=1).astype(bool))

    if upper == 'delta':
        delta_df = (pivot_table - pivot_table.values.diagonal()).T
        pivot_table_2 = delta_df.where(np.triu(np.ones(delta_df.shape), k=1).astype(bool))
        vmin2 = np.nanmin(pivot_table_2, axis=(0,1))
        vmax2 = np.nanmax(pivot_table_2, axis=(0,1))
    elif upper == 'antisymmetry':
        delta_df = get_stat_df(df, 'antisymmetry', new_dir, preds)
        pivot_table_2 = delta_df.drop(['weight1','weight2'],axis=1).pivot_table(values='antisymmetry', index='model1', columns='model2') 
        pivot_table_2 = pivot_table_2.iloc[diagonal_indices, diagonal_indices]
        pivot_table_2 = pivot_table_2.where(np.triu(np.ones(pivot_table_2.shape), k=1).astype(bool))
        vmin2 = np.nanmin(pivot_table_2, axis=(0,1))
        vmax2 = np.nanmax(pivot_table_2, axis=(0,1))
    elif upper == 'bias':
        delta_df = get_stat_df(df, 'bias', new_dir, preds)
        pivot_table_2 = delta_df.drop(['weight1','weight2'],axis=1).pivot_table(values='bias', index='model1', columns='model2')
        pivot_table_2 = pivot_table_2.iloc[diagonal_indices, diagonal_indices]
        pivot_table_2 = pivot_table_2.where(np.triu(np.ones(pivot_table_2.shape), k=1).astype(bool))
        vmin2 = np.nanmin(pivot_table_2, axis=(0,1))
        vmax2 = np.nanmax(pivot_table_2, axis=(0,1))
    elif upper == 'corr':
        vmin2 = np.nanmin(pivot_table_2, axis=(0,1))
        vmax2 = np.nanmax(pivot_table_2, axis=(0,1))

    v = max(vmax2, -vmin2)

    sns.heatmap(pivot_table, annot=annot, cmap='viridis', cbar=False, fmt='.2f' if threshold is None else '.0f', ax=ax, vmin=threshold,
        annot_kws={"size": 14} if threshold is not None else None)
    cbar2 = plt.colorbar(ax.collections[0], ax=ax, location="right", use_gridspec=False, pad=0.03)
    cbar2.ax.tick_params(labelsize=30)
    #cbar2.set_label(statistic.upper() + ' for Best Ensemble')

    if upper is not None:
        sns.heatmap(pivot_table_2, annot=annot, cmap='coolwarm', cbar=False, vmin=-v, vmax=v, fmt='.2f', ax=ax)
        cbar1 = plt.colorbar(ax.collections[1], ax=ax, location="right", use_gridspec=False, pad=0.1)
        cbar1.ax.tick_params(labelsize=30)

    table_out = pivot_table.fillna(0) + pivot_table_2.fillna(0)
    table_out = table_out.reset_index() 
    table_out['model1'] = table_out['model1'].apply(lambda x: remap_names.get(x, x))
    table_out = table_out.set_index('model1')
    table_out.columns = table_out.index
    table_out.to_csv(saveloc_table)
    
    for tick_label in ax.get_xticklabels():
        tick_label.set_color(determine_base_color(tick_label))
    for tick_label in ax.get_yticklabels():
        tick_label.set_color(determine_base_color(tick_label))

    remapped_x = [remap_names[tick.get_text()] if tick.get_text() in remap_names.keys() else tick.get_text() for tick in ax.get_xticklabels()]
    ax.set_xticklabels(remapped_x)
    ax.set_yticklabels(remapped_x)

    try:
        statistic_ = {'weighted_ndcg': 'wNDCG', 'mean_ndcg': 'mean NDCG', 'weighted_spearman': 'wρ', 'spearman': 'ρ', 'weighted_auprc': 'wAUPRC', 'auprc': 'AUPRC', 'auppc': 'AUPPC', 'net_stabilization': 'Net Stabilization'}[statistic]
    except:
        statistic_ = statistic
    try:
        upper_ = {'corr': 'Rank Correlation', 'antisymmetry': 'Antisymmetry', 'bias': 'Bias'}[upper]
    except:
        upper_ = upper

    plt.title(title, fontsize=50, y=1.03)
    if upper == None:
        plt.title(f'Prediction {statistic} of Models', fontsize=50, fontweight='bold')
    else:
        plt.text(1.05, 0.5, 'Improvement Over Best Constituent', va='center', ha='center', fontsize=36, rotation=270, rotation_mode='anchor', transform=plt.gca().transAxes)
        plt.text(-0.4, 0.5, f'Max {statistic_} of Model Combinations', va='center', ha='center', fontsize=36, rotation=90, rotation_mode='anchor', transform=plt.gca().transAxes)
    #plt.ylabel('Reference model')
    #plt.xlabel('Added Model')
    plt.ylabel(None)
    plt.xlabel(None)
    plt.xticks(fontsize=26, rotation=90)
    plt.yticks(fontsize=26, rotation=0)
    plt.show()

    return stat_df, fig


def custom_barplot(data, x, y, hue, width, ax, use_color=None, legend_labels=None, legend_colors=None, std=True, capsize=5):

    data = data.dropna(how='all', axis=1)

    if legend_labels is not None and legend_colors is not None:
        lut = dict(zip(legend_labels, legend_colors))

    unique_x = list(data[x].unique())
    data = data.copy(deep=True)
    if legend_labels is not None:
        #print(legend_labels)
        unique_hue = legend_labels
        try:
            unique_width = data.drop(x, axis=1).groupby([hue, width]).mean().astype(int).reset_index(level=1).loc[unique_hue][width]
        except ValueError as e:
            print(data.drop(x, axis=1).groupby([hue, width]).mean().fillna(0).astype(int).reset_index(level=1).loc[unique_hue][width])
            unique_width = data.drop(x, axis=1).groupby([hue, width]).mean().fillna(0).astype(int).reset_index(level=1).loc[unique_hue][width]
            print('Error:', e, 'probably caused by missing any data in one scaffold during bootstrapping, make less agressive scaffolds')
    else:
        unique_hue = data[hue].unique()
        unique_width = data[width].unique()

    try:
        assert len(unique_hue) == len(unique_width)
    except:        
        print(unique_hue)
        print(unique_width)
        raise AssertionError('This assertion usually fails when there are missing predictions which were not dropped in the input')

    if use_color == None:
        colors = legend_colors
    else:
        colors = [use_color]

    max_width = sum(unique_width)

    bar_centers = np.zeros((len(unique_x), len(unique_hue)))
    for i in range(len(unique_x)):
        bar_centers[i, :] = i

    #print(unique_width)

    w_sum = 0
    for j, w in enumerate(unique_width):
        w_sum += w
        bar_centers[:, j] += (-max_width / 2 + w_sum -w/2) / (max_width * 1.1)

    for j, (width_value, hue_value, color) in enumerate(zip(unique_width, unique_hue, colors)):
        y_max = -1
        for i, x_value in enumerate(unique_x):
            if 'ddG' in x_value or 'dTm' in x_value:
                continue
            filtered_data = data[(data[x] == x_value) & (data[width] == width_value)]
            y_value = filtered_data[f'{y}_mean'].item()#.mean()
            y_std = 0
            if std:
                y_std = filtered_data[f'{y}_std'].item()

            if 'upper_bound' not in x_value:
                y_max = max(y_max, y_value)
            bar_width = filtered_data[width].mean() / (max_width * 1.1)

            if legend_labels is not None and legend_colors is not None:
                color = lut[hue_value]
            ax.bar(bar_centers[i, j], y_value, color=color, width=bar_width, alpha=1 if not use_color else 0.4)#, yerr=y_std)
            if std:
                ax.errorbar(bar_centers[i, j], y_value, yerr=y_std, fmt='none', ecolor='black', capsize=capsize, elinewidth=0.5)
        ax.axhline(y=y_max, color=color, linestyle='dashed')

    ax.set_xticks(np.arange(len(unique_x)))
    ax.set_xticklabels(unique_x, fontsize=28)
    ax.set_xlabel(x, fontsize=28)
    ax.set_ylabel(y, fontsize=28)

    if legend_labels is not None and legend_colors is not None:
        legend_elements = [Patch(facecolor=lut[hue_value], label=f'{hue_value}: {int(width_value)}') for hue_value, width_value in zip(unique_hue, unique_width)]
    else:
        legend_elements = [Patch(facecolor=color, label=f'{hue_value}: {int(width_value)}') for hue_value, width_value, color in zip(unique_hue, unique_width, colors)]
    #ax.legend(handles=legend_elements, title=hue)
    return legend_elements


def compare_performance(dbc,
                        threshold_1 = 1.5, 
                        threshold_2 = None, 
                        split_col = 'hbonds', 
                        split_col_2 = None, 
                        measurement = 'ddG',
                        statistic = 'MCC',
                        statistic_2 = None,
                        n_bootstraps = 100,
                        count_proteins = False,
                        count_muts = False,
                        subset = None,
                        grouper = 'cluster',
                        noise = 0,
                        duplicates = False,
                        order = None,
                        plots = 'both',
                        drop_label = False,
                        asterisk = (),
                        double_asterisk = (),
                        split_first = True,
                        split_last = True,
                        legend_loc = 'lower left',
                        saveloc_perf = '../data/extended/figure_data/perf_data.csv',
                        saveloc_dist = '../data/extended/figure_data/dist_data.csv'
                        ):

    rename_dict = {'delta_kdh': 'Δ hydrophobicity', 'delta_vol': 'Δ volume', 'rel_ASA': 'relative ASA', 'neff': 'N eff. seqs'}
    title_prefix = rename_dict.get(split_col, split_col)
    if split_col_2 is not None:
        title_prefix += f' and {rename_dict.get(split_col_2, split_col_2)}'
    
    if statistic_2 is None:
        statistic_2 = statistic

    #font = {'size'   : 20}
    #matplotlib.rc('font', **font)

    if plots == 'both':
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15,17.25), sharex=True, dpi=300)
    elif plots == 'top':
        fig, ax1 = plt.subplots(1, 1, figsize=(15,10), dpi=300)
    elif plots == 'bottom':
        fig, ax2 = plt.subplots(1, 1, figsize=(15,10), dpi=300)

    fig.patch.set_facecolor('white')
    sns.set_palette('tab10')

    db_complete = dbc.copy(deep=True)
    if subset is not None:
        db_complete = db_complete[subset + [c for c in dbc.columns if '_dir' not in c]]

    db_complete = db_complete.dropna(subset=measurement)
    if drop_label:
        db_complete = db_complete.drop('ddG_dir', axis=1)

    dbs_bs = bootstrap_by_grouper(db_complete, n_bootstraps, grouper=grouper, noise=noise, drop=False, duplicates=duplicates)

    # Ungrouped performance (doesn't change)
    for i in range(n_bootstraps):
        if i == 0:
            #full = compute_stats(db_complete.sample(frac=1, replace=True), measurements=[measurement], stats=[statistic] + ['n'])
            full = compute_stats(dbs_bs[0], measurements=[measurement], stats=[statistic] + ['n'], grouper=grouper, quiet=True)
        else:
            #ungrouped = compute_stats(db_complete.sample(frac=1, replace=True), measurements=[measurement], stats=[statistic] + ['n'], quiet=True)
            ungrouped = compute_stats(dbs_bs[i], measurements=[measurement], stats=[statistic] + ['n'], quiet=True, grouper=grouper)
            cur = ungrouped.reset_index()[['n', 'model', 'class', statistic_2] \
                + ([f'n_proteins_{statistic}'] if count_proteins else []) \
                + ([f'n_muts_{statistic}'] if count_muts else [])]
            full = full.merge(cur, on=['model', 'class'], suffixes=('', f'_{i}'))
            #print(full)

    full = full.rename({statistic_2: statistic_2+'_0', 'n': 'n_0'}, axis=1)
    full[f'{statistic_2}_mean'] = full[[f'{statistic_2}_{i}' for i in range(n_bootstraps)]].mean(axis=1)
    if count_proteins:
        full = full.rename({f'n_proteins_{statistic}': f'n_proteins_{statistic}_0'}, axis=1)
        full['n'] = full[[f'n_proteins_{statistic}_{i}' for i in range(n_bootstraps)]].mean(axis=1).astype(int)
    elif count_muts:
        full = full.rename({f'n_muts_{statistic}': f'n_muts_{statistic}_0'}, axis=1)
        full['n'] = full[[f'n_muts_{statistic}_{i}' for i in range(n_bootstraps)]].mean(axis=1).astype(int)
    else:
        full['n'] = full[[f'n_{i}' for i in range(n_bootstraps)]].mean(axis=1).astype(int)
    full[f'{statistic_2}_std'] = full[[f'{statistic_2}_{i}' for i in range(n_bootstraps)]].std(axis=1)
    
    if statistic_2 == 'ndcg':
        full[f'{statistic_2}_mean'] = 100**full[f'{statistic_2}_mean']
        full[f'{statistic_2}_std'] = 100**full[f'{statistic_2}_std']

    #full = full.drop(statistic_2, axis=1)
    ungrouped = full#.rename({f'{statistic_2}_mean': statistic_2}, axis=1)
    #print(ungrouped)
    ungrouped0 = ungrouped
    ungrouped0 = ungrouped0.sort_values(f'{statistic_2}_mean', ascending=False)

    if order is not None:
        if not drop_label and not 'ddG_dir' in order:
            order = pd.concat([pd.Series(['ddG_dir']), order])
        ungrouped0 = ungrouped0.set_index('model').loc[order, :].reset_index()

    # Unnormalized split performance
    #c = compute_stats(db_complete, split_col=split_col, split_col_2=split_col_2, split_val=threshold_1, split_val_2=threshold_2, measurements=[measurement], stats=[statistic_2] + ['n', 'tp', 'tn', 'fp', 'fn'])
    
    full = pd.DataFrame()
    for i in tqdm(range(n_bootstraps)):
        if i == 0:
            #full = compute_stats(db_complete.sample(frac=1, replace=True),
            full = compute_stats(dbs_bs[0],
                split_col=split_col, split_col_2=split_col_2, split_val=threshold_1, split_val_2=threshold_2, measurements=[measurement], 
                    stats=[statistic] + ['n'], grouper=grouper, quiet=True, split_first=split_first, split_last=split_last)
        else:
            #c = compute_stats(db_complete.sample(frac=1, replace=True), quiet=True,
            c = compute_stats(dbs_bs[i], quiet=True,
                split_col=split_col, split_col_2=split_col_2, split_val=threshold_1, split_val_2=threshold_2, measurements=[measurement],
                    stats=[statistic] + ['n'], grouper=grouper, split_first=split_first, split_last=split_last)
            cur = c.reset_index()[['n', 'model', 'class', statistic_2]
                            + ([f'n_proteins_{statistic}'] if count_proteins else []) \
                            + ([f'n_muts_{statistic}'] if count_muts else [])]
            full = full.merge(cur, on=['model', 'class'], suffixes=('', f'_{i}'))
            
    full = full.rename({statistic_2: statistic_2+'_0', 'n': 'n_0'}, axis=1)
    full[f'{statistic_2}_mean'] = full[[f'{statistic_2}_{i}' for i in range(n_bootstraps)]].mean(axis=1)
    if count_proteins:
        full = full.rename({f'n_proteins_{statistic}': f'n_proteins_{statistic}_0'}, axis=1)
        full['n'] = full[[f'n_proteins_{statistic}_{i}' for i in range(n_bootstraps)]].mean(axis=1).astype(int)
    elif count_muts:
        full = full.rename({f'n_muts_{statistic}': f'n_muts_{statistic}_0'}, axis=1)
        full['n'] = full[[f'n_muts_{statistic}_{i}' for i in range(n_bootstraps)]].mean(axis=1).astype(int)
        print(full['n'].mean())
        print(full[[f'n_{i}' for i in range(n_bootstraps)]].mean(axis=1).astype(int).mean())
    else:
        full['n'] = full[[f'n_{i}' for i in range(n_bootstraps)]].mean(axis=1).astype(int)
    full[f'{statistic_2}_std'] = full[[f'{statistic_2}_{i}' for i in range(n_bootstraps)]].std(axis=1)

    if statistic_2 == 'ndcg':
        full[f'{statistic_2}_mean'] = 100**full[f'{statistic_2}_mean']
        full[f'{statistic_2}_std'] = 100**full[f'{statistic_2}_std']

    splits = full.set_index('model') 
    splits = splits.loc[ungrouped0['model']]#.reset_index()
    splits = splits.loc[ungrouped0['model']].reset_index() #.loc[splits['measurement']=='ddG']

    dbc = db_complete.copy(deep=True)

    if split_col_2 is not None and not split_first:
        dbc[f'{split_col} > {threshold_1}'] = dbc[split_col] > threshold_1
        dbc[f'{split_col} <= {threshold_1} & {split_col_2} > {threshold_2}'] = (dbc[split_col] <= threshold_1) & (dbc[split_col_2] > threshold_2)
        dbc[f'{split_col} <= {threshold_1} & {split_col_2} <= {threshold_2}'] = (dbc[split_col] <= threshold_1) & (dbc[split_col_2] <= threshold_2)
        vvs = [f'{split_col} <= {threshold_1} & {split_col_2} <= {threshold_2}',
               f'{split_col} <= {threshold_1} & {split_col_2} > {threshold_2}',
               f'{split_col} > {threshold_1}']
    elif split_col_2 is not None and not split_last:
        dbc[f'{split_col_2} > {threshold_2}'] = dbc[split_col_2] > threshold_2
        dbc[f'{split_col} > {threshold_1} & {split_col_2} <= {threshold_2}'] = (dbc[split_col] > threshold_1) & (dbc[split_col_2] <= threshold_2)
        dbc[f'{split_col} <= {threshold_1} & {split_col_2} <= {threshold_2}'] = (dbc[split_col] <= threshold_1) & (dbc[split_col_2] <= threshold_2)
        vvs = [f'{split_col} <= {threshold_1} & {split_col_2} <= {threshold_2}',
               f'{split_col} > {threshold_1} & {split_col_2} <= {threshold_2}',
               f'{split_col_2} > {threshold_2}']
    elif split_col_2 is not None:
        dbc[f'{split_col} > {threshold_1} & {split_col_2} > {threshold_2}'] = (dbc[split_col] > threshold_1) & (dbc[split_col_2] > threshold_2)
        dbc[f'{split_col} <= {threshold_1} & {split_col_2} > {threshold_2}'] = (dbc[split_col] <= threshold_1) & (dbc[split_col_2] > threshold_2)
        dbc[f'{split_col} > {threshold_1} & {split_col_2} <= {threshold_2}'] = (dbc[split_col] > threshold_1) & (dbc[split_col_2] <= threshold_2)
        dbc[f'{split_col} <= {threshold_1} & {split_col_2} <= {threshold_2}'] = (dbc[split_col] <= threshold_1) & (dbc[split_col_2] <= threshold_2)
        vvs = [f'{split_col} <= {threshold_1} & {split_col_2} <= {threshold_2}',
               f'{split_col} <= {threshold_1} & {split_col_2} > {threshold_2}',
               f'{split_col} > {threshold_1} & {split_col_2} <= {threshold_2}',
               f'{split_col} > {threshold_1} & {split_col_2} > {threshold_2}']
    elif threshold_2 is None:
        dbc[f'{split_col} > {threshold_1}'] = dbc[split_col] > threshold_1
        dbc[f'{split_col} <= {threshold_1}'] = dbc[split_col] <= threshold_1
        vvs = [ f'{split_col} <= {threshold_1}', f'{split_col} > {threshold_1}']
    else:
        dbc[f'{split_col} > {threshold_1}'] = dbc[split_col] > threshold_1
        dbc[f'{threshold_1} >= {split_col} > {threshold_2}'] = (dbc[split_col] <= threshold_1) & (dbc[split_col] > threshold_2)
        dbc[f'{split_col} <= {threshold_2}'] = dbc[split_col] <= threshold_2
        vvs = [f'{split_col} <= {threshold_2}', f'{threshold_1} >= {split_col} > {threshold_2}', f'{split_col} > {threshold_1}',]

    print(vvs)
    dbc = dbc.dropna(subset=measurement)
    dbc = dbc.melt(id_vars=dbc.columns.drop(vvs), value_vars=vvs, value_name='value_')
    dbc = dbc.loc[dbc['value_']].rename({'variable':'split'}, axis=1)
    vvs2 = ungrouped0['model'].unique()

    dbc = dbc.melt(id_vars=['split'], value_vars=vvs2)
    std = dbc.groupby('variable')['value'].transform('std')
    dbc['value'] /= std
    ungrouped1 = pd.DataFrame()
    for key in ungrouped0['model'].unique():
        subset = dbc.loc[dbc['variable']==key]
        ungrouped1 = pd.concat([ungrouped1, subset])

    categories = ungrouped0['model'].unique()

    if plots not in ['bottom', 'both']:
        ax2 = fig.add_axes([0, 0, 0, 0], visible=False, sharex=ax1)

    full_quants = ungrouped1.groupby(['split', 'variable']).count().groupby('split').first()
    # Scaffold-wise predicted distribution
    my_palette = ['#d01c8b','#f1b6da','#4dac26','#b8e186'] #["#34aeeb", "#eb9334", "#3452eb", "#eb4634"]
    legend_elements = sns.boxplot(data=ungrouped1,x='variable',y='value',hue=f'split',ax=ax2, palette=my_palette).legend_
                        # split=True if threshold_2 is None else False, bw=0.2, cut=0,
    labels = [t.get_text() for t in legend_elements.texts]
    colors = [c.get_facecolor() for c in legend_elements.legendHandles]

    ax2.set_title(title_prefix + ' scaffold-wise predicted distribution', fontsize=28)
    ax2.set_ylabel('stability or Δ log-likelihood', fontsize=28)
    ax2.set_xlabel('')
    ax2.grid()
    ax2.yaxis.grid(False)
    ax2.axhline(y=0, color='r', linestyle='dashed')

    splits_out = splits.reset_index().drop(['index', 'runtime'], axis=1)
    splits_out['model'] = splits_out['model'].apply(lambda x: remap_names_2.get(x, x))
    splits_out.to_csv(saveloc_perf, encoding='utf-8-sig')
    dbc_out = dbc.copy(deep=True).rename({'split': 'split', 'variable': 'model', 'value': 'prediction'}, axis=1)
    dbc_out['model'] = dbc_out['model'].apply(lambda x: remap_names_2.get(x, x))
    dbc_out.to_csv(saveloc_dist, encoding='utf-8-sig')

    if plots in ['top', 'both']:
        legend_elements = custom_barplot(data=splits.drop_duplicates(), x='model', y=statistic_2, hue='class', width='n', ax=ax1, legend_colors=colors, legend_labels=labels)
        _ = custom_barplot(data=ungrouped.reset_index().set_index('model').loc[ungrouped0['model']].drop_duplicates().reset_index(), 
                    x='model', y=statistic_2, hue='class', width='n', ax=ax1, use_color='grey', std=False) #capsize=12

        statistic_ = {'weighted_ndcg': 'wNDCG', 'weighted_spearman': 'wρ', 'weighted_auprc': 'wAUPRC', 'auprc': 'AUPRC', 'spearman': 'Spearman\'s ρ', 'PPV': 'PPV', 'accuracy': 'Accuracy', 'MCC': 'MCC'}[statistic_2]
        ax1.set_title(title_prefix + ' scaffold-wise mean performance', fontsize=28) #'Delta vs. split mean'
        ax1.set_ylabel(statistic_, fontsize=28)
        ax1.grid()
        ax1.yaxis.grid(False)
        ax1.set_xlabel('')

    new_legend_elements = []
    new_legend_elements_lower = []

    import copy

    try:
        legend_elements[0]
    except TypeError:
        legend_elements = [legend_elements]
    for legend_element in legend_elements:
        original_label = legend_element.get_label()
        new_label = legend_element.get_label()
        legend_element_2 = copy.deepcopy(legend_element)

        if plots in ['both', 'bottom']:
            cat = new_label.split(':')[0]
            num = new_label.split(':')[-1]
            print(new_label)
            new_label = new_label.replace(num, f' {full_quants.loc[cat].item()}')
            print(new_label)
            legend_element_2.set_label(new_label)

        if 'conservation' in original_label:
            original_label = original_label.split(':')
            original_label = original_label[0] + '% :' + original_label[1]
            legend_element.set_label(original_label)

            new_label = new_label.split(':')
            new_label = new_label[0] + '% :' + new_label[1]
            legend_element_2.set_label(new_label)

        for key in rename_dict.keys():
            if key in original_label:
                original_label = original_label.replace(key, rename_dict[key])
                new_label = new_label.replace(key, rename_dict[key])
                # Update label if it exists in the dictionary
                #if original_label in rename_dict:
                legend_element.set_label(original_label)
                legend_element_2.set_label(new_label)
        
        new_legend_elements.append(legend_element)
        new_legend_elements_lower.append(legend_element_2)
    
        #print('nle', new_legend_elements)

    if plots == 'both':
        ax1.legend(handles=new_legend_elements, loc=legend_loc, fontsize=22)
        ax2.legend(handles=new_legend_elements_lower, loc='lower left', fontsize=22)
        ax1.set_yticklabels(ax1.get_yticklabels(), fontsize=28)
        ax2.set_yticklabels(ax2.get_yticklabels(), fontsize=28)
    if plots == 'top':
        ax1.set_xticks(ax1.get_xticks(), categories, rotation=45, ha='right', fontsize=24)
        for tick_label in ax1.get_xticklabels():
            tick_label.set_color(determine_base_color(tick_label))
        ax1.legend(handles=new_legend_elements, loc=legend_loc, fontsize=22)
        ax1.set_yticklabels(ax1.get_yticklabels(), fontsize=28)
    elif plots in ['both', 'bottom']:
        ax2.set_xticks(ax2.get_xticks(), categories, rotation=45, ha='right', fontsize=24)
        for tick_label in ax2.get_xticklabels():
            tick_label.set_color(determine_base_color(tick_label))
            
        #print([tick.get_text() for tick in ax2.get_xticklabels()])
        #ax2.legend(handles=new_legend_elements, loc=legend_loc)
    
    ax = ax1 if plots in ['both', 'top'] else ax2
    remapped_x = [remap_names_2[tick.get_text()] if tick.get_text() in remap_names_2.keys() else tick.get_text() for tick in ax2.get_xticklabels()]
    remapped_x = [x+'*' if x in asterisk else x for x in remapped_x]
    remapped_x = [x+'**' if x in double_asterisk else x for x in remapped_x]
    ax.set_xticklabels(remapped_x)

    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)

    print(splits.groupby('class')['n'].mean().astype(int))
    plt.show()
    return splits, ungrouped0['model'], fig


def stack_frames(dbf):
    db_stack = dbf.copy(deep=True)
    df = db_stack.melt(var_name='pred_col', value_name='value', value_vars=db_stack.columns, ignore_index=False)
    df = df.reset_index()
    df.loc[df['pred_col'].str.contains('dir'), 'direction'] = 'dir'
    df.loc[df['pred_col'].str.contains('inv'), 'direction'] = 'inv'
    over = len(df.loc[df['pred_col'].str.contains('dir') & df['pred_col'].str.contains('inv')])
    if over != 0:
        df.loc[~df['pred_col'].str.contains('dir') & df['pred_col'].str.contains('inv')]
        print('Dropped', over, 'columns with errors')
    df = df.set_index(['direction', 'uid'])  # Assuming 'index' is the name of uid column
    df['pred_col'] = df['pred_col'].str.replace('_dir', '')
    df['pred_col'] = df['pred_col'].str.replace('_inv', '')
    dbf = df.pivot_table(index=['direction', 'uid'], columns='pred_col', values='value')
    return dbf


def extract_decimal_number(text):
    # Define a regular expression to match decimal numbers
    decimal_regex = r"\d+\.\d+"
    
    # Search for the decimal number in the text using the regular expression
    match = re.search(decimal_regex, text)
    
    # If a match is found, return the float value of the matched string
    if match:
        return float(match.group())
    
    # If no match is found, return NaN
    return np.nan


def compute_stats_bidirectional(db_gt_preds, split_col=None, split_val=None, split_col_2=None, split_val_2=None, stats=(), directions=('dir', 'inv'), stacked=False, grouper='code'):

    '''Summarizes the response of all models on one or two feature splits'''

    db_internal = db_gt_preds.copy(deep=True)
    if not stacked:
        db_grouper = db_internal[[grouper]]
    else:
        db_grouper = db_internal.reset_index('direction', drop=True).reset_index().groupby('uid').first()[[grouper]]
    #db_ddg = db_internal[['ddG']]
    db_internal = db_internal.drop(grouper, axis=1)
    
    #if split_col is not None:
    #    print(db_discrete[db_discrete[split_col].isna()])

    if split_col_2 is None and split_val_2 is not None:
        split_col_2 = split_col
    if split_col is not None:
        db_internal = db_internal.copy(deep=True).dropna(subset=[split_col])
    if split_col_2 is not None and split_col_2 != split_col:
        db_internal = db_internal.copy().dropna(subset=[split_col_2])

    if (split_col is None) or (split_val is None):
        split_col = 'tmp'
        split_val = 0
    if (split_col_2 is None) or (split_val_2 is None):
        split_col_2 = 'tmp2'
        split_val_2 = 0

    db_discrete = db_internal.copy(deep=True).astype(float)
    #f ddG{("_"+direction) if not stacked else ""} gets flipped by convention, such that positivef ddG{("_"+direction) if not stacked else ""} means destabilizing
    #db_discrete[f'ddG{("_"+direction) if not stacked else ""}'] = -db_discrete[f'ddG{("_"+direction) if not stacked else ""}']
    
    if not stacked:
        # binarize the ddG{("_"+direction) if not stacked else ""} column: above 0 (or equal to) means destabilized (0), below means stabilized
        if 'dir' in directions:
            db_discrete.loc[db_discrete['ddG_dir'] > 0, 'ddG_dir'] = 1
            db_discrete.loc[db_discrete['ddG_dir'] < 0, 'ddG_dir'] = 0
        if 'inv' in directions:
            db_discrete.loc[db_discrete['ddG_inv'] > 0, 'ddG_inv'] = 1
            db_discrete.loc[db_discrete['ddG_inv'] < 0, 'ddG_inv'] = 0     
    else:
        db_discrete.loc[db_discrete['ddG'] > 0, 'ddG'] = 1
        db_discrete.loc[db_discrete['ddG'] < 0, 'ddG'] = 0

    cols = db_discrete.columns #.drop([f'ddG{("_"+direction) if not stacked else ""}'])
    
    db_complete_bin = db_discrete.copy(deep=True)

    # binarize all columns (needed for about half of the statistics to be calculated)
    if split_col != 'tmp':
        db_complete_bin = db_discrete.copy(deep=True).dropna(subset=[split_col]).drop(split_col, axis=1).astype(float)
    if split_col_2 != 'tmp2' and split_col_2 != split_col:
        db_complete_bin = db_complete_bin.dropna(subset=[split_col_2]).drop(split_col_2, axis=1).astype(float)

    # more likely means predicted stabilized
    db_complete_bin[db_complete_bin > 0] = 1
    db_complete_bin[db_complete_bin < 0] = 0

    if split_col != 'tmp':
        db_complete_bin = db_complete_bin.join(db_discrete[[split_col]])
    if split_col_2 != 'tmp2':
        db_complete_bin = db_complete_bin.join(db_discrete[[split_col_2]])

    # avoid grouping columns at all
    if split_col == 'tmp' and split_col_2 == 'tmp2':
        split = ['']
    # only one type of column, so just want higher or lower than value
    elif split_col_2 == 'tmp2':
        split = [f'{split_col} > {split_val}', f'{split_col} <= {split_val}']
    # if we specify two of the same split column we are assuming there are two different threshold values
    elif split_col == split_col_2:
        split = [f'{split_col} > {split_val}', f'{split_val} >= {split_col} > {split_val_2}', f'{split_col} <= {split_val_2}']
    # if we specify two different splits we need all permutations
    else:
        split = [f'{split_col} > {split_val} & {split_col_2} > {split_val_2}', 
                 f'{split_col} <= {split_val} & {split_col_2} > {split_val_2}',
                 f'{split_col} > {split_val} & {split_col_2} <= {split_val_2}',
                 f'{split_col} <= {split_val} & {split_col_2} <= {split_val_2}']

    # create a multi-index such that each row looks like ('dir','mpnn_dir'): statistics
    idx = pd.MultiIndex.from_product([['dir', 'inv', 'combined'], split, cols])
    df_out = pd.DataFrame(index=idx)
    for direction in directions: 
        for sp in split:
            # reset on each iteration
            current_binary_split = db_complete_bin.copy(deep=True)
            current_continuous_split = db_discrete.copy(deep=True)
            current_full_split = db_gt_preds.copy(deep=True) 

            # invert the ground truth label when predicting the inverse mutation
            #if direction == 'inv':
                #current_binary_split[f'ddG{("_"+direction) if not stacked else ""}'] = 1 - current_binary_split[f'ddG{("_"+direction) if not stacked else ""}']
                #current_continuous_split[f'ddG{("_"+direction) if not stacked else ""}'] = 1 - current_continuous_split[f'ddG{("_"+direction) if not stacked else ""}']
                #current_full_split[f'ddG{("_"+direction) if not stacked else ""}'] = 1 - current_continuous_split[f'ddG{("_"+direction) if not stacked else ""}']
            # invert the change to a property when predicting the inverse mutation
            if 'delta' in split_col and direction == 'inv':
                current_binary_split[split_col] = -current_binary_split[split_col]
                current_continuous_split[split_col] = -current_continuous_split[split_col]
                current_full_split = -current_continuous_split[split_col]
            # don't invert the second column if it is the same as the first (would cancel out)
            if 'delta' in split_col_2 and direction == 'inv' and split_col != split_col_2:
                current_binary_split[split_col_2] = -current_binary_split[split_col_2]
                current_continuous_split[split_col_2] = -current_continuous_split[split_col_2]
                current_full_split = -current_continuous_split[split_col_2]
            
            # case where the columns are both specified and different
            if split_col != 'tmp' and split_col_2 != 'tmp2' and split_col != split_col_2:
                
                # if the first column is > threshold (& is to specify the simultanous conditions)
                if '>' in sp.split('&')[0]:
                    # only get the slice where this column is greater than its threshold
                    current_binary_split = current_binary_split.loc[current_binary_split[split_col] > split_val]
                    current_continuous_split = current_continuous_split.loc[current_continuous_split[split_col] > split_val]
                    current_full_split = current_full_split.loc[current_full_split[split_col] > split_val]
                elif '<=' in sp.split('&')[0]:
                    # get the complement
                    current_binary_split = current_binary_split.loc[current_binary_split[split_col] <= split_val]
                    current_continuous_split = current_continuous_split.loc[current_continuous_split[split_col] <= split_val]
                    current_full_split = current_full_split.loc[current_full_split[split_col] <= split_val]
                # if the second column is > threshold
                if '>' in sp.split('&')[1]:
                    # only get the slice where this column is greater than its threshold
                    current_binary_split = current_binary_split.loc[current_binary_split[split_col_2] > split_val_2]
                    current_continuous_split = current_continuous_split.loc[current_continuous_split[split_col_2] > split_val_2]
                    current_full_split = current_full_split.loc[current_full_split[split_col_2] > split_val_2]
                elif '<=' in sp.split('&')[1]:
                    # get the complement
                    current_binary_split = current_binary_split.loc[current_binary_split[split_col_2] <= split_val_2]
                    current_continuous_split = current_continuous_split.loc[current_continuous_split[split_col_2] <= split_val_2]
                    current_full_split = current_full_split.loc[current_full_split[split_col_2] <= split_val_2]

            # case where the columns are the same, and specified (most common case)
            elif split_col == split_col_2 and split_col != 'tmp':
                
                # case where we are only getting the highest values, and there should be two thresholds such that there are also intermediate and low values
                if ('>' in sp and not '>=' in sp):
                    current_binary_split = current_binary_split.loc[current_binary_split[split_col] > split_val]
                    current_continuous_split = current_continuous_split.loc[current_continuous_split[split_col] > split_val]
                    current_full_split = current_full_split.loc[current_continuous_split[split_col] > split_val]
                # lowest values case
                elif '<=' in sp:
                    # since split_col = split_col_2 we can reference split_col but we want to use split_val_2 which is the lower split_val
                    current_binary_split = current_binary_split.loc[current_binary_split[split_col] <= split_val_2]
                    current_continuous_split = current_continuous_split.loc[current_continuous_split[split_col] <= split_val_2]
                    current_full_split = current_full_split.loc[current_full_split[split_col] <= split_val_2]
                # intermediate values case
                else:
                    print('case intermediate')
                    # greater than the low split_val but less than or equal to the higher split_val
                    current_binary_split = current_binary_split.loc[current_binary_split[split_col] > split_val_2].loc[current_binary_split[split_col] <= split_val]
                    current_continuous_split = current_continuous_split.loc[current_continuous_split[split_col] > split_val_2].loc[current_continuous_split[split_col] <= split_val]
                    current_full_split = current_full_split.loc[current_full_split[split_col] > split_val_2].loc[current_full_split[split_col] <= split_val]

            # case where there is only one split columns
            elif split_col_2 == 'tmp2' and split_col != 'tmp':
                if '>' in sp:
                    # only get the slice where this column is greater than its threshold
                    current_binary_split = current_binary_split.loc[current_binary_split[split_col] > split_val]
                    current_continuous_split = current_continuous_split.loc[current_continuous_split[split_col] > split_val]
                    current_full_split = current_full_split.loc[current_full_split[split_col] > split_val]
                else:
                    # get the complement
                    current_binary_split = current_binary_split.loc[current_binary_split[split_col] <= split_val]
                    current_continuous_split = current_continuous_split.loc[current_continuous_split[split_col] <= split_val]  
                    current_full_split = current_full_split.loc[current_full_split[split_col] <= split_val]
            
            # only calculate statistics for predictions that correspond to the direction specified
            if not stacked:
                current_pred_cols = [pred_col for pred_col in cols if direction in pred_col]
            else:
                current_pred_cols = cols
            for pred_col in tqdm(current_pred_cols):
                #print(pred_col)
                # only want to consider not-NA values, which we will count as well
                true_col = f'ddG{("_"+direction) if not stacked else ""}'
                if not stacked:
                    current_binary_predictions = current_binary_split[[pred_col,true_col]].dropna().T.drop_duplicates().T
                else:
                    current_binary_predictions = current_binary_split.xs(direction, level=0) #[current_binary_split.index.str.contains('_'+direction)]
                    current_binary_predictions = current_binary_predictions[[pred_col,'ddG']].dropna().T.drop_duplicates().T
                    if len(current_binary_predictions.columns) == 2:
                        current_binary_predictions.columns = [pred_col, true_col]
                    else:
                        current_binary_predictions.columns = [true_col]

                # note - this will change for each column to maximize values assessed (not always exactly apples-to-apples)
                if 'n' in stats or stats == ():
                    df_out.loc[(direction,sp,pred_col), 'n'] = len(current_binary_predictions)
                #try:
                #    tn, fp, fn, tp = metrics.confusion_matrix(current_binary_predictions[true_col], current_binary_predictions[pred_col]).ravel()
                #except Exception as e:
                #    print(pred_col)
                #    print(e)
                #    continue
                try:
                    tn, fp, fn, tp = metrics.confusion_matrix(current_binary_predictions[true_col], current_binary_predictions[pred_col]).ravel()
                # there are no values in this column, probably because the inverse was not calculated.
                except:
                    print(direction,sp,pred_col)
                    continue
                if 'tp' in stats or stats == ():
                    df_out.loc[(direction,sp,pred_col), 'tp'] = tp
                if 'fp' in stats or stats == ():
                    df_out.loc[(direction,sp,pred_col), 'fp'] = fp
                if 'tn' in stats or stats == ():
                    df_out.loc[(direction,sp,pred_col), 'tn'] = tn 
                if 'fn' in stats or stats == ():  
                    df_out.loc[(direction,sp,pred_col), 'fn'] = fn   
                if 'sensitivity' in stats or stats == (): 
                    df_out.loc[(direction,sp,pred_col), 'sensitivity'] = tp/(tp+fn)
                if 'specificity' in stats or stats == ():         
                    df_out.loc[(direction,sp,pred_col), 'specificity'] = tn/(tn+fp)
                if 'PPV' in stats or stats == (): 
                    df_out.loc[(direction,sp,pred_col), 'PPV'] = tp/(tp+fp)
                if 'pred_positives_ratio' in stats or stats == ():
                    df_out.loc[(direction,sp,pred_col), 'pred_positives_ratio'] = (tp+fp)/(tp+fn)
                if 'accuracy' in stats or stats == (): 
                    df_out.loc[(direction,sp,pred_col), 'accuracy'] = metrics.accuracy_score(current_binary_predictions[true_col], current_binary_predictions[pred_col])
                if 'f1_score' in stats or stats == (): 
                    df_out.loc[(direction,sp,pred_col), 'f1_score'] = metrics.f1_score(current_binary_predictions[true_col], current_binary_predictions[pred_col])
                if 'MCC' in stats or stats == ():
                    df_out.loc[(direction,sp,pred_col), 'MCC'] = metrics.matthews_corrcoef(current_binary_predictions[true_col], current_binary_predictions[pred_col])

                if not stacked:
                    current_continuous_predictions = current_continuous_split[[pred_col,f'ddG_{direction}']].dropna().T.drop_duplicates().T
                else:
                    current_continuous_predictions = current_continuous_split.xs(direction, level=0) #.loc[current_continuous_split.index.str.contains(direction)]
                    current_continuous_predictions = current_continuous_predictions[[pred_col,'ddG']].dropna().T.drop_duplicates().T
                    if len(current_continuous_predictions.columns) == 2:
                        current_continuous_predictions.columns = [pred_col, true_col]
                    else:
                        current_continuous_predictions.columns = [true_col]

                auroc = metrics.roc_auc_score(current_continuous_predictions[true_col], current_continuous_predictions[pred_col])
                auprc = metrics.average_precision_score(current_continuous_predictions[true_col], current_continuous_predictions[pred_col])
                if 'auroc' in stats or stats == (): 
                    df_out.loc[(direction,sp,pred_col), 'auroc'] = auroc
                if 'auprc' in stats or stats == (): 
                    df_out.loc[(direction,sp,pred_col), 'auprc'] = auprc   

                topk = current_continuous_predictions.sort_values(pred_col, ascending=False).index
                stable_ct = len(current_continuous_predictions.loc[current_continuous_predictions[true_col] > 0])

                if 'n_stable' in stats or stats == ():
                    df_out.loc[(direction,sp,pred_col), 'n_stable'] = stable_ct

                # precision of the top-k predicted-most-stable proteins across the whole slice of data
                for stat in [s for s in stats if 'precision@' in s] if stats != [] else ['precision@k']:
                    k = stat.split('@')[-1]
                    if k == 'k':
                        # precision @ k is the fraction of the top k predictions that are actually stabilizing, 
                        # where k is the number of stabilizing mutations in the slice.
                        df_out.loc[(direction,sp,pred_col), 'precision@k'] = current_continuous_predictions.loc[topk[:stable_ct], true_col].sum() / stable_ct
                    else:
                        k = int(k)
                        if k > stable_ct:
                            print('The number of stabilizing mutations is fewer than k')
                        df_out.loc[(direction,sp,pred_col), stat] = current_continuous_predictions.loc[topk[:k], true_col].sum() / min(k, stable_ct)

                    #topn = current_continuous_predictions.sort_values(true_col, ascending=False).index
                    #df_out.loc[(direction,sp,pred_col), 'sensitivity @ k'] = current_continuous_predictions.loc[topn[:stable_ct], pred_col].sum() / stable_ct
                
                if not stacked:
                    current_full_predictions = current_full_split[[pred_col,f'ddG_{direction}']].dropna().T.drop_duplicates().T
                else:
                    current_full_predictions = current_full_split.xs(direction, level=0) #.loc[current_full_split.index.str.contains('_dir')]
                    current_full_predictions = current_full_predictions[[pred_col,'ddG']].dropna().T.drop_duplicates().T
                    if len(current_full_predictions.columns) == 2:
                        current_full_predictions.columns = [pred_col, true_col]
                    else:
                        current_full_predictions.columns = [true_col]

                #current_full_predictions['code'] = list(current_full_predictions.reset_index()['uid'].str[:4])
                len1 = len(current_full_predictions)
                current_full_predictions = current_full_predictions.join(db_grouper)
                assert len(current_full_predictions[grouper].dropna()) == len1

                # recall of the top-k predicted-most-stable proteins across the whole slice of data
                for stat in [s for s in stats if 'recall@' in s] if stats != () else ['recall@k0.0', 'recall@k1.0']:
                    k = stat.split('@')[-1].strip('k')
                    if k == '':
                        k = 0.
                    else:
                        k = float(k)
                    
                    pred_df_discrete_k = current_full_predictions.copy(deep=True).drop_duplicates()
                    pred_df_discrete_k[true_col] = pred_df_discrete_k[true_col].apply(lambda x: 1 if x > k else 0)
                    stable_ct = pred_df_discrete_k[true_col].sum()

                    gain = current_full_predictions.loc[current_full_predictions[true_col] > k, true_col].sum()
                    #print(stable_ct)
                    #print(stable_ct)
                    df_out.loc[(direction,sp,pred_col), f'{k}_n_stable'] = stable_ct
                
                    sorted_preds = pred_df_discrete_k.sort_values(pred_col, ascending=False).index
                    df_out.loc[(direction,sp,pred_col), f'recall@k{k}'] = pred_df_discrete_k.loc[sorted_preds[:stable_ct], true_col].sum() / stable_ct
                    df_out.loc[(direction,sp,pred_col), f'gain@k{k}'] = current_full_predictions.drop_duplicates().loc[(sorted_preds[:stable_ct]), true_col].sum() / gain

                if 'mean_stabilization' in stats or stats == ():
                    df_out.loc[(direction,sp,pred_col), 'mean_stabilization'] = current_full_predictions.loc[current_full_predictions[pred_col]>0, true_col].mean()
                if 'net_stabilization' in stats or stats == ():
                    df_out.loc[(direction,sp,pred_col), 'net_stabilization'] = current_full_predictions.loc[current_full_predictions[pred_col]>0, true_col].sum()
                if 'mean_stable_pred' in stats or stats == ():
                    df_out.loc[(direction,sp,pred_col), 'mean_stable_pred'] = current_full_predictions.loc[current_full_predictions[true_col]>0, pred_col].mean()

                if ('t1s' in stats) or (stats == ()): 
                    top_1_stab = 0
                    for code, group in current_full_predictions.groupby(grouper):
                        top_1_stab += group.sort_values(pred_col, ascending=False)[true_col].head(1).item()

                    df_out.loc[(direction,sp,pred_col), 'mean_t1s'] = top_1_stab / len(current_full_predictions[grouper].unique())

                # inverse of the assigned rank of the number one most stable protein per group
                if ('mean_reciprocal_rank' in stats) or (stats == ()): 
                    reciprocal_rank_sum = 0
                    unique_groups = current_full_predictions[grouper].unique()
                    for code, group in current_full_predictions.groupby(grouper):
                        group = group.drop_duplicates()
                        sorted_group = group.sort_values(pred_col, ascending=False)
                        highest_meas_rank = sorted_group[true_col].idxmax()

                        rank_of_highest_meas = sorted_group.index.get_loc(highest_meas_rank)
                        if type(rank_of_highest_meas) == slice:
                            print('Something went wrong with MRR for', col, code)
                            continue

                        rank_of_highest_meas += 1
                        reciprocal_rank_sum += 1 / rank_of_highest_meas

                    mean_reciprocal_rank = reciprocal_rank_sum / len(unique_groups)
                    df_out.loc[(direction, sp, pred_col), 'mean_reciprocal_rank'] = mean_reciprocal_rank

                if ('ndcg' in stats) or (stats == ()):
                    df_out.loc[(direction,sp,pred_col), 'ndcg'] = compute_ndcg(current_full_predictions, pred_col, true_col)
                    cum_ndcg = 0
                    w_cum_ndcg = 0
                    cum_d = 0
                    w_cum_d = 0
                    for _, group in current_full_predictions.groupby(grouper): 
                        if len(group.loc[group[true_col]>0]) > 1 and not all(group[true_col]==group[true_col][0]):
                            cur_ndcg = compute_ndcg(group, pred_col, true_col)
                            # can happen if there are no stable mutants
                            if np.isnan(cur_ndcg):
                                continue
                            cum_ndcg += cur_ndcg
                            cum_d += 1
                            w_cum_ndcg += cur_ndcg * np.log(len(group.loc[group[true_col]>0]))
                            w_cum_d += np.log(len(group.loc[group[true_col]>0]))
                    df_out.loc[(direction,sp,pred_col), 'mean_ndcg'] = cum_ndcg / (cum_d if cum_d > 0 else 1)
                    #print(pred_col)
                    #print(w_cum_ndcg)
                    df_out.loc[(direction,sp,pred_col), 'weighted_ndcg'] = w_cum_ndcg / (w_cum_d if w_cum_d > 0 else 1)
                    #if np.isnan(df_out.loc[(direction,sp,pred_col), 'weighted_ndcg']):
                    #    assert False
                if ('pearson' in stats) or (stats == ()):
                    whole_r, _ = pearsonr(current_full_predictions[pred_col], current_full_predictions[true_col])
                    df_out.loc[(direction,sp,pred_col), 'pearson'] = whole_r

                if ('spearman' in stats) or (stats == ()):
                    whole_p, _ = spearmanr(current_full_predictions[pred_col], current_full_predictions[true_col])
                    df_out.loc[(direction,sp,pred_col), 'spearman'] = whole_p
                    cum_p = 0
                    w_cum_p = 0
                    cum_d = 0
                    w_cum_d = 0
                    for code, group in current_full_predictions.groupby(grouper): 
                        if len(group) > 1 and not all(group[true_col]==group[true_col][0]):
                            spearman, _ = spearmanr(group[pred_col], group[true_col])
                            if np.isnan(spearman):
                                #print(code, group[[pred_col, true_col]])
                                spearman = 0
                            cum_p += spearman
                            cum_d += 1
                            w_cum_p += spearman * np.log(len(group))
                            w_cum_d += np.log(len(group))
                    df_out.loc[(direction,sp,pred_col), 'mean_spearman'] = cum_p / (cum_d if cum_d > 0 else 1)
                    #print(pred_col)
                    #print(w_cum_p)
                    df_out.loc[(direction,sp,pred_col), 'weighted_spearman'] = w_cum_p / (w_cum_d if w_cum_d > 0 else 1)
                    #if df_out.loc[(direction,sp,pred_col), 'weighted_spearman'].isna():
                    #    assert False

                #current_continuous_predictions['code'] = list(current_continuous_predictions.reset_index()['uid'].str[:4])
                len1 = len(current_continuous_predictions)
                current_continuous_predictions = current_continuous_predictions.join(db_grouper)
                assert len(current_continuous_predictions[grouper].dropna()) == len1

                if ('auprc' in stats) or (stats == ()):
                    cum_ps = 0
                    w_cum_ps = 0
                    cum_d = 0
                    w_cum_d = 0
                    for _, group in current_continuous_predictions.groupby(grouper): 
                        if len(group) > 1 and not all(group[true_col]==group[true_col][0]):
                            cur_ps = metrics.average_precision_score(group[true_col], group[pred_col])
                            if np.isnan(cur_ps):
                                continue
                            cum_ps += cur_ps
                            cum_d += 1
                            w_cum_ps += cur_ps * np.log(len(group))
                            w_cum_d += np.log(len(group))

                    df_out.loc[(direction,sp,pred_col), 'mean_auprc'] = cum_ps / (cum_d if cum_d > 0 else 1)
                    df_out.loc[(direction,sp,pred_col), 'weighted_auprc'] = w_cum_ps / (w_cum_d if cum_d > 0 else 1)

                if split_col == 'tmp':
                    if ('auppc' in stats) or (stats == ()):
                        percentiles = [str(int(s))+'%' for s in range(1, 100)]
                    else:
                        percentiles = [s for s in stats if '%' in s]

                    percentile_values = [int(s.split('%')[0]) for s in percentiles]

                    # Apply the function to each group and reset the index
                    results_df = current_full_predictions.groupby(grouper).apply(
                        calculate_ppc, pred_col=pred_col, meas=true_col, percentile_values=percentile_values
                        ).reset_index()

                    stat_dict = {}
                    # Aggregate results
                    for stat in percentiles:
                        try:
                            stat_dict[stat] = results_df[stat].sum() / results_df[f"pos_{stat}"].sum()
                        except ZeroDivisionError:
                            stat_dict[stat] = 0

                    # Assign to df_out
                    df_out.loc[(direction, sp, pred_col), percentiles] = list(stat_dict.values())
                    df_out.loc[(direction, sp, pred_col), 'auppc'] = df_out.loc[(direction, sp, pred_col), percentiles].mean()

                    #if ('auppc' in stats) or (stats == ()):
                    #   df_out.loc[(direction,sp,pred_col), 'auppc'] = df_out.loc[(direction,sp,pred_col), [c for c in df_out.columns if '%' in c and not 'stab_' in c]].mean()

                    # mean stability vs prediction percentile curve
                    if ('aumsc' in stats) or (stats == ()):
                        percentiles = [str(int(s))+'$' for s in range(1, 100)]
                    else:
                        percentiles = [s for s in stats if '$' in s]
                    percentile_values = [int(s.split('$')[0]) for s in percentiles]

                    # Apply the function to each group and reset the index
                    results_df = current_full_predictions.groupby(grouper).apply(
                        calculate_msc, pred_col=pred_col, meas=true_col, percentile_values=percentile_values
                        ).reset_index()

                    stat_dict = {}
                    # Aggregate results
                    for stat in percentiles:
                        try:
                            stat_dict[stat] = results_df[stat].sum() / results_df[f"pos_{stat}"].sum()
                        except ZeroDivisionError:
                            stat_dict[stat] = 0

                    # Assign to df_out
                    df_out.loc[(direction, sp, pred_col), percentiles] = list(stat_dict.values())
                    df_out.loc[(direction, sp, pred_col), 'aumsc'] = df_out.loc[(direction, sp, pred_col), percentiles].mean()

                # only get combined data for entries which have an inverse since they are more likely to have a corresponding forward as well
                if direction == 'inv':
                    try:
                        if not stacked:
                            pred_col_comb = pred_col[:-4]
                            current_binary_predictions = current_binary_split.loc[:, [pred_col_comb+'_dir',pred_col_comb+'_inv','ddG_dir', 'ddG_inv']].dropna().T.drop_duplicates().T
                            current_continuous_predictions = current_continuous_split.loc[:, [pred_col_comb+'_dir',pred_col_comb+'_inv','ddG_dir', 'ddG_inv']].dropna().T.drop_duplicates().T
                            current_full_predictions = current_full_split.loc[:, [pred_col_comb+'_dir',pred_col_comb+'_inv','ddG_dir', 'ddG_inv']].dropna().T.drop_duplicates().T
                        else:
                            pred_col_comb = pred_col
                            current_binary_predictions = current_binary_split.loc[:, [pred_col_comb,pred_col_comb,'ddG']].dropna().T.drop_duplicates().T
                            current_continuous_predictions = current_continuous_split.loc[:, [pred_col_comb,pred_col_comb,'ddG']].dropna().T.drop_duplicates().T
                            current_full_predictions = current_full_split.loc[:, [pred_col_comb,pred_col_comb,'ddG']].dropna().T.drop_duplicates().T

                    except KeyError as e:
                        print(e)
                        continue

                    if not stacked:
                        current_continuous_predictions = stack_frames(current_continuous_predictions)
                        current_binary_predictions = stack_frames(current_binary_predictions)
                        current_full_predictions = stack_frames(current_full_predictions)

                    current_continuous_predictions = current_continuous_predictions.join(db_grouper)
                    current_full_predictions = current_full_predictions.join(db_grouper)

                    fwd = current_full_predictions.xs('dir', level=0)[pred_col_comb]
                    #fwd.index = fwd.index.str[:-4]
                    fwd.columns = [pred_col_comb+'_dir']
                    rvs = current_full_predictions.xs('inv', level=0)[pred_col_comb]
                    #rvs.index = rvs.index.str[:-4]
                    rvs.columns = [pred_col_comb+'_inv']

                    df_out.loc[('combined',sp,pred_col_comb), 'antisymmetry'] = antisymmetry(fwd, rvs)
                    df_out.loc[('combined',sp,pred_col_comb), 'bias'] = bias(fwd, rvs)
                    
                    if 'n' in stats or stats == ():
                        df_out.loc[('combined',sp,pred_col_comb), 'n'] = len(current_binary_predictions)
                    try:
                        tn, fp, fn, tp = metrics.confusion_matrix(current_binary_predictions[['ddG']], current_binary_predictions[pred_col_comb]).ravel()
                    except Exception as e:
                        print(pred_col_comb)
                        print(pd.concat([current_binary_predictions[['ddG']], current_binary_predictions[pred_col_comb]], axis=1))
                        #print(e)
                        continue
                    if 'tp' in stats or stats == ():
                        df_out.loc[('combined',sp,pred_col_comb), 'tp'] = tp
                    if 'fp' in stats or stats == ():
                        df_out.loc[('combined',sp,pred_col_comb), 'fp'] = fp
                    if 'tn' in stats or stats == ():
                        df_out.loc[('combined',sp,pred_col_comb), 'tn'] = tn 
                    if 'fn' in stats or stats == ():  
                        df_out.loc[('combined',sp,pred_col_comb), 'fn'] = fn   
                    if 'sensitivity' in stats or stats == (): 
                        df_out.loc[('combined',sp,pred_col_comb), 'sensitivity'] = tp/(tp+fn)
                    if 'specificity' in stats or stats == ():         
                        df_out.loc[('combined',sp,pred_col_comb), 'specificity'] = tn/(tn+fp)
                    if 'PPV' in stats or stats == (): 
                        df_out.loc[('combined',sp,pred_col_comb), 'PPV'] = tp/(tp+fp)
                    if 'pred_positives_ratio' in stats or stats == ():
                        df_out.loc[('combined',sp,pred_col_comb), 'pred_positives_ratio'] = (tp+fp)/(tp+fn)
                    if 'accuracy' in stats or stats == (): 
                        df_out.loc[('combined',sp,pred_col_comb), 'accuracy'] = metrics.accuracy_score(current_binary_predictions[['ddG']], current_binary_predictions[pred_col_comb])
                    if 'f1_score' in stats or stats == (): 
                        df_out.loc[('combined',sp,pred_col_comb), 'f1_score'] = metrics.f1_score(current_binary_predictions[['ddG']], current_binary_predictions[pred_col_comb])
                    if 'MCC' in stats or stats == ():
                        df_out.loc[('combined',sp,pred_col_comb), 'MCC'] = metrics.matthews_corrcoef(current_binary_predictions[['ddG']], current_binary_predictions[pred_col_comb])

                    current_continuous_predictions = current_continuous_predictions[[pred_col_comb,'ddG']].dropna().T.drop_duplicates().T
                    auroc = metrics.roc_auc_score(current_continuous_predictions['ddG'], current_continuous_predictions[pred_col_comb])
                    auprc = metrics.average_precision_score(current_continuous_predictions['ddG'], current_continuous_predictions[pred_col_comb])

                    if 'auroc' in stats or stats == (): 
                        df_out.loc[('combined',sp,pred_col_comb), 'auroc'] = auroc
                    if 'auprc' in stats or stats == (): 
                        df_out.loc[('combined',sp,pred_col_comb), 'auprc'] = auprc   

                    topk = current_continuous_predictions.sort_values(pred_col_comb, ascending=False).index
                    stable_ct = len(current_continuous_predictions.loc[current_continuous_predictions['ddG'] > 0])

                    if 'n_stable' in stats or stats == ():
                        df_out.loc[('combined',sp,pred_col_comb), 'n_stable'] = stable_ct

                    # precision of the top-k predicted-most-stable proteins across the whole slice of data
                    for stat in [s for s in stats if 'precision@' in s] if stats != [] else ['precision@k']:
                        k = stat.split('@')[-1]
                        if k == 'k':
                            # precision @ k is the fraction of the top k predictions that are actually stabilizing, 
                            # where k is the number of stabilizing mutations in the slice.
                            df_out.loc[('combined',sp,pred_col_comb), 'precision@k'] = current_continuous_predictions.loc[topk[:stable_ct], 'ddG'].sum() / stable_ct
                        else:
                            k = int(k)
                            if k > stable_ct:
                                print('The number of stabilizing mutations is fewer than k')
                            df_out.loc[('combined',sp,pred_col_comb), stat] = current_continuous_predictions.loc[topk[:k], 'ddG'].sum() / min(k, stable_ct)

                    #current_full_predictions['code'] = list(current_full_predictions.reset_index()['uid'].str[:4])
                    #current_full_predictions = current_full_predictions.join(db_grouper)
                                    # recall of the top-k predicted-most-stable proteins across the whole slice of data
                    for stat in [s for s in stats if 'recall@' in s] if stats != () else ['recall@k0.0', 'recall@k1.0']:
                        k = stat.split('@')[-1].strip('k')
                        if k == '':
                            k = 0.
                        else:
                            k = float(k)
                        
                        pred_df_discrete_k = current_full_predictions.copy(deep=True).drop_duplicates()
                        pred_df_discrete_k['ddG'] = pred_df_discrete_k['ddG'].apply(lambda x: 1 if x > k else 0)
                        stable_ct = pred_df_discrete_k['ddG'].sum()

                        gain = current_full_predictions.loc[current_full_predictions['ddG'] > k, 'ddG'].sum()
                        df_out.loc[('combined',sp,pred_col_comb), f'{k}_n_stable'] = stable_ct
                    
                        sorted_preds = pred_df_discrete_k.sort_values(pred_col_comb, ascending=False).index
                        df_out.loc[('combined',sp,pred_col_comb), f'recall@k{k}'] = pred_df_discrete_k.loc[sorted_preds[:stable_ct], 'ddG'].sum() / stable_ct
                        df_out.loc[('combined',sp,pred_col_comb), f'gain@k{k}'] = current_full_predictions.drop_duplicates().loc[(sorted_preds[:stable_ct]), 'ddG'].sum() / gain

                    if 'mean_stabilization' in stats or stats == ():
                        df_out.loc[('combined',sp,pred_col_comb), 'mean_stabilization'] = current_full_predictions.loc[current_full_predictions[pred_col_comb]>0, 'ddG'].mean()
                    if 'net_stabilization' in stats or stats == ():
                        df_out.loc[('combined',sp,pred_col_comb), 'net_stabilization'] = current_full_predictions.loc[current_full_predictions[pred_col_comb]>0, 'ddG'].sum()
                    if 'mean_stable_pred' in stats or stats == ():
                        df_out.loc[('combined',sp,pred_col_comb), 'mean_stable_pred'] = current_full_predictions.loc[current_full_predictions['ddG']>0, pred_col_comb].mean()

                    if ('t1s' in stats) or (stats == ()): 
                        top_1_stab = 0
                        for code, group in current_full_predictions.groupby(grouper):
                            top_1_stab += group.sort_values(pred_col_comb, ascending=False)['ddG'].head(1).item()

                        df_out.loc[('combined',sp,pred_col_comb), 'mean_t1s'] = top_1_stab / len(current_full_predictions[grouper].unique())

                    # inverse of the assigned rank of the number one most stable protein per group
                    if ('mean_reciprocal_rank' in stats) or (stats == ()): 
                        reciprocal_rank_sum = 0
                        unique_groups = current_full_predictions[grouper].unique()
                        for code, group in current_full_predictions.groupby(grouper):
                            group = group.drop_duplicates()
                            sorted_group = group.sort_values(pred_col, ascending=False)
                            highest_meas_rank = sorted_group[true_col].idxmax()

                            rank_of_highest_meas = sorted_group.index.get_loc(highest_meas_rank)
                            if type(rank_of_highest_meas) == slice:
                                print('Something went wrong with MRR for', col, code)
                                continue

                            rank_of_highest_meas += 1
                            reciprocal_rank_sum += 1 / rank_of_highest_meas

                        mean_reciprocal_rank = reciprocal_rank_sum / len(unique_groups)
                        df_out.loc[('combined', sp, pred_col_comb), 'mean_reciprocal_rank'] = mean_reciprocal_rank

                    if ('ndcg' in stats) or (stats == ()):
                        df_out.loc[('combined',sp,pred_col_comb), 'ndcg'] = compute_ndcg(current_full_predictions, pred_col_comb, 'ddG')
                        cum_ndcg = 0
                        w_cum_ndcg = 0
                        cum_d = 0
                        w_cum_d = 0
                        for _, group in current_full_predictions.groupby(grouper): 
                            if len(group) > 1 and not all(group['ddG']==group['ddG'][0]):
                                cur_ndcg = compute_ndcg(group, pred_col_comb, 'ddG')
                                # can happen if there are no stable mutants
                                if np.isnan(cur_ndcg):
                                    continue
                                cum_ndcg += cur_ndcg
                                cum_d += 1
                                w_cum_ndcg += cur_ndcg * np.log(len(group))
                                w_cum_d += np.log(len(group))
                        df_out.loc[('combined',sp,pred_col_comb), 'mean_ndcg'] = cum_ndcg / (cum_d if cum_d > 0 else 1)
                        df_out.loc[('combined',sp,pred_col_comb), 'weighted_ndcg'] = w_cum_ndcg / (w_cum_d if w_cum_d > 0 else 1)

                    if ('pearson' in stats) or (stats == ()):
                        whole_r, _ = pearsonr(current_full_predictions[pred_col_comb], current_full_predictions['ddG'])                    
                        df_out.loc[('combined',sp,pred_col_comb), 'pearson'] = whole_r

                    if ('spearman' in stats) or (stats == ()):
                        whole_p, _ = spearmanr(current_full_predictions[pred_col_comb], current_full_predictions['ddG'])
                        df_out.loc[('combined',sp,pred_col_comb), 'spearman'] = whole_p
                        cum_p = 0
                        w_cum_p = 0
                        cum_d = 0
                        w_cum_d = 0
                        for code, group in current_full_predictions.groupby(grouper): 
                            if len(group) > 1 and not all(group['ddG']==group['ddG'][0]):
                                spearman, _ = spearmanr(group[pred_col_comb], group['ddG'])
                                if np.isnan(spearman):
                                    #print('combined')
                                    #print(code, group[[pred_col_comb, 'ddG']])
                                    spearman = 0
                                cum_p += spearman
                                cum_d += 1
                                w_cum_p += spearman * np.log(len(group))
                                w_cum_d += np.log(len(group))
                        df_out.loc[('combined',sp,pred_col_comb), 'mean_spearman'] = cum_p / (cum_d if cum_d > 0 else 1)
                        df_out.loc[('combined',sp,pred_col_comb), 'weighted_spearman'] = w_cum_p / (w_cum_d if w_cum_d > 0 else 1)

                    #current_continuous_predictions['code'] = list(current_continuous_predictions.reset_index()['uid'].str[:4])
                    current_continuous_predictions = current_continuous_predictions.join(db_grouper)
                    
                    if ('auprc' in stats) or (stats == ()):
                        cum_ps = 0
                        w_cum_ps = 0
                        cum_d = 0
                        w_cum_d = 0
                        for _, group in current_continuous_predictions.groupby(grouper): 
                            if len(group) > 1:
                                cur_ps = metrics.average_precision_score(group['ddG'], group[pred_col_comb])
                                if np.isnan(cur_ps):
                                    continue
                                cum_ps += cur_ps
                                cum_d += 1
                                w_cum_ps += cur_ps * np.log(len(group))
                                w_cum_d += np.log(len(group))

                        df_out.loc[('combined',sp,pred_col_comb), 'mean_auprc'] = cum_ps / (cum_d if cum_d > 0 else 1)
                        df_out.loc[('combined',sp,pred_col_comb), 'weighted_auprc'] = w_cum_ps / (w_cum_d if cum_d > 0 else 1)

                    if split_col == 'tmp':
                        if ('auppc' in stats) or (stats == ()):
                            percentiles = [str(int(s))+'%' for s in range(1, 100)]
                        else:
                            percentiles = [s for s in stats if '%' in s]

                        percentile_values = [int(s.split('%')[0]) for s in percentiles]

                        # Apply the function to each group and reset the index
                        results_df = current_full_predictions.groupby(grouper).apply(
                            calculate_ppc, pred_col=pred_col_comb, meas='ddG', percentile_values=percentile_values
                            ).reset_index()

                        stat_dict = {}
                        # Aggregate results
                        for stat in percentiles:
                            try:
                                stat_dict[stat] = results_df[stat].sum() / results_df[f"pos_{stat}"].sum()
                            except ZeroDivisionError:
                                stat_dict[stat] = 0

                        # Assign to df_out
                        df_out.loc[('combined', sp, pred_col_comb), percentiles] = list(stat_dict.values())
                        df_out.loc[('combined', sp, pred_col_comb), 'auppc'] = df_out.loc[('combined', sp, pred_col_comb), percentiles].mean()

                        if ('aumsc' in stats) or (stats == ()):
                            percentiles = [str(int(s))+'$' for s in range(1, 100)]
                        else:
                            percentiles = [s for s in stats if '$' in s]
                        percentile_values = [int(s.split('$')[0]) for s in percentiles]

                        # Apply the function to each group and reset the index
                        results_df = current_full_predictions.groupby(grouper).apply(
                            calculate_msc, pred_col=pred_col_comb, meas='ddG', percentile_values=percentile_values
                            ).reset_index()

                        stat_dict = {}
                        # Aggregate results
                        for stat in percentiles:
                            try:
                                stat_dict[stat] = results_df[stat].sum() / results_df[f"pos_{stat}"].sum()
                            except ZeroDivisionError:
                                stat_dict[stat] = 0

                        # Assign to df_out
                        df_out.loc[('combined', sp, pred_col_comb), percentiles] = list(stat_dict.values())
                        df_out.loc[('combined', sp, pred_col_comb), 'aumsc'] = df_out.loc[('combined', sp, pred_col_comb), percentiles].mean()

    df_out = df_out.reset_index()
    df_out = df_out.rename({'pred_col': 'level_2'}, axis=1)

    df_out = df_out.rename({'level_0': 'direction', 'level_1': 'class', 'level_2': 'model'}, axis=1)
    df_out['model_type'] = 'unknown'
    df_out['model_type'] = df_out['model'].map(determine_category)

    df_out = df_out.set_index(['direction', 'model_type', 'model', 'class'])
    df_out = df_out.sort_index(level=1).sort_index(level=0)
    #df_out.to_csv('../../zeroshot suppl/class_na_2.csv')

    return df_out.dropna(how='all')


def custom_recursive_feature_addition(df_train, dfs_test, cols, target, model, lines='cartesian_ddg_dir', fillna=False, source='unknown', max_feats=8, bs=0, 
                                      saveloc_upper='../data/extended/figure_data/data_upper.csv', saveloc_lower='../data/extended/figure_data/data_lower.csv'):
       
    remaining_features = deepcopy(cols)
    selected_features = []
    spearman_scores = []
    if bool(bs):
        column_tuples = [(name[0].upper() + name[1:], str(i)) for name in dfs_test.keys() for i in range(bs)]
        multiindex_columns = pd.MultiIndex.from_tuples(column_tuples, names=['dataset', 'bootstrap'])
        test_scores = pd.DataFrame(columns=multiindex_columns, index=range(1, len(cols)+1))
        lines_scores = pd.DataFrame(index=multiindex_columns, columns=[lines])
    else:
        test_scores = pd.DataFrame(columns=[name[0].upper() + name[1:] for name in dfs_test.keys()], index=range(1, len(cols)+1))
        lines_scores = pd.DataFrame(index=[name[0].upper() + name[1:] for name in dfs_test.keys()], columns=[lines])
    weights_df = pd.DataFrame(columns=cols, index=range(1, len(cols)+1))

    scaler = StandardScaler()
    df_train[cols] = scaler.fit_transform(df_train[cols])
    y_train = df_train[target]

    if bs:
         for name_, df_test_list in dfs_test.items():
            print(name_)
            for rep in range(bs):
                df_test = df_test_list[rep]
                name = name_[0].upper() + name_[1:]
                score, _ = spearmanr(df_test[lines], df_test[target])
                lines_scores.at[(name, rep), lines] = score       
    else:
        # Calculate the Spearman correlation for the 'lines' column
        for name_, df_test in dfs_test.items():
            name = name_[0].upper() + name_[1:]
            score, _ = spearmanr(df_test[lines], df_test[target])
            lines_scores.at[name, lines] = score

    while len(remaining_features) > 0:
        best_score = -np.inf
        best_feature = None

        for feature in remaining_features:
            current_features = selected_features + [feature]
            X_train = df_train[current_features]

            model_clone = clone(model)  # Create a fresh clone of the model
            model_clone.fit(X_train, y_train)

            predictions = model_clone.predict(X_train)
            score, _ = spearmanr(predictions, y_train)

            if score > best_score:
                best_score = score
                best_feature = feature
                if hasattr(model_clone, 'coef_'):
                    best_weights = model_clone.coef_.flatten()
                elif hasattr(model_clone, 'feature_importances_'):
                    best_weights = model_clone.feature_importances_
                else:
                    best_weights = [0] * len(current_features)

        remaining_features.remove(best_feature)
        selected_features.append(best_feature)
        spearman_scores.append(best_score)
        weights_df.loc[len(selected_features), selected_features] = best_weights

        #print(f"Added {best_feature}, Spearman Correlation: {best_score}")

        if bs:
            for name_, df_test_list in dfs_test.items():
                for rep in range(bs):
                    df_test = df_test_list[rep]
                    name = name_[0].upper() + name_[1:]
                    X_train = df_train[selected_features]
                    model_clone = clone(model)
                    model_clone.fit(X_train, y_train)

                    df_new = deepcopy(df_test)
                    df_new[cols] = scaler.transform(df_new[cols])
                    if fillna:
                        X_new = df_new[selected_features].fillna(0)
                    else:
                        X_new = df_new[selected_features]
                    y_new = df_new[target]
                    predictions = model_clone.predict(X_new)
                    score, _ = spearmanr(predictions, y_new)
                    test_scores.at[len(selected_features), (name, rep)] = score
                    df_test[f'{source}_rfa_{len(selected_features)}'] = predictions
        else:
            for name_, df_test in dfs_test.items():
                name = name_[0].upper() + name_[1:]
                X_train = df_train[selected_features]
                model_clone = clone(model)
                model_clone.fit(X_train, y_train)

                df_new = deepcopy(df_test)
                df_new[cols] = scaler.transform(df_new[cols])
                if fillna:
                    X_new = df_new[selected_features].fillna(0)
                else:
                    X_new = df_new[selected_features]
                y_new = df_new[target]
                predictions = model_clone.predict(X_new)
                score, _ = spearmanr(predictions, y_new)
                test_scores.at[len(selected_features), name] = score
                df_test[f'{source}_rfa_{len(selected_features)}'] = predictions
    
    # Figure C section (weights used for training dataset)
    weights_df_out = weights_df.copy(deep=True)
    weights_df_out.columns = [remap_names_2.get(c, c) for c in weights_df_out.columns]
    weights_df_out.to_csv(saveloc_upper, encoding='utf-8-sig')
    weights_df = weights_df.loc[weights_df.index <= max_feats].dropna(how='all', axis=1)

    weights_df_melted = weights_df.reset_index().melt(id_vars='index')
    wdf_ = weights_df_melted.sort_values(by=['index', 'value'], ascending=[True, False]).dropna(subset='value')
    hue_order_ = wdf_.dropna(subset='value')['variable'].unique()

    wdf = []
    hue_order = []

    for k, v in mapping_categories.items():
        for hue in hue_order_:
            if any([c in hue for c in v]):
                #print(hue, v)
                wdf.append(wdf_.loc[wdf_['variable']==hue])
                hue_order.append(hue)

    wdf = pd.concat(wdf)
    #print(wdf)
    # Create color mapping
    color_mapping = get_color_mapping(wdf, 'variable')

    #fig, axes = plt.subplots(2,1, figsize=(12, 16), sharex=True, dpi=300)
    #ax1 = axes[0]
    #ax2 = axes[1]

    # Set up figure with a gridspec to handle layout
    fig = plt.figure(figsize=(12, 16), dpi=300)
    gs = gridspec.GridSpec(2, 3, height_ratios=[5, 5], width_ratios=[1, 5, 1], hspace=0.0, wspace=0.0)

    ax_c = fig.add_subplot(gs[0, 1])

    # Create the barplot with specified order
    ax = sns.barplot(data=wdf, hue='variable', x='index', y='value', hue_order=hue_order, ax=ax_c)
    
    #ax_c.set_title('Regression Weights on Whole Dataset', fontsize=20)
    ax_c.set_xticks([])
    ax_c.set_xticklabels([])
    ax_c.set_xlabel('')
    ax_c.set_ylabel('Regression Weights', fontsize=18, labelpad=10)
    ax_c.set_yticks(ax_c.get_yticks()[1:])
    ax_c.set_title(f'RFA on {source.split("_")[0]}', pad=10, fontsize=24)
    
    # Get the x-ticks positions
    num_hues = len(hue_order)
    num_x_categories = len(wdf['index'].unique())
    bar_width = ax.patches[0].get_width()

    # Iterate over the bars
    for patch, row in zip(ax.patches, wdf['variable']):
        hue_value = row
        # Apply color based on the hue_value
        patch.set_facecolor(color_mapping[hue_value])

    # Remove the original legend
    #ax.legend_.remove()

    # Create a new legend
    #print([remap_names_2[var] for var in wdf['variable'].unique()])
    legend_patches = [mpatches.Patch(color=color_mapping[var], label=remap_names_2.get(var, var)) for var in wdf['variable'].unique()]
    
    ax_c.legend(handles=legend_patches, title='', fontsize=16, loc='lower left' if bs else 'upper right') #bbox_to_anchor=(0., 0.4), #loc='lower left',
    ax_c.set_xlim(-0.5, max_feats - 0.5)
    #ax_c.set_xlabel('Number of Features', fontsize=20)

    # Define colors for the line plots
    colors = ['black', 'purple', 'red', 'green']

    test_scores.index.name = 'n_features'
    test_scores.dropna(how='all', axis=1).T.to_csv(saveloc_lower)

    # Figure D section (lines for performance vs features)
    if bs:
        test_scores_ = test_scores.reset_index(drop=True)
        test_scores_ = test_scores_.apply(pd.to_numeric, errors='coerce')
        lines_scores = lines_scores.apply(pd.to_numeric, errors='coerce')

        # Calculate the median and interquartile range (IQR) across the 'Suffix' level for each 'dataset'
        median_scores = test_scores_.groupby(axis=1, level='dataset').median()
        first_quartile_scores = test_scores_.groupby(axis=1, level='dataset').quantile(0.25)
        third_quartile_scores = test_scores_.groupby(axis=1, level='dataset').quantile(0.75)

        median_benchmarks = lines_scores.groupby(axis=0, level='dataset').median()
        first_quartile_benchmarks = lines_scores.groupby(axis=0, level='dataset').quantile(0.25)
        third_quartile_benchmarks = lines_scores.groupby(axis=0, level='dataset').quantile(0.75)

        ax_marg_l = fig.add_subplot(gs[1, 0])
        ax_main = fig.add_subplot(gs[1, 1])
        ax_marg_r = fig.add_subplot(gs[1, 2])

        for i, (dataset_name, dataset_data) in enumerate(median_scores.iteritems()):
            if i < 4:  # Limit to first 4 datasets if necessary
                # Plot the median line
                sns.lineplot(x=median_scores.index + 1, y=dataset_data, ax=ax_main,
                            color=colors[i], label=f'{dataset_name} Median', marker='o')
                
                # Add fill between with very light fill
                ax_main.fill_between(median_scores.index + 1, 
                                    first_quartile_scores[dataset_name], 
                                    third_quartile_scores[dataset_name], color=colors[i], alpha=0.1)
                
                # Add lines at the edges of the fill_between area for first and third quartiles
                ax_main.plot(median_scores.index + 1, first_quartile_scores[dataset_name], color=colors[i], alpha=0.1)
                ax_main.plot(median_scores.index + 1, third_quartile_scores[dataset_name], color=colors[i], alpha=0.1)
                
                # Add a horizontal line for the median benchmark
                ax_main.axhline(median_benchmarks.at[dataset_name, lines], color=colors[i], linestyle='--')

                # Extract values for current dataset from lines_scores and plot KDE
                benchmark_values = lines_scores.xs(dataset_name, level='dataset')
                sns.kdeplot(benchmark_values.iloc[:, 0], ax=ax_marg_l, vertical=True, color=colors[i], label=f'{dataset_name} Distribution')

                # Extract values for current dataset from lines_scores and plot KDE
                final_values = test_scores.T.xs(dataset_name, level='dataset').T
                sns.kdeplot(final_values.iloc[max_feats, :], ax=ax_marg_r, vertical=True, color=colors[i], label=f'{dataset_name} Distribution')

        # Clean up Gaussian plot area
        ax_marg_r.set_xticks([])
        ax_marg_r.set_yticks([])
        ax_marg_r.set_xlabel(f'{max_feats}-Feat Ens', labelpad=30, fontsize=18)
        ax_marg_r.set_ylabel('')
        #ax_marg_r.spines['top'].set_visible(False)
        #ax_marg_r.spines['right'].set_visible(False)
        #ax_marg_r.spines['left'].set_visible(False)
        #ax_marg_r.spines['bottom'].set_visible(False)

        ax_marg_l.set_xticks([])
        #ax_marg_l.set_yticks([])
        ax_marg_l.set_xlabel('CartDDG' if lines=='cartesian_ddg_dir' else 'UNK', labelpad=30, fontsize=18)
        ax_marg_l.set_ylabel('')
        #ax_marg_l.spines['top'].set_visible(False)
        #ax_marg_l.spines['right'].set_visible(False)
        #ax_marg_l.spines['left'].set_visible(False)
        #ax_marg_l.spines['bottom'].set_visible(False)
        ax_marg_l.invert_xaxis()
        ax_marg_l.set_ylabel('Test Spearman\'s ρ', fontsize=18, labelpad=10)

        ax_main.set_xlabel('Number of Features', fontsize=18)
        ax_main.set_ylabel('')
        ax_main.set_yticks([])
        #ax_main.set_ylabel('Test Spearman\'s Rho', labelpad=60, fontsize=15)
        #ax_main.set_title('Bootstrapped Test Set Spearman\'s Rho', fontsize=20)
        ax_main.set_xlim(0.5, max_feats + 0.5)

        ax_main.set_ylim((0.3,0.9))
        ax_marg_r.set_ylim(ax_main.get_ylim())
        ax_marg_l.set_ylim(ax_main.get_ylim())

        ax_main.legend(fontsize=16, loc='upper left')
        ax_main.set_xticks(range(1, max_feats+1))
        ax_main.set_xticklabels(range(1, max_feats+1))
            
    else:
        ax_main = fig.add_subplot(gs[1, 1])
        test_scores_ = test_scores.reset_index(drop=True)
        for i, name in enumerate(test_scores.columns):
            if i < 4:
                sns.lineplot(data=test_scores_[name], ax=ax_main, color=colors[i], label=f'{name} Ensemble')
        ax_main.set_xlim(-0.5, max_feats-0.5)
        current_ticks = ax_main.get_xticks()  # Get current x-tick locations
        new_tick_positions = [int(tick + 1) for tick in current_ticks if tick + 1 <= max_feats]  # Shift ticks by 1, within bounds
        ax_main.set_xticklabels(new_tick_positions)  # Set new tick positions

        # Plot the horizontal lines for the 'lines' column
        for i, name in enumerate(lines_scores.index):
            if i < 4:
                ax_main.axhline(lines_scores.at[name, lines], color=colors[i], linestyle='--', label=f'{name} Benchmark')
            
        ax_main.legend(fontsize=12)
        ax_main.set_ylabel('Test Spearman\'s ρ')

    #ax_c.set_xlim(-0.5, max_feats - 0.5) 
    #ax1.legend(loc='upper left', bbox_to_anchor=(1, 1))

    xticks = ax_main.get_xticks()

    # For each x-tick, draw a vertical line at the halfway point to the next x-tick
    for i in range(len(xticks) - 1):
        mid_point = (xticks[i] + xticks[i+1]) / 2 - 1
        ax_c.axvline(x=mid_point, color='gray', linestyle='--', lw=0.5, alpha=0.5)
        ax_main.axvline(x=mid_point + 1, color='gray', linestyle='--', lw=0.5, alpha=0.5)

    plt.tight_layout()
    #plt.suptitle('Recursive Feature Addition', fontsize=20)
    plt.subplots_adjust(top=0.95)
    plt.show()

    return test_scores, weights_df, dfs_test, fig


def compare_distributions_boxes(dfs, shared_columns=None, sources=None, kind='strip'):
    """
    Generate a distribution plot for selected columns from multiple DataFrames with different hues for each DataFrame,
    including annotations for min and max values for each feature and a zero-line annotation.

    Parameters:
        dfs (list of pd.DataFrame): List of DataFrames to compare.
        shared_columns (list of str): List of column names to include in the comparison. If None, computes intersection of all DataFrames.
        sources (list of str): Labels for the sources corresponding to each DataFrame in `dfs`.
        kind (str): One of strip, violin or boxen to specify the plot type

    Returns:
        None: Displays a distribution plot.
    """
    remap_names = {'delta_chg_dir': 'Δ charge', 'delta_vol_dir': 'Δ volume', 'hbonds': 'wt hydrogen bonds', 
                'multimer': 'chains in assembly', 'neff': 'n effective sequences', 'b_factor': 'beta factor',
                'rel_ASA_dir': 'SASA', 'delta_kdh_dir': 'Δ hydrophobicity', 'ddG': 'ΔΔG', 'structure_length': 'n residues', 
                'completeness_score': 'alignment completeness', 'log_neff': 'log(N effective seqs)', 'conservation': 'conservation (%)'}

    if shared_columns is None:
        # Compute the intersection of columns across all DataFrames
        shared_columns = set(dfs[0].columns)
        for df in dfs[1:]:
            shared_columns.intersection_update(df.columns)
        shared_columns = list(shared_columns)

    # Exclude boolean columns
    shared_columns = [col for col in shared_columns if all(df[col].dtype != 'bool' for df in dfs)]

    # Create a combined dataframe with an extra column to indicate the source dataframe
    combined_df = pd.DataFrame()
    for df, source in zip(dfs, sources):
        df_copy = df[shared_columns].copy()
        df_copy['Source'] = source
        df_copy['frac. cluster occ.'] = df.groupby('cluster')['cluster'].transform('count') / len(df_copy)

        combined_df = pd.concat([combined_df, df_copy])

    shared_columns += ['frac. cluster occ.']

    # Apply Min-Max Scaling per feature to handle global min and max properly
    scaler = MinMaxScaler()
    scaled_data = pd.DataFrame(scaler.fit_transform(combined_df[shared_columns]), columns=shared_columns)
    scaled_data['Source'] = combined_df['Source'].values
    scaled_zero = scaler.transform([[0]*len(shared_columns)])[0]
    print(scaled_zero)

    # Melt the scaled dataframe to long format for plotting
    melted_df = scaled_data.melt(id_vars='Source', value_vars=shared_columns)

    # Plot a distribution plot for each shared column
    fig = plt.figure(figsize=(12, 6))
    if kind == 'strip':
        ax = sns.stripplot(x='variable', y='value', hue='Source', data=melted_df, dodge=True, alpha=0.05, jitter=0.3, palette=sns.color_palette('tab10'))
    elif kind == 'boxen':
        ax = sns.boxenplot(x='variable', y='value', hue='Source', data=melted_df)
    elif kind == 'violinplot':
        ax = sns.violinplot(x='variable', y='value', hue='Source', data=melted_df, split=True, bw=0.1, inner='quart')

    # Enhance annotations and place legend outside
    #plt.title('Feature Distributions Comparison')
    plt.xlabel('Feature')
    plt.ylabel('Normalized Value')
    plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels for readability

    # Update the alpha for the legend markers
    legend = ax.legend(title='Source', loc='upper left', bbox_to_anchor=(1.05, 1), borderaxespad=0.)
    for lh in legend.legendHandles: 
        lh.set_alpha(1)
        lh.set_linewidth(2) 

    # Annotate min and max values for each feature correctly within the scaled context and draw zero lines
    for i, col in enumerate(shared_columns):
        # Get global min and max after scaling
        feature_values = scaled_data[col]
        scaled_min = feature_values.min()
        scaled_max = feature_values.max()

        # Add text annotations
        ax.text(i, scaled_min, f'{combined_df[col].min():.2f}', color='blue', ha='center', va='top', fontsize=12)
        ax.axhline(y=scaled_min, xmin=(i)/len(shared_columns), xmax=(i + 1)/len(shared_columns), color='blue', linestyle='--', linewidth=1)
        ax.text(i, scaled_max, f'{combined_df[col].max():.2f}', color='red', ha='center', va='bottom', fontsize=12)
        ax.axhline(y=scaled_max, xmin=(i)/len(shared_columns), xmax=(i + 1)/len(shared_columns), color='red', linestyle='--', linewidth=1)

        if combined_df[col].min() > 1 or combined_df[col].min() < 0 :
            # Draw zero line
            ax.axhline(y=scaled_zero[i], xmin=(i)/len(shared_columns), xmax=(i + 1)/len(shared_columns), color='red', linestyle='--', linewidth=2)

        if col == 'ddG':
            ax.axhline(y=scaler.transform([[1]*len(shared_columns)])[0][i], xmin=(i)/len(shared_columns), xmax=(i + 1)/len(shared_columns), color='green', linestyle='--', linewidth=1)
            ax.axhline(y=scaler.transform([[-3]*len(shared_columns)])[0][i], xmin=(i)/len(shared_columns), xmax=(i + 1)/len(shared_columns), color='green', linestyle='--', linewidth=1)
    
    remapped_x = [remap_names[tick.get_text()] if tick.get_text() in remap_names.keys() else tick.get_text() for tick in ax.get_xticklabels()]
    ax.set_xticklabels(remapped_x)
    plt.show()

    return fig


def plot_joint_histogram(df1, df2, column, name1='K2369', name2='Q3421'):
    """
    Plots a joint histogram for the specified column from two DataFrames, 
    with bars from each DataFrame placed adjacently.

    Parameters:
        df1 (pd.DataFrame): The first DataFrame.
        df2 (pd.DataFrame): The second DataFrame.
        column (str): The column name to plot the histogram for.

    Returns:
        None: Displays the plot.
    """
    # Count the occurrences of each category in the specified column for both dataframes
    counts_df1 = df1[column].value_counts().reset_index()
    counts_df2 = df2[column].value_counts().reset_index()

    # Rename columns to a common format for easier handling in seaborn
    counts_df1.columns = ['Amino Acid', 'count']
    counts_df2.columns = ['Amino Acid', 'count']
    
    # Add a column to each to distinguish between the datasets
    counts_df1['dataset'] = name1
    counts_df2['dataset'] = name2

    # Concatenate the two DataFrames vertically
    combined_counts = pd.concat([counts_df1, counts_df2], ignore_index=True)
    
    # Plotting using seaborn
    fig = plt.figure(figsize=(10, 6))
    sns.barplot(data=combined_counts, x='count', y='Amino Acid', hue='dataset', ci=None)

    plt.xlabel('Count')
    plt.ylabel('Amino Acid')
    plt.title('Comparison of Amino Acid Frequencies')
    plt.legend(title='Dataset')

    plt.show()
    return fig


def bootstrap_table(df, 
                    plot_cols, plot_models, plot_title, ylim, right_y_lim,
                    table_cols, var_cols, sort_col, sort_order='descending',
                    saveloc_full='../data/extended/figure_data/data_bootstrapped_formatted.csv',
                    saveloc_formatted='../data/extended/data_bootstrapped_formatted.csv'):

    new_remap_cols = {}
    for key, value in remap_cols.items():
        new_remap_cols[key + '_mean'] = value + ' mean'
        new_remap_cols[key + '_std'] = value + ' stdev'

    df = df.sort_values(f'{sort_col}_mean', ascending=False if sort_order=='descending' else True).dropna(how='all', axis=1).reset_index()
    tab_cols = [c+'_mean' for c in table_cols] + [c+'_std' for c in table_cols]
    s4 = df[['model', 'model_type']+tab_cols]
    s4['model'] = s4['model'].str[:-4].replace(remap_names)

    stds = s4.drop(0).loc[:, [c for c in s4.columns if 'std' in c]].mean()
    stds = pd.DataFrame(stds).T
    stds.index = ['Avg. Dev.']
    stds.index.name = 'model'
    stds = stds.reset_index()
    stds.columns = [c.replace('_std', '_mean') for c in stds.columns]
    s4 = pd.concat([s4, stds])

    s5 = s4.set_index(['model_type', 'model'])
    s5.columns = [new_remap_cols[c] for c in s5.columns]
    var_cols = [remap_cols[c]+ ' mean' for c in var_cols]

    plt_cols = [c+'_mean' for c in plot_cols] + [c+'_std' for c in plot_cols]
    tmp2 = df[['model']+plt_cols].set_index('model')

    tmp2.columns = [new_remap_cols[c] for c in tmp2.columns]
    tmp2 = tmp2.reset_index().drop_duplicates()

    fig = make_scatter_chart(tmp2, models=plot_models, title=plot_title, ylim=ylim, figsize=(7, 4), 
        use_dual_y_axis=True, right_y_lim=right_y_lim, sw=len(plot_cols)/3, scale_y2=(plot_cols[-1] == 'net_stabilization'))

    s5_full = s5.copy(deep=True)
    s5_full.to_csv(saveloc_full, encoding='utf-8-sig')
    #print(s5_full)

    # Iterate over the DataFrame and update the mean columns
    for column in s5.columns:
        if 'mean' in column:
            stdev_column = column.replace('mean', 'stdev')
            #s5_full[column] = s5_full[column].astype(str) + ' ± ' + s5_full[stdev_column].astype(str)
            if column in var_cols and stdev_column in s5.columns:
                stdev_column = column.replace('mean', 'stdev')
                s5[column] = s5[column].apply(format_fixed_total_digits).astype(str) + ' ± ' + s5[stdev_column].apply(format_fixed_total_digits).astype(str)
            else:
                s5[column] = s5[column].apply(format_fixed_total_digits).astype(str)

    s5 = s5[[col for col in s5.columns if 'stdev' not in col]]
    s5.columns = [col[:-5] for col in s5.columns]
    s5 = s5.replace('± nan', '', regex=True)
    s5_full = s5_full.replace('± nan', '', regex=True)

    #s5_full = s5_full[[col for col in s5_full.columns if 'stdev' not in col]]
    #s5_full.columns = [col[:-5] for col in s5_full.columns]

    s6 = s5.reset_index().rename({'model_type': 'Model Type', 'model': 'Model'}, axis=1).set_index(['Model Type', 'Model'])
    s6.to_csv(saveloc_formatted, encoding='utf-8-sig')
    return s6, fig