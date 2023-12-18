# Protein Sequence Likelihood Modelling for Thermostability Prediction

This repository is for facilitating access to and benchmarking self-supervised deep learning models, which predict the likelihood of amino acids in their biochemical context, in order to make zero-shot predictions of thermostability measurements.

# System Requirements

* Linux-based Operating System (any modern Linux distribution should work)
  * tested on GNU/Linux 3.10.0-1160.71.1.el7.x86_64, 4.18.0-477.21.1.el8_8.x86_64
  * tested on WSL2
  * tested on Fedora 38 (cannot use apt here of course)
* Anaconda / Python 3.8
  * dependencies included in requirements.txt (additionally requirements_inference.txt for running inference)
* NVIDIA GPU (if running inference)
  * tested on A100, RTX 3090
* NVIDIA CUDA (tested v11) and CUDNN (if running inference)
* HMMER (if generating MSAs)

# Demo /  Instructions

You can use the Jupyter notebooks from the analysis_notebooks folder to reproduce the figures, **modifying the path at the start of the file** and running each cell sequentially. 

The notebooks draw from precomputed features, the original databases from their respective authors, and predictions generated on a high-performance compute cluster (also tested on a flagship consumer GPU). All of these data sources are included in the repository, and instructions for reproducing the predictions and features are provided below. 

The expected outputs are shown below the cells. The expected runtimes are often included per-cell, for a total runtime of <60 minutes on a typical PC (or <5 minutes for the S461 analysis). To reduce the demo time, the number of bootstrapped replicates have been greatly reduced from those reported in the text. Additionally, the ensemble predictions step has been precomputed and is loaded from a file, but can be easily reproduced by uncommenting the relevant line(s).

We recommend demoing the more thoroughly documented and tidy fireprot_analysis.ipynb .

# Installation Guide

Note: only general setup is required to demo code.

The expected installation time for basic functionality is >10 minutes, assuming you only install the first requirements.txt. Approximatly 30 minutes would be required to install CUDA / CUDNN / Pytorch and inference_requirements but depends on internet connection. Similarly, installation time for tested models for inference depends on the internet connection speed, as some models are many GiB in size.

The sections after general setup are for reproducing the experiments starting from raw data.

## General Setup

We provide the processed predictions for FireProtDB and S461 in `./data/fireprot_mapped_preds_original.csv` and `./data/s461_mapped_preds_original.csv`, respectively. However, to reproduce the predictions you can follow the below sections for preprocessing and inference. We also provide the pre-extracted features for analysis in the corresponding `./data/{dataset}_mapped_feats.csv` files, but you can reproduce those according to the feature analysis section.

Clone the repository:

`git clone https://github.com/skalyaanamoorthy/ThermoTL.git`

`cd ThermoTL`

If you do not have root permissions (ability to sudo) you should use conda, as it will make future steps faster and easier. Otherwise you can use VirtualEnv.

To install with conda, you might need to `module load anaconda` and/or `module load python` first:

`conda create --name pslm python=3.8`

`conda activate pslm`

To be able to run the notebooks:

`conda install -c conda-forge notebook`

**If you don't have conda and/or don't have root permissions:**

Make a new virual environment instead (tested with Python=3.8). On a cluster, you might need to `module load python` first:

`virtualenv pslm`

`source pslm/bin/activate`

On the ComputeCanada cluster, you will have to comment out pyarrow and cmake dependencies and load the arrow module instead with `module load arrow`. You will also have to use the --no-deps flag.

**After setting up either a virtualenv OR conda environment**:

You can install the pip requirements:

`pip install -r requirements.txt`

Finally, install evcouplings with no dependencies (it is an old package which will create conflicts):

`pip install evcouplings --no-deps`

## Inference Setup

**Note: you can skip this step to demo results. This is for reproducing predictions.**

If you have a sufficient NVIDIA GPU (tested on 3090 and A100) you can make predictions with the deep learning models.

Start by installing CUDA if you have not already: https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html. At time of writing you will need to get CUDA 11.X in order to be able to install the torch-* requirements. If you are on a cluster, make sure you have the cuda module loaded e.g. `module load cuda` as well as any compiler necessary e.g. `module load gcc`. If you are using WSL2, you should be able to just use `sh ./convenience_scripts/cuda_setup_wsl.sh`. MIF-ST also requires cuDNN: https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html.

Then install Pytorch according to the instructions: https://pytorch.org/get-started/locally/ . In most cases, it will suffice to `pip install torch`. On the ComputeCanada cluster, there may be dependency issues between the numpy and torch versions. In this case, since Modeller cannot be installed anyway, we suggest that preprocessing be performed locally, followed by only installing the requirements_inference on the cluster environment.

Finally, you can install the inference-specific requirements (remember the --no-deps flag on the cluster):

`pip install -r requirements_inference.txt`

You will also need to install the following inference repositories if you wish to use these specific models:

ProteinMPNN:

`git clone https://github.com/dauparas/ProteinMPNN`
	
Note: ProteinMPNN directory will be used as input for ProteinMPNN scripts; it will need to be specified when calling the Python script (--mpnn_loc).

Tranception:

`git clone https://github.com/OATML-Markslab/Tranception`

Follow the instructions in the repo to get the Tranception_Large (parameters) binary and config. You do not need to the setup the conda environment.
Again, you will need to specify the location of the repository (--tranception_loc) and the model weights (--checkpoint).

KORPM:

Make sure to have Git LFS in order to obtain the potential maps used by KORPM, otherwise you can download the repository as a .zip and extract it.

`git clone https://github.com/chaconlab/korpm`

You will need to compile korpm with the gcc compiler:

`cd korpm/sbg`

`sh ./compile_korpm.sh`

Like the above methods, there is a wrapper script in inference_scripts where you will need to specify the installation directory with the argument --korpm_loc.

## Preprocessing

**Note: you can skip this step to demo results. This is for reproducing predictions.**

In order to perform inference you will first need to preprocess the structures and sequences. Follow the above instructions before proceeding.

You will need the following additional tools for preprocessing:

Modeller (for repairing PDB structures): https://salilab.org/modeller/download_installation.html

You will need a license, which is free for academic use; follow the download page instructions to make sure it is specified. 

Assuming you are using conda:

`conda config --add channels salilab`

`conda install modeller`

**If you install using conda, you can skip the following workaround and resume at the python call.**

To make modeller visible to the Python scripts from within the VirtualEnv, you can append to your `./pslm/bin/activate` file following the pattern shown in `convenience_scripts/append_modeller_paths.sh`:

`sh convenience_scripts/append_modeller_paths.sh`

**Ensure to replace the modeller version and system architecture as required (you can find these with `uname` and `uname -m` respectively). Then make sure to restart the virtualenv**:

`source pslm/bin/activate`

To run inference on either FireProtDB or S669/S461 you will need to preprocess the mutants in each database, obtaining their structures and sequences and modelling missing residues. You can accomplish this with preprocess.py. Assuming you are in the base level of the repo, you can call the following (this will use the raw FireProtDB obtained from https://loschmidt.chemi.muni.cz/fireprotdb/ Browse Database tab).:

`python preprocessing/preprocess.py --dataset fireprotdb`

Add the --internal_path argument to specify a different location for the repository where the calculations will be run, for instance if preprocessing locally and then running inference on the cluster

Note that the output dataframe `./data/fireprot_mapped.csv` is already generated, but the other files are not prepared.

It is expected to see the message '507 observations lost from the original dataset' for FireProtDB. Note that you will also have to do this for S669. S461 is a subset of S669, so you can call either dataset for the `--dataset` argument, and the same preprocessing will occur; the subset will be generated in the analysis notebook. For inverse/reversion mutations on S669/S461, you will use the predicted mutant structures in the structures_mut folder, which we obtained from the authors (thank you, Drs. Birolo and Fariselli): https://academic.oup.com/bib/article/23/2/bbab555/6502552. They will have to be preprocessed as well; add the --inverse flag if you need to get information to run Rosetta. We also obtained the original data file Data_s669_with_predictions.csv from the Supplementary information of this paper, adjusting one record to accurately reflect the structure. Citation: Pancotti, C. et al. Predicting protein stability changes upon single-point mutation: a thorough comparison of the available tools on a new dataset. Briefings in Bioinformatics 23, bbab555 (2022).

`python preprocessing/preprocess.py --dataset s669`

You can also use a custom database for inference. The preprocessing script will facilitate making predictions (and MSAs) with all methods by collecting the corresponding UniProt sequence (if available) as well as modelling, preprocessing, and validating all structures as required. To use this functionality, you can create a csv file with columns for code (PDB ID) chain (chain in PDB structure), wild_type (one letter code for wild-type identity at mutated position), position (corresponds to the **index in the PDB sequence (not PDB coordinates)**, and mutation (one-letter code), with as many rows as desired. Then run preprocessing pointing to the database and giving it a desired name which will appear in the prefix:

`python preprocessing/preprocess.py --dataset MY_CUSTOM_NAME --db_loc ./data/my_custom_dataset.csv`

If you are not sure of how your numbering aligns with the PDB sequence numbering, you can try to use the --infer_pos flag.

## Running Inference

**Note: you MUST run the preprocessing scripts to generate the correct file mappings for your system. If you run into problems with missing files when running inference, this is probably why.**

Then, you can run any of the inference scripts in inference scripts. You can use the template calls from cluster_inference_scripts in order to determine the template for calling each method's wrapper script (they are designed to be called from the cluster_inference_scripts directory, though). On the other hand, to run ProteinMPNN **from the repository root** with 0.2 Angstrom backbone noise on FireProtDB:

`cp data/fireprot_mapped.csv data/fireprot_mapped_preds.csv`

`python inference_scripts/mpnn.py --db_location 'data/fireprot_mapped.csv' --output 'data/fireprot_mapped_preds.csv' --mpnn_loc ~/software/ProteinMPNN --noise '20'`

**Again, note that you must specify the install location for ProteinMPNN, Tranception, and KORPM because they originate from repositories.**

If you are running on a cluster, you will likely find it convenient to modify the `cluster_inference_scripts` and directly submit them; they are designed to be submitted from their own folder as the working directory, rather than the root of the repo like all other files. Note that ESM methods (ESM-1V, MSA-Transformer, ESM-IF) and MIF methods (MIF and MIF-ST) will require substantial storage space and network usage to download the model weights on their first run (especially ESM-1V). To run inference of inverse/reversion mutations for structural methods you will need the predicted mutants as stated above, and you will have to use the _inv versions of each structural method.

Note that ProteinMPNN and Tranception require the location where the github repository was installed as arguments.

## Feature Analysis Setup

For analysis based on features, you can compute the features using preprocessing/compute_features.py. Note that the features have been precomputed and appear in `./data/fireprot_mapped_feats.csv`:
You will need the following tools to help recompute features:

AliStat (for getting multiple sequence alignment statistics): https://github.com/thomaskf/AliStat

`git clone https://github.com/thomaskf/AliStat`

`cd AliStat`

`make`

DSSP (for extracting secondary structure and residue accessibility): https://github.com/cmbi/dssp

`sudo apt install dssp`

OR

`git clone https://github.com/cmbi/dssp` and follow instructions.

Finally, you can run the following to compute the features. 

`python3 preprocessing/compute_features.py --alistat_loc YOUR_ALISTAT_INSTALLATION`

It is expected that there will be some errors in computing features. However, if you see that DSSP did not produce an output, this is an issue with the DSSP version. Make sure you have version 4, or else install via github. AliStats might fail for large alignments if you do not have enough RAM. Remember that the features have been pre-computed for your convience as stated above, and any missing features can be handled by merging dataframes.
