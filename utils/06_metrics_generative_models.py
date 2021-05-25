import pickle
import warnings

import numpy as np
import pandas as pd
import rdkit
from fcd import get_fcd, load_ref_model, canonical_smiles, get_predictions, calculate_frechet_distance
from rdkit import RDLogger

RDLogger.DisableLog('rdApp.*')
warnings.filterwarnings("ignore")
np.random.seed(1234)
print("RDKit: ", rdkit.__version__)

# Load ChemNet model
model = load_ref_model()

# Load generated molecules from an input file, which contains one generated SMILES per line
gen_mol_file = "resources/LSTM_Segler.smi"
generated_molecules = pd.read_csv(gen_mol_file, header=None)[0]

# Sample two subsets, treat one as "real" SMILES and the other as "generated" SMILES for now
# IMPORTANT: take at least 10000 molecules as FCD can vary with sample size 
smiles_real = np.random.choice(generated_molecules, 10000, replace=False)
smiles_gen = np.random.choice(generated_molecules, 10000, replace=False)

# get canonical smiles and filter invalid ones
smiles_real_can = [w for w in canonical_smiles(smiles_real) if w is not None]
smiles_gen_can = [w for w in canonical_smiles(smiles_gen) if w is not None]

# Get CHEMBLNET activations of generated molecules
act_real = get_predictions(model, smiles_real_can)
act_gen = get_predictions(model, smiles_gen_can)

# Calculate mean and covariance statistics from these activations for both sets
mu_real = np.mean(act_real, axis=0)
sigma_real = np.cov(act_real.T)

mu_gen = np.mean(act_gen, axis=0)
sigma_gen = np.cov(act_gen.T)

# Calculate the FCD
fcd_value = calculate_frechet_distance(
    mu1=mu_real,
    mu2=mu_gen,
    sigma1=sigma_real,
    sigma2=sigma_gen)

print('FCD: ', fcd_value)

fcd_value = get_fcd(smiles_real_can, smiles_gen_can, model=model)
print('FCD: ', fcd_value)

fcd_value = get_fcd(smiles_real_can, smiles_gen, model=model)
print('FCD no canonicalization: ', fcd_value)

# Load sample submission
gen_mol_file = "resources/sample_submission.txt"
smiles_gen = pd.read_csv(gen_mol_file, header=None)[0].iloc[:10000]

# get canonical smiles and filter invalid ones
smiles_gen_can = [w for w in canonical_smiles(smiles_gen) if w is not None]

# Calculate statistics for the sample submission
act_gen = get_predictions(model, smiles_gen_can)
mu_gen = np.mean(act_gen, axis=0)
sigma_gen = np.cov(act_gen.T)

# Load precomputed test mean and covariance
with open("resources/test_stats.p", 'rb') as f:
    mu_test, sigma_test = pickle.load(f)

fcd_value = calculate_frechet_distance(
    mu1=mu_gen,
    mu2=mu_test,
    sigma1=sigma_gen,
    sigma2=sigma_test)
print('FCD: ', fcd_value)

validity = len(smiles_gen_can) / len(smiles_gen)
print("Validity: ", validity)

smiles_unique = set(smiles_gen_can)
uniqueness = len(smiles_unique) / len(smiles_gen)
print("Uniqueness: ", uniqueness)

# load training set for novelty
with open("resources/smiles_train.txt") as f:
    smiles_train = {s for s in f.read().split() if s}

smiles_novel = smiles_unique - smiles_train
novelty = len(smiles_novel) / len(smiles_gen)
print("Novelty: ", novelty)
