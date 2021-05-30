# from fcd import get_fcd, load_ref_model, canonical_smiles, get_predictions, calculate_frechet_distance
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
import config


def load_train_and_val_batches_data():
    if not config.host:
        f = torch.load("Generate-novel-molecules-with-LSTM/generative_model/data/smiles/train_val_batches.npz")
    else:
        f = torch.load("/home/mila/g/golemofl/data/smiles-project/train_val_batches.npz")
    return f['train'], f['val']


def max_length_smile():
    max_len_smiles = 0
    with open("resources/sample_submission.txt", "r") as file:
        content_file = file.readlines()
        for smile in content_file:
            smile = smile.strip()
            len_smile = len(smile)
            max_len_smiles = max(max_len_smiles, len_smile)

    print("Longest smile found has length", max_len_smiles)


