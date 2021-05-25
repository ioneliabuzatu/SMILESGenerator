# from fcd import get_fcd, load_ref_model, canonical_smiles, get_predictions, calculate_frechet_distance
import numpy as np
import pandas as pd
import torch
from torch.autograd import Variable
from tqdm import tqdm


def added_to_dictionary(smile, dictionary):
    for char in smile:
        if char not in dictionary:
            dictionary[char] = True


def to_numpy_compress(save=False):
    smiles = pd.read_csv("evaluation/data/smiles_train.txt", header=None)[0]

    data_set = []
    dictionary = {}

    for smile in smiles:
        smile = smile.strip()
        smile = '!' + smile + ' '
        data_set.append(smile)
        added_to_dictionary(smile, dictionary)

    vocabs = [ele for ele in dictionary]
    print(len(vocabs))

    if save:
        np.savez_compressed(
            'data/smiles/smiles_data.npz',
            data_set=np.array(data_set, dtype=object),
            vocabs=np.array(vocabs), dtype=object
        )


def tensor_from_chars_list(chars_list, vocabs):
    tensor = torch.zeros(len(chars_list)).long()
    for c in range(len(chars_list)):
        tensor[c] = vocabs.index(chars_list[c])
    return tensor.view(1, -1)


def process_batch(sequences, batch_size, vocabs, cuda):
    batches = []
    for i in range(0, len(sequences), batch_size):
        input_list = []
        output_list = []
        for j in range(i, i + batch_size, 1):
            if j < len(sequences):
                input_list.append(tensor_from_chars_list(sequences[j][:-1], vocabs, cuda))
                output_list.append(tensor_from_chars_list(sequences[j][1:], vocabs, cuda))
        inp = Variable(torch.cat(input_list, 0))
        target = Variable(torch.cat(output_list, 0))
        # if cuda:
        #     inp = inp.cuda()
        #     target = target.cuda()
        batches.append((inp, target))
    train_split = int(0.9 * len(batches))
    return batches[:train_split], batches[train_split:]


def crrate_train_and_val_data():
    from datetime import datetime

    start_time = datetime.now()
    batch_size = 128
    cuda = True
    f = np.load('Generate-novel-molecules-with-LSTM/generative_model/data/smiles/smiles_data.npz', allow_pickle=True)
    data, vocabs = f['data_set'], f['vocabs']
    vocabs = list(vocabs)
    print()
    hash_length_data = {}
    train_batches = []
    val_batches = []
    batches = []

    for ele in tqdm(data):
        l = len(ele)
        if l not in hash_length_data:
            hash_length_data[l] = []
        hash_length_data[l].append(ele)

    idx = 0
    for length in tqdm(hash_length_data):
        train, val = process_batch(hash_length_data[length], batch_size, vocabs, cuda)
        train_batches.extend(train)
        val_batches.extend(val)
        idx += 1
        if idx == 2:
            break
    print('Duration: {}'.format(datetime.now() - start_time))


def max_length_smile():
    max_len_smiles = 0
    with open("resources/sample_submission.txt", "r") as file:
        content_file = file.readlines()
        for smile in content_file:
            smile = smile.strip()
            len_smile = len(smile)
            max_len_smiles = max(max_len_smiles, len_smile)

    print("Longest smile found has length", max_len_smiles)


def load_train_and_val_batches_data():
    f = torch.load("Generate-novel-molecules-with-LSTM/generative_model/data/smiles/train_val_batches.npz")
    return f['train'], f['val']
