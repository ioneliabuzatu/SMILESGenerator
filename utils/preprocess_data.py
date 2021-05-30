import tqdm
from torch.autograd import Variable
import torch
import pandas as pd
import numpy as np


def added_to_dictionary(smile, dictionary):
    for char in smile:
        if char not in dictionary:
            dictionary[char] = True


def to_numpy_compress(save=True):
    smiles = pd.read_csv("utils/resources/data/data/smiles_train.txt", header=None)[0]

    data_set = []
    dictionary = {}

    for smile in smiles:
        smile = smile.strip()
        smile = '!' + smile + ' '
        data_set.append(smile)
        added_to_dictionary(smile, dictionary)

    vocabs = [ele for ele in dictionary]

    data_set = np.array(data_set, dtype=object)
    vocabs = np.array(vocabs, dtype=object)

    if save:
        np.savez_compressed('utils/resources/data/smiles_data.npz', data_set=data_set, vocabs=vocabs)
    return data_set, vocabs


def tensor_from_chars_list(chars_list, vocabs):
    tensor = torch.zeros(len(chars_list)).long()
    for c in range(len(chars_list)):
        tensor[c] = vocabs.index(chars_list[c])
    return tensor.view(1, -1)


def process_batch(sequences, batch_size, vocabs):
    batches = []
    for i in range(0, len(sequences), batch_size):
        input_list = []
        output_list = []
        for j in range(i, i + batch_size, 1):
            if j < len(sequences):
                input_list.append(tensor_from_chars_list(sequences[j][:-1], vocabs))
                output_list.append(tensor_from_chars_list(sequences[j][1:], vocabs))
        inp = Variable(torch.cat(input_list, 0))
        target = Variable(torch.cat(output_list, 0))
        batches.append((inp, target))
    train_split = int(0.9 * len(batches))
    return batches[:train_split], batches[train_split:]


def save_preprocessed_data_as_batches(data, vocabs, batch_size=128):
    hash_length_data = {}
    for ele in data:
        l = len(ele)
        if l >= 3:
            if l not in hash_length_data:
                hash_length_data[l] = []
            hash_length_data[l].append(ele)
    train_batches = []
    val_batches = []
    for length in tqdm(hash_length_data):
        train, val = process_batch(hash_length_data[length], batch_size, vocabs)
        train_batches.extend(train)
        val_batches.extend(val)
    torch.save({"train": train_batches, "val": val_batches}, "utils/resources/data/train_val_batches.npz")
    print(f"saved train and val batches to file utils/resources/data/train_val_batches.npz")


if __name__ == "__main__":
    data, vocabs = to_numpy_compress(save=True)
    save_preprocessed_data_as_batches(data, vocabs)
