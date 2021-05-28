import numpy as np
import torch
from torch.autograd import Variable
from tqdm import tqdm

import config
from model import GenerativeMoleculesModel
from utils.data import tensor_from_chars_list

SMILES_DATA_FILEPATH = "/home/mila/g/golemofl/data/smiles-project/smiles_data.npz"
# CHECKPOINT_FILEPATH = "/home/mila/g/golemofl/data/smiles-project/checkpoint_generator_model.pt_bidirectional"
# CHECKPOINT_FILEPATH = "/home/mila/g/golemofl/data/smiles-project/smiles_generator_model.pt"
CHECKPOINT_FILEPATH = "/home/mila/g/golemofl/data/smiles-project/checkpoint_generator_model.pt"
SUBMISSION_SMILES_FILEPATH = "/home/mila/g/golemofl/data/smiles-project/my_smiles.txt"
# SMILES_DATA_FILEPATH = "Generate-novel-molecules-with-LSTM/generative_model/data/smiles/smiles_data.npz"
# CHECKPOINT_FILEPATH = "Generate-novel-molecules-with-LSTM/generative_model/smiles_generator_model.pt"
# SUBMISSION_SMILES_FILEPATH = "utils/resources/my_smiles.txt"


def evaluate(model, vocabs, device, prime_str='!', end_token=" ", temperature=0.8):
    max_length = 110
    inp = Variable(tensor_from_chars_list(prime_str, vocabs)).to(device)
    batch_size = inp.size(0)
    hidden = model.init_hidden(batch_size)

    inp = inp.to(device)
    hidden = (hidden[0].to(device), hidden[1].to(device))
    predicted = prime_str
    while True:
        output, hidden = model(inp, hidden)
        # Sample from the network as a multinomial distribution
        output_dist = output.data.view(-1).div(temperature).exp()
        top_i = torch.multinomial(output_dist, 1)[0]
        # Add predicted character to string and use as next input
        predicted_char = vocabs[top_i]

        if predicted_char == end_token or len(predicted) > max_length:
            return predicted

        predicted += predicted_char
        inp = Variable(tensor_from_chars_list(predicted_char, vocabs)).to(device)


def generate_smiles(generated_smiles=300):
    vocabs_size = 40
    output_size = 40
    hidden_size = 1024
    embedding_dimension = 248
    n_layers = 3

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on device: {device}")
    f = np.load(SMILES_DATA_FILEPATH, allow_pickle=True)
    vocabs = list(f['vocabs'])
    model = GenerativeMoleculesModel(vocabs_size, hidden_size, output_size, embedding_dimension, n_layers, config.bidirectional)
    model = model.to(device)
    model.load_state_dict(torch.load(CHECKPOINT_FILEPATH))
    with open(SUBMISSION_SMILES_FILEPATH, "a") as file:
        for _ in tqdm(range(generated_smiles)):
            new_smile = evaluate(model, vocabs, device, temperature=0.85)
            if new_smile[0] == "!":
                new_smile = new_smile[1:]
            if set(new_smile) == "C":
                continue
            file.write(f"{new_smile}\n")
    print(f"File created for submission with {generated_smiles}k smiles in my_smiles.txt")


generate_smiles(generated_smiles=10000)
