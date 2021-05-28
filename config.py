import torch
import experiment_buddy

import torch.nn as nn

host = ""

hidden_size = 1024
embedding_dimension = 248
n_layers = 3
lr = 0.0005
optimizer = "adam"
momentum = 0.96
bidirectional = True

batch_size = 128
epochs = 300

criterion = nn.CrossEntropyLoss()

if host == "":
    pretrained_filepath = "Generate-novel-molecules-with-LSTM/generative_model/smiles_generator_model.pt"
    checkpoint_filepath = "checkpoint_generator_model.pt"
else:
    pretrained_filepath = "/home/mila/g/golemofl/data/smiles-project/smiles_generator_model.pt"
    checkpoint_filepath = "/home/mila/g/golemofl/data/smiles-project/checkpoint_generator_model.pt"

experiment_buddy.register(locals())
tensorboard = experiment_buddy.deploy(
    host,
    sweep_yaml="",
    proc_num=1,
    wandb_kwargs={"entity": "ionelia"}
)
