import torch
import experiment_buddy

import torch.nn as nn

host = "mila"
hidden_size = 100  # 1024
embedding_dimension = 248 # 248
n_layers = 2 # 3
lr = 0.0005
optimizer = "adam"
momentum = 0.96
bidirectional = False

batch_size = 128
epochs = 200

criterion = nn.CrossEntropyLoss()

if host == "":
    checkpoint_filepath = "checkpoint_smiles.pt"
else:
    checkpoint_filepath = "/home/mila/g/golemofl/data/smiles-project/checkpoint_smiles.pt"

experiment_buddy.register(locals())
tensorboard = experiment_buddy.deploy(
    host,
    sweep_yaml="",
    proc_num=1,
    wandb_kwargs={"entity": "ionelia"}
)
