import torch
import experiment_buddy

import torch.nn as nn

hidden_size = 100  # 1024
embedding_dimension = 248 # 248
n_layers = 1 # 3
lr = 0.0005

batch_size = 128
vocabs_size = 40
output_size = 40
print_every = 500
plot_every = 10
save_every = 1000
epochs = 200
seed = 1112

criterion = nn.CrossEntropyLoss()
optimizer = "adam"
momentum = 0.96

experiment_buddy.register(locals())
tensorboard = experiment_buddy.deploy(
    "mila",
    sweep_yaml="",
    proc_num=1,
    wandb_kwargs={"entity": "ionelia"}
)
