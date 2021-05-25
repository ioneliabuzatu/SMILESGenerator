import torch
import experiment_buddy

import torch.nn as nn

vocabs_size = 40
output_size = 40
batch_size = 128
cuda = True
hidden_size = 1024
embedding_dimension = 248
n_layers = 3
lr = 0.0005
n_batches = 200000
print_every = 500
plot_every = 10
save_every = 1000
end_token = ' '

epochs = 2

criterion = nn.CrossEntropyLoss()

experiment_buddy.register(locals())
tensorboard = experiment_buddy.deploy(
    "mila",
    sweep_yaml="",
    proc_num=1,
    wandb_kwargs={"entity": "ionelia"}
)
