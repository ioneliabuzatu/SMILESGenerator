import torch
from torch.utils.data import DataLoader

import config
from utils.dataset import SmilesDataset
from model import generative_model
from tqdm import tqdm
from utils.data import load_train_and_val_batches_data

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running on {device}")


@torch.enable_grad()
def train(device):
    train_dataset_already_batched = SmilesDataset()
    # dataloader = DataLoader(train_dataset_already_batched, batch_size=1, shuffle=True, num_workers=4)
    # print(f"DataLoader length: {len(dataloader)}")
    model = generative_model(
        config.vocabs_size, config.hidden_size, config.output_size, config.embedding_dimension, config.n_layers
    )
    model.load_state_dict(torch.load("Generate-novel-molecules-with-LSTM/generative_model/smiles_generator_model.pt"))
    model = model.to(device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

    for epoch in tqdm(range(config.epochs)):
        for i_batch, sample_batched in enumerate(train_dataset_already_batched):
            input_smile, target = sample_batched[0].squeeze(), sample_batched[1].squeeze()
            batch_size = input_smile.size(0)
            sequence_length = input_smile.size(1)
            hidden = model.init_hidden(batch_size)

            input_smile = input_smile.to(device)
            target = target.to(device)
            hidden = (hidden[0].to(device), hidden[1].to(device))

            optimizer.zero_grad()

            loss = 0
            for c in range(sequence_length):
                output, hidden = model(input_smile[:, c], hidden)
                loss += config.criterion(output, target[:, c])

            loss.backward()
            optimizer.step()

            # return loss.item() / sequence_length


train(device)
