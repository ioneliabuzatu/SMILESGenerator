import torch
from torch.utils.data import DataLoader
from utils.data import load_train_and_val_batches_data

import config
from utils.dataset import SmilesDataset
from model import GenerativeMoleculesModel
from tqdm import tqdm
from utils.data import load_train_and_val_batches_data

writer = config.tensorboard

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running on {device}")


def train(device, save=False):
    torch.manual_seed(1112)
    torch.cuda.manual_seed_all(1112)

    train_batches, val_batches = load_train_and_val_batches_data()
    train_dataset_already_batched = SmilesDataset(train_batches)
    len_train_dataset_smiles = len(train_dataset_already_batched)
    print(f"Train dataset length: {len_train_dataset_smiles}")
    model = GenerativeMoleculesModel(
        40, config.hidden_size, 40, config.embedding_dimension, config.n_layers,
        bidirectional=config.bidirectional
    )
    print(model)

    if not config.bidirectional:
        model.load_state_dict(torch.load(config.pretrained_filepath))

    model = model.to(device)
    model.train()

    if config.optimizer == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    else:
        optimizer = torch.optim.SGD(model.parameters(), momentum=config.momentum, lr=config.lr)
    scalars_loss_step = 0
    for epoch in tqdm(range(config.epochs)):
        epoch_loss = 0
        for i_batch, sample_batched in enumerate(train_dataset_already_batched):
            input_smile, target = sample_batched[0].squeeze(), sample_batched[1].squeeze()
            batch_size = input_smile.size(0)
            sequence_length = input_smile.size(1)
            hidden = model.init_hidden(batch_size)
            hidden = (hidden[0].to(device), hidden[1].to(device))

            model.zero_grad()
            optimizer.zero_grad()

            loss = 0
            for c in range(sequence_length):
                output, hidden = model(input_smile[:, c].to(device), hidden)
                loss += config.criterion(output, target[:, c].to(device))

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() / sequence_length

            writer.add_scalar('train_loss', loss.item() / sequence_length, scalars_loss_step)
            scalars_loss_step += 1

            # if i_batch == 3:
            #     break
        print(f"Done epoch # {epoch} - loss {epoch_loss}")
        if save:
            if config.bidirectional:
                torch.save(model.state_dict(), f"{config.checkpoint_filepath}_bidirectional")
            else:
                torch.save(model.state_dict(), config.checkpoint_filepath)


if __name__ == "__main__":
    train(device, save=True)
