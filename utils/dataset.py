from torch.utils.data import Dataset

from utils import load_train_and_val_batches_data


class SmilesDataset(Dataset):
    """smiles dataset."""

    def __init__(self):
        """ """

        self.train_batches, self.val_batches = load_train_and_val_batches_data()

    def __len__(self):
        return len(self.train_batches)

    def __getitem__(self, idx):
        sample = self.train_batches[idx]
        return sample
