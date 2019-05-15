import pandas as pd
import numpy as np
import os
import torch.utils.data
from torch.utils.data.sampler import SubsetRandomSampler


class SpectrogramDataset(torch.utils.data.Dataset):
    def __init__(self, path, filename):
        self.path = path
        spect_frame = pd.read_csv(os.path.join(path, filename))
        self.files = spect_frame.iloc[:, 0]
        self.classes = spect_frame.iloc[:, 1]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        spect_name = os.path.join(self.path, self.files[idx])
        spectrogram = np.load(spect_name)
        cls = self.classes.iloc[idx]
        sample = {'spectrogram': spectrogram, 'cls': cls}

        return sample


def load_data(batch_size, num_workers, validation_split, shuffle, seed, path, filename):
    spect_dataset = SpectrogramDataset(path, filename)

    # Creating data indices for training and validation splits:
    dataset_size = len(spect_dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(validation_split * dataset_size))
    if shuffle:
        np.random.seed(seed)
        np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]

    # Creating PT data samplers and loaders:
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)

    train_loader = torch.utils.data.DataLoader(spect_dataset, batch_size=batch_size,
                                               sampler=train_sampler, num_workers=num_workers)
    validation_loader = torch.utils.data.DataLoader(spect_dataset, batch_size=batch_size,
                                                    sampler=valid_sampler, num_workers=num_workers)
    return train_loader, validation_loader
