import pandas as pd
import numpy as np
import os
import torch.utils.data
from torch.utils.data.sampler import SubsetRandomSampler
import torch.nn.functional


class SpectrogramDataset(torch.utils.data.Dataset):
    def __init__(self, path, filename, class_num):
        self.path = path
        self.class_num = class_num
        spect_frame = pd.read_csv(os.path.join(path, filename))
        self.files = spect_frame.iloc[:, 0]
        self.classes = spect_frame.iloc[:, 1]

        classes_unique = pd.Series(self.classes.unique())
        self.cls_dict = classes_unique.to_dict()
        inv_map = {v: k for k, v in self.cls_dict.items()}
        self.classes = self.classes.map(inv_map)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        spect_name = os.path.join(self.path, self.files[idx])
        spectrogram = np.load(spect_name)
        cls = self.classes.iloc[idx]
        # one_hot = torch.nn.functional.one_hot(cls, self.class_num)
        one_hot = np.eye(self.class_num)[cls].astype(int)

        sample = (spectrogram, cls)

        return sample

    def get_cls_dict(self):
        return self.cls_dict

def load_data(batch_size, num_workers, validation_split, shuffle, seed, classes_num, path, filename):
    spect_dataset = SpectrogramDataset(path, filename, classes_num)
    cls_dict = spect_dataset.get_cls_dict()

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
    return train_loader, validation_loader, cls_dict
