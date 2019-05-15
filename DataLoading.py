import pandas as pd
import numpy as np
import os
import torch.utils.data


class SpectrogramDataset(torch.utils.data.Dataset):
    def __init__(self, path, filename):
        self.path = path
        spect_frame = pd.read_csv(os.path.join(path, filename))
        self.files = spect_frame.iloc[:, 0]
        self.classes = spect_frame.iloc[:, 1]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        spect_name = os.path.join(self.path, self.files.iloc[idx, 0])
        spectrogram = np.load(spect_name)
        cls = self.classes.iloc[idx, 0]
        sample = {'spectrogram': spectrogram, 'cls': cls}

        return sample
