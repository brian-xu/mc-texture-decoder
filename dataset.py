import os

import numpy as np
import torch
from torch.utils.data import Dataset


class TextureDataset(Dataset):
    def __init__(self, data_dir="processed"):
        self.images = np.load(os.path.join(data_dir, "images.npy"))
        self.labels = np.load(os.path.join(data_dir, "embeddings.npy"))

    def __getitem__(self, index):
        data = torch.tensor(self.images[index], dtype=torch.float)
        label = torch.tensor(self.labels[index], dtype=torch.float)
        return data, label

    def __len__(self):
        return self.images.shape[0]
