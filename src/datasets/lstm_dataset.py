import os
import numpy as np
import torch
from torch.utils.data import Dataset

class LSTMDataset(Dataset):
    def __init__(self, data_dir, split='train'):
        """
        Args:
            data_dir (str): Path to the directory containing the .npy files
            split (str): 'train' or 'test'
        """
        assert split in ['train', 'test'], "split must be 'train' or 'test'"

        X_path = os.path.join(data_dir, f'X_{split}.npy')
        y_path = os.path.join(data_dir, f'y_{split}.npy')

        self.X = torch.tensor(np.load(X_path), dtype=torch.float32)
        self.y = torch.tensor(np.load(y_path), dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return {
            'x': self.X[idx],
            'y': self.y[idx]
        }