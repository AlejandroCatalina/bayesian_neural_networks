import torch
import torch.utils.data as data


class Dataset(data.Dataset):
    """Characterizes a basic dataset for PyTorch"""

    def __init__(self, data, target):
        """Initialization"""
        self.data = data
        self.target = target

    def __len__(self):
        """Denotes the total number of samples"""
        return len(self.target)

    def __getitem__(self, index):
        """Generates one sample of data"""
        return self.data[index, ], self.target[index]