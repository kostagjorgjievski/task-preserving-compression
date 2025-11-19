import pandas as pd
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset


class ETTWindowDataset(Dataset):
    """
    Converts an ETT csv file into overlapping normalized windows.

    Returns:
        x: [L, C] float32
    """

    def __init__(self, csv_path, context_length, stride=1, drop_time_col=True):
        self.csv_path = Path(csv_path)
        self.context_length = context_length
        self.stride = stride

        df = pd.read_csv(self.csv_path)

        if drop_time_col and "date" in df.columns:
            df = df.drop(columns=["date"])

        values = df.values.astype("float32")

        # per feature normalization
        self.mean = values.mean(axis=0, keepdims=True)
        self.std = values.std(axis=0, keepdims=True) + 1e-6
        values_norm = (values - self.mean) / self.std

        self.values_norm = values_norm
        self.num_channels = values_norm.shape[1]

        self.indexes = self._build_indexes(len(values_norm), context_length, stride)

    @staticmethod
    def _build_indexes(T, L, stride):
        idx = []
        for start in range(0, T - L + 1, stride):
            end = start + L
            idx.append((start, end))
        return idx

    def __len__(self):
        return len(self.indexes)

    def __getitem__(self, idx):
        start, end = self.indexes[idx]
        x = self.values_norm[start:end]  # [L, C]
        return x
