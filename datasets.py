import math
import torch

class NoiseDataset:
    def __init__(self, dim=2):
        self.dim = dim

    def next_batch(self, batch_size=64):
        return torch.randn(batch_size, self.dim)
