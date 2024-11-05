"""
nn_life_expectancy.py
zachkaupp@gmail.com
"""

import os
import numpy as np
import pandas as pd
import torch
from torch import nn

cur_dir = os.path.dirname(__file__)
fp = os.path.join(cur_dir, '../../data/')
    # import ACS
df = pd.read_pickle(fp+"clean/acs_data.pkl")

class NeuralNetwork(nn.Module):
    """NeuralNetwork"""
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.stack = nn.Sequential(
            nn.Linear(len(df.columns), 1),
        )

    def forward(self, x):
        """forward(self, x)"""
        x = self.flatten(x)
        logits = self.stack(x)
        return logits

def main():
    """main()"""
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

    model = NeuralNetwork().to(device)
    print(model)

if __name__ == "__main__":
    main()
