"""
nn_life_expectancy.py
zachkaupp@gmail.com
"""

import os
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.utils.data as data_utils

# open file to know how many input values to have in the model
cur_dir = os.path.dirname(__file__)
fp = os.path.join(cur_dir, '../../data/')
    # import ACS
acs_data_df = pd.read_pickle(fp+"clean/acs_data.pkl")

class NeuralNetwork(nn.Module):
    """NeuralNetwork"""
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten(start_dim=1)
        self.stack = nn.Sequential(
            nn.Linear(len(acs_data_df.columns), 1)
        )

    def forward(self, x):
        """forward(self, x)"""
        x = self.flatten(x)
        logits = self.stack(x)
        return logits

def train(dataloader, model, loss_fn, optimizer, device):
    """train(dataloader, model, loss_fn, optimizer, device)"""
    #size = len(dataloader.dataset)
    model.train()
    for batch, (x, y) in enumerate(dataloader):
        x, y = x.to(device), y.to(device)

        # Compute prediction error
        pred = model(x)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # print loss of every tenth batch
        if batch % 10 == 0:
            #loss, current = loss, (batch + 1) * len(X)
            #print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
            pass

def test(dataloader, model, loss_fn, device):
    """test(dataloader, model, loss_fn, device)"""
    #size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            test_loss += loss_fn(pred, y)
    test_loss /= num_batches
    print(f"Avg loss: {test_loss:>8f} \n")

def main():
    """main()"""
    # first, format datasets and convert to dataloader
    acs_id_df = pd.read_pickle(fp+"clean/acs_id.pkl")
    acs_data_df = pd.read_pickle(fp+"clean/acs_data.pkl") #pylint: disable=W0621
    le_df = pd.read_pickle(fp+"clean/life_expectancy.pkl")
    # make sure the rows match
    le_df = le_df[le_df["ID"].isin(acs_id_df["ID"].tolist())]
    acs_data_df = acs_data_df[acs_id_df["ID"].isin(le_df["ID"].tolist())]
    # combine labels (life expectancy) with data (acs)
    acs_data_df.index = le_df.index
    df = pd.concat([le_df,acs_data_df], axis=1)
    # convert to tensors
    train_labels_df = df.iloc[:-300, 1]
    train_data_df = df.iloc[:-300, 2:]
    test_labels_df = df.iloc[-300:, 1]
    test_data_df = df.iloc[-300:, 2:]
    train_labels = torch.from_numpy(train_labels_df.values.astype(np.float32))
    train_data = torch.from_numpy(train_data_df.values.astype(np.float32))
    test_labels = torch.from_numpy(test_labels_df.values.astype(np.float32))
    test_data = torch.from_numpy(test_data_df.values.astype(np.float32))
    # make sure data has the right dimensions
    train_labels = train_labels.unsqueeze(1)
    test_labels = test_labels.unsqueeze(1)
    # convert to tensor dataset
    train_dataset = data_utils.TensorDataset(train_data, train_labels)
    test_dataset = data_utils.TensorDataset(test_data, test_labels)
    #print(train_dataset[0])
    # create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True)

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

    model = NeuralNetwork().to(device)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

    epochs = 10
    for t in range(epochs):
        print(f"Epoch {t+1}:", end="")
        train(train_loader, model, loss_fn, optimizer, device)
        test(test_loader, model, loss_fn, device)
        print("\n")

    torch.save(model.state_dict(), fp+"models/nn_life_expectancy.pth")
    print("Saved model state to nn_life_expectancy.pth")

if __name__ == "__main__":
    main()
