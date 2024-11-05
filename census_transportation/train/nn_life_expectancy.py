"""
nn_life_expectancy.py
zachkaupp@gmail.com
"""

import os
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset, TensorDataset

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

def test(dataloader, model, loss_fn, device, out=True):
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
    if out:
        print(f"Avg loss = {test_loss:>8f}")
    return test_loss

def train_val_dataset(dataset, val_split=0.25):
    """train_val_dataset(dataset, val_split)"""
    train_idx, val_idx = train_test_split(list(range(len(dataset))), test_size=val_split)
    datasets = {}
    datasets['train'] = Subset(dataset, train_idx)
    datasets['val'] = Subset(dataset, val_idx)
    return datasets

def main(learn_rate=1e-3, epochs=10, out=True):
    """main()"""
    torch.serialization.add_safe_globals([TensorDataset])
    dataset = torch.load(fp+"tensor_dataset/life_expectancy.pt", weights_only=True)
    datasets = train_val_dataset(dataset, .25)
    # create dataloaders
    train_loader = DataLoader(datasets["train"], batch_size=64, shuffle=True)
    test_loader = DataLoader(datasets["val"], batch_size=64, shuffle=True)

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

    model = NeuralNetwork().to(device)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learn_rate)

    loss_list = []
    for t in range(epochs):
        if out:
            print(f"Epoch {t+1}: ", end="")
        train(train_loader, model, loss_fn, optimizer, device)
        loss_list.append(test(test_loader, model, loss_fn, device, out))

    torch.save(model.state_dict(), fp+"models/nn_life_expectancy.pth")
    if out:
        print("Saved model state to nn_life_expectancy.pth")

    # return loss for each epoch to graph later
    return loss_list

if __name__ == "__main__":
    main()
