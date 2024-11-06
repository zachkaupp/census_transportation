"""
main.py
zachkaupp@gmail.com
"""

import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset
import census_transportation as ct

cur_dir = os.path.dirname(__file__)
fp = os.path.join(cur_dir, 'data/')

def data_transform():
    """perform all necessary data cleaning and processing"""
    ct.process_acs()
    ct.process_life_expectancy()
    ct.process_travel_time()
    ct.process_means_of_transport()
    ct.to_tensor_acs_life_expectancy()
    ct.to_tensor_m_of_t_life_expectancy()

def acs_life_expectancy_val_test(idx=0):
    """print prediction and actual value for given row index"""
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    torch.serialization.add_safe_globals([TensorDataset])
    model = ct.model_acs_life_expectancy()
    model.load_state_dict(torch.load(fp+"models/nn_acs_life_expectancy.pth", weights_only=True))
    model.to(device,dtype=torch.float32)
    model.eval()
    data = torch.load(fp+"tensor_dataset/acs_life_expectancy.pt", weights_only=True)
    x,y = data[idx]
    x,y = x.to(device=device,dtype=torch.float32), y.to(device=device,dtype=torch.float32)
    x,y = x.unsqueeze(1), y.unsqueeze(1)
    pred = model(torch.transpose(x,0,1))
    print(pred.item())
    print(y.item())

def acs_life_expectancy_model_vs_mean():
    """compare model predictions with constant mean prediction.
    print average absolute value residual"""
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    torch.serialization.add_safe_globals([TensorDataset])
    model = ct.model_acs_life_expectancy()
    model.load_state_dict(torch.load(fp+"models/nn_acs_life_expectancy.pth", weights_only=True))
    model.to(device,dtype=torch.float32)
    model.eval()
    data = torch.load(fp+"tensor_dataset/acs_life_expectancy.pt", weights_only=True)

    df = pd.read_pickle(fp+"clean/life_expectancy.pkl")
    mean = df.iloc[:,1].mean()

    # I know this can be more efficient by doing it all at the same time,
    # but right now this is easiest for me
    model_error = 0
    mean_error = 0
    for x,y in data:
        x,y = x.to(device=device,dtype=torch.float32), y.to(device=device,dtype=torch.float32)
        x,y = x.unsqueeze(1), y.unsqueeze(1)
        pred = model(torch.transpose(x,0,1))
        model_error += abs(pred.item() - y.item())
        mean_error += abs(mean - y.item())
    model_error /= len(data)
    mean_error /= len(data)
    print(f"Average error by guessing with model: {model_error:.3f} years")
    print(f"Average error by guessing with mean: {mean_error:.3f} years")

def m_of_t_life_expectancy_model_vs_mean():
    """compare model predictions with constant mean prediction.
    print average absolute value residual"""
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    torch.serialization.add_safe_globals([TensorDataset])
    model = ct.model_m_of_t_life_expectancy()
    model.load_state_dict(torch.load(fp+"models/nn_m_of_t_life_expectancy.pth", weights_only=True))
    model.to(device,dtype=torch.float32)
    model.eval()
    data = torch.load(fp+"tensor_dataset/m_of_t_life_expectancy.pt", weights_only=True)

    df = pd.read_pickle(fp+"clean/life_expectancy.pkl")
    mean = df.iloc[:,1].mean()

    # I know this can be more efficient by doing it all at the same time,
    # but right now this is easiest for me
    model_error = 0
    mean_error = 0
    for x,y in data:
        x,y = x.to(device=device,dtype=torch.float32), y.to(device=device,dtype=torch.float32)
        x,y = x.unsqueeze(1), y.unsqueeze(1)
        pred = model(torch.transpose(x,0,1))
        model_error += abs(pred.item() - y.item())
        mean_error += abs(mean - y.item())
    model_error /= len(data)
    mean_error /= len(data)
    print(f"Average error by guessing with model: {model_error:.3f} years")
    print(f"Average error by guessing with mean: {mean_error:.3f} years")


def epoch_test(train_f, epochs=10, learn_rate=1e-3):
    """train several models with a given number of epochs and track average loss by epoch"""
    losses = []
    for i in range(15):
        losses.append(train_f(learn_rate=learn_rate, epochs=epochs, out=False))
        print(f"Trained model #{i}")
    avg_loss = []
    for i in range(len(losses[0])):
        count = 0
        for j in losses:
            count += j[i].item()
        avg_loss.append(count/len(losses))
    print(avg_loss)
    x = []
    y = []
    exclude = 4
    for i,j in enumerate(avg_loss[exclude:]):
        x.append(i+exclude)
        y.append(j)
    plt.close()
    plt.scatter(x,y)
    plt.show()


if __name__ == "__main__":
    #data_transform()
    #epoch_test(ct.train_m_of_t_life_expectancy, epochs=40)
    m_of_t_life_expectancy_model_vs_mean()
