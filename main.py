"""
main.py
zachkaupp@gmail.com
"""

import os
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import TensorDataset
import census_transportation as ct

cur_dir = os.path.dirname(__file__)
fp = os.path.join(cur_dir, 'data/')

def data_transform():
    """data_transform()"""
    ct.process_acs()
    ct.process_life_expectancy()
    ct.process_travel_time()
    ct.process_means_of_transport()
    ct.to_tensor_life_expectancy()

def life_expectancy_epoch_test(epochs=10, learn_rate=1e-3):
    """life_expectancy_epoch_test()"""
    losses = []
    for i in range(20):
        losses.append(ct.train_life_expectancy(learn_rate=learn_rate, epochs=epochs, out=False))
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
    exclude = 2
    for i,j in enumerate(avg_loss[exclude:]):
        x.append(i+exclude)
        y.append(j)
    plt.close()
    plt.scatter(x,y)
    plt.show()
    # does slightly better than the standard deviation of life expectancy

def life_expectancy_val_test(idx=0):
    """life_expectancy_val_test()"""
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    torch.serialization.add_safe_globals([TensorDataset])
    model = ct.model_life_expectancy()
    model.load_state_dict(torch.load(fp+"models/nn_life_expectancy.pth", weights_only=True))
    model.to(device,dtype=torch.float32)
    model.eval()
    data = torch.load(fp+"tensor_dataset/life_expectancy.pt", weights_only=True)
    x,y = data[idx]
    x,y = x.to(device=device,dtype=torch.float32), y.to(device=device,dtype=torch.float32)
    x,y = x.unsqueeze(1), y.unsqueeze(1)
    pred = model(torch.transpose(x,0,1))
    print(pred.item())
    print(y.item())


if __name__ == "__main__":
    data_transform()
    life_expectancy_epoch_test(epochs=100, learn_rate=1e-4)
