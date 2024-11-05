"""
main.py
zachkaupp@gmail.com
"""

import os
import matplotlib.pyplot as plt
import numpy as np
import torch
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

def life_expectancy_epoch_test():
    """life_expectancy_epoch_test()"""
    epochs = 70
    losses = []
    for i in range(20):
        losses.append(ct.train_life_expectancy(learn_rate=1e-2, epochs=epochs, out=False))
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

def life_expectancy_val_test():
    """life_expectancy_val_test()"""
    model = torch.load(fp+"models/nn_life_expectancy.pth")
    data = torch.load(fp+"tensor_dataset")

if __name__ == "__main__":
    #data_transform()
    life_expectancy_val_test()
