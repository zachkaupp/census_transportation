"""
life_expectancy.py
zachkaupp@gmail.com
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.utils.data as data_utils

cur_dir = os.path.dirname(__file__)
fp = os.path.join(cur_dir, '../../data/')

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
    labels_df = df.iloc[:, 1]
    data_df = df.iloc[:, 2:]
    labels = torch.from_numpy(labels_df.values.astype(np.float32))
    data = torch.from_numpy(data_df.values.astype(np.float32))
    # make sure data has the right dimensions
    labels = labels.unsqueeze(1)
    # convert to tensor dataset
    dataset = data_utils.TensorDataset(data, labels)
    torch.save(dataset, fp+"tensor_dataset/life_expectancy.pt")

if __name__ == "__main__":
    main()
