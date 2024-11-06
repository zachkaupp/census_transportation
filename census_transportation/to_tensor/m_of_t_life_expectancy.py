"""
m_of_t_life_expectancy.py
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
    # first, import datasets
    acs_id_df = pd.read_pickle(fp+"clean/acs_id.pkl")
    m_of_t_df = pd.read_pickle(fp+"clean/means_of_transport.pkl")
    le_df = pd.read_pickle(fp+"clean/life_expectancy.pkl")
    # make sure the rows match
    le_df = le_df[le_df["ID"].isin(acs_id_df["ID"].tolist())]
    le_df = le_df[le_df["ID"].isin(m_of_t_df["ID"].tolist())]
    m_of_t_df = m_of_t_df[m_of_t_df["ID"].isin(le_df["ID"].tolist())]
    # make sure values are sorted
    le_df = le_df.sort_values(by=["ID"])
    m_of_t_df = m_of_t_df.sort_values(by=["ID"])
    # convert to tensors
    labels_df = le_df.iloc[:,1]
    data_df = m_of_t_df.iloc[:,2:]
    labels = torch.from_numpy(labels_df.values.astype(np.float32))
    data = torch.from_numpy(data_df.values.astype(np.float32))
    labels = labels.unsqueeze(1)
    # convert to tensor dataset
    dataset = data_utils.TensorDataset(data, labels)
    torch.save(dataset, fp+"tensor_dataset/m_of_t_life_expectancy.pt")

if __name__ == "__main__":
    main()
