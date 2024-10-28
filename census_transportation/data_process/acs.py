"""
acs.py
zachkaupp@gmail.com
"""
import os
import numpy as np
import pandas as pd

def main():
    """main()"""
    cur_dir = os.path.dirname(__file__)
    fp = os.path.join(cur_dir, '../../data/')
    # import ACS
    df = pd.read_csv(fp+"raw/acs_data.csv", low_memory=False)

    # match GEOID to county
    df_id = df.iloc[1:,[0,1]]
    df_id = df_id.rename(columns={"GEO_ID": "ID"})
    df_id["ID"] = df_id["ID"].apply(lambda x : x[-5:])
    df_id["ID"] = pd.to_numeric(df_id["ID"], downcast="integer")

    # select desired columns that end in PE plus GEOID and Name
    cols = [True if i[-2:] == "PE" else False for i in df.columns]
    df = df.loc[1:,cols]

    # convert non-percentages to np.nan
    pd.set_option('future.no_silent_downcasting', True)
    df = df.replace("(X)", np.nan)
    for i in range(0,len(df.columns)):
        df.iloc[:,i] = pd.to_numeric(df.iloc[:,i], downcast="float")
        df.iloc[:,i] = df.iloc[:,i].apply(lambda x : x if x <= 100 else np.nan)

    # drop all columns with missing values
    df = df.dropna(axis=1,how="any")

    # divide by 100 to get 0-1 percentages instead of 0-100
    for i in range(len(df.columns)):
        df.iloc[:,i] = df.iloc[:,i].apply(lambda x : x / 100)

    # result:
    # df = raw percentages
    df.to_pickle(fp+"clean/acs_data", compression=None)
    # df_id = ID and county name corresponding to df by row
    df_id.to_pickle(fp+"clean/acs_id", compression=None)
    df_id.head()

if __name__ == "__main__":
    main()
