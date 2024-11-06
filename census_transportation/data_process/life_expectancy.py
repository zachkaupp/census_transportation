"""
life_expectancy.py
zachkaupp@gmail.com
"""

import os
import pandas as pd

def main():
    """main()"""
    cur_dir = os.path.dirname(__file__)
    fp = os.path.join(cur_dir, '../../data/')

    # import life_expectancy.csv
    df = pd.read_csv(fp+"raw/life_expectancy.csv", low_memory=False)

    # remove unnecessary columns
    df = df.iloc[:,[0,4]]

    # convert columns to the correct data type
    old_df = df
    df = df[0:0]
    df.iloc[:,0] = old_df.iloc[:,0].astype(int)
    df.iloc[:,1] = old_df.iloc[:,1].astype(float)

    # rename the columns
    df = df.rename(columns={"Tract ID": "ID", "e(0)":"life_expectancy"})

    # truncate ID to county
    df["ID"] = df["ID"].apply(lambda x : x // 1000000)

    # average out life expectancy by census tract
    # without weighting by population
    df = df.groupby("ID", as_index=False).mean()

    # scale expectancy between 0-1
    #mn = df.iloc[:,1].min()
    #mx = df.iloc[:,1].max()
    #df.iloc[:,1] = df.iloc[:,1].apply(lambda x: (x - mn)/(mx-mn))
    #^previously scaled from 0-1, I decided this was unnecessary
    mean = df.iloc[:,1].mean()
    df.iloc[:,1] = df.iloc[:,1].apply(lambda x: (x - mean))
    #scaled to have mean of 0 instead

    df.to_pickle(fp+"clean/life_expectancy.pkl")

    return df


if __name__ == "__main__":
    main()
