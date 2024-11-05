"""
means_of_transport.py
zachkaupp@gmail.com
"""

import os
import pandas as pd

def main():
    """main()"""

    cur_dir = os.path.dirname(__file__)
    fp = os.path.join(cur_dir, '../../data/')

    # import means_of_transportation_to_work.csv
    df = pd.read_csv(fp+"/raw/means_of_transportation_to_work.csv", low_memory=False)

    # remove unnecessary columns
    df = df.iloc[:,[1,10,11,12,13,14,15,16,17,18]]

    # convert columns to numbers instead of strings
    old_df = df
    df = df[0:0]
    df.iloc[:,0] = old_df.iloc[:,0].astype(int)
    df.iloc[:,1:(len(df.columns))] = old_df.iloc[:,1:(len(old_df.columns))].astype(float)

    # match GEOID to IDs from acs_id
    df["GEOID"] = df["GEOID"].apply(lambda x : x // 100000)

    # group all data points in the same county
    df = df.groupby("GEOID", as_index=False).sum()

    # divide to get percentages
    for i in range(2,len(df.columns)):
        df.iloc[:,i] = df.iloc[:,i].div(pd.to_numeric(df.iloc[:,1], downcast="float"))

    # rename columns
    cols = df.columns
    cols = map((lambda x : x[8:] if len(x) > 15 else x), cols)
    df.columns = cols
    df = df.rename(columns={"GEOID": "ID"})

    # store in pickle
    df.to_pickle(fp+"/clean/means_of_transport.pkl", compression=None)

    return df

if __name__ == "__main__":
    main()
