"""
main.py
zachkaupp@gmail.com
"""
import numpy as np
import pandas as pd
import torch

# import ACS first -------------------------
df = pd.read_csv("data/raw/acs_data.csv", low_memory=False)
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
# result:
# df = raw percentages
df.to_pickle("data/clean/acs_data", compression=None)
# df_id = ID and county name corresponding to df by row
df_id.to_pickle("data/clean/acs_id", compression=None)

# import travel_time_to_work.csv --------------------------

df = pd.read_csv("data/raw/travel_time_to_work.csv", low_memory=False)
# remove unnecessary columns
df = df.iloc[:,[1,10,11,12,13,14,15,16,17]]
# convert columns to integers instead of strings
for i in range(0,len(df.columns)):
    df.iloc[:,i] = pd.to_numeric(df.iloc[:,i], downcast="integer")
# match GEOID to IDs from acs_id
df["GEOID"] = df["GEOID"].apply(lambda x : x // 100000)
df = df.groupby("GEOID", as_index=False).sum()



def main():
    """main()"""
    print(df.head())


if __name__ == "__main__":
    main()
