"""
main.py
zachkaupp@gmail.com
"""
import numpy as np
import pandas as pd
import torch

# import ACS first -------------------------
df = pd.read_csv("data/raw/acs_data.csv")
# match GEOID to county
df_id = df.iloc[1:,[0,1]]
df_id = df_id.rename(columns={"GEO_ID": "ID"})
df_id["ID"] = df_id["ID"].apply(lambda x : x[-5:])
# select desired columns that end in PE plus GEOID and Name
cols = [True if i[-2:] == "PE" else False for i in df.columns]
df = df.loc[1:,cols]
# convert non-percentages to np.nan
pd.set_option('future.no_silent_downcasting', True)
df = df.replace("(X)", np.nan)
for i in range(0,len(df.columns)):
    df.iloc[:,i] = pd.to_numeric(df.iloc[:,i], downcast="float")
    df.iloc[:,i] = df.iloc[:,i].apply(lambda x : x if x <= 100 else np.nan)
print(len(df.columns))
# drop all columns with missing values
df = df.dropna(axis=1,how="any")
# result:
# df = raw percentages
# df_id = ID and county name corresponding to df by row




def main():
    """main()"""
    print("hello world")


if __name__ == "__main__":
    main()
