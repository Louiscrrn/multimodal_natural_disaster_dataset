import pandas as pd

df = pd.read_csv("../data/processed/ibtracs_era5_20251218_1520_reliefweb.csv")

# number of cyclones to keep
x = 15   

first_x_cyclones = df["Storm_ID"].unique()[:x]
df_subset = df[df["Storm_ID"].isin(first_x_cyclones)]

output_path = f"../data/processed/small_{x}_ibtracs_era5_20251218_1520_reliefweb.csv"
df_subset.to_csv(output_path, index=False)