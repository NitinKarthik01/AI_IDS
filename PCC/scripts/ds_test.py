import pandas as pd

df = pd.read_csv("/home/dhanush2026/dhanush2026/AI_IDS/DATASETS/MERGED_CSV_Sampled_5Percent.csv")

print(df.shape)

print(df["Label"].value_counts())
