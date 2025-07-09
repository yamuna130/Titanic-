import pandas as pd
df = pd.read_csv('train.csv')
print("Dataset Loaded ✅")
print("Shape:", df.shape)
print(df.head())
print(df.info())
print(df.describe(include='all'))