import pandas as pd

df = pd.read_csv("data/processed/final_merged.csv")

print("Dataset loaded successfully")
print("Shape:", df.shape)

print("\nColumns:")
print(df.columns.tolist())

print("\nPreview:")
print(df.head())