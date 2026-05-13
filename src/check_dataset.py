import pandas as pd

df = pd.read_csv("data/processed/final_merged.csv")

print("===================================")
print(" DATASET OVERVIEW")
print("===================================")

print("\nDataset loaded successfully")
print("Shape:", df.shape)

print("\nColumns:")
print(df.columns.tolist())

print("\nPreview:")
print(df.head())

print("\n===================================")
print(" DATA TYPES & INFO")
print("===================================")

print("\nData Types:")
print(df.dtypes)

print("\nBasic Info:")
df.info()

print("\n===================================")
print(" SUMMARY STATISTICS")
print("===================================")

print(df.describe())

print("\n===================================")
print(" VALIDATION CHECKS")
print("===================================")

# Missing values
print("\n--- Missing Values (%) ---")
missing_df = (df.isnull().sum() / len(df)) * 100
print(missing_df.sort_values(ascending=False))

# Duplicate check
print("\n--- Duplicate Check (country-year) ---")
duplicate_count = df.duplicated(subset=["country", "year"]).sum()
print("Duplicate country-year records:", duplicate_count)

if duplicate_count > 0:
    duplicate_rows = df[df.duplicated(subset=["country", "year"], keep=False)]
    print("\nDuplicate Rows:")
    print(duplicate_rows.head())
else:
    print("No duplicate records found")
    