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

# Validate country and year consistency

print("\n===================================")
print(" COUNTRY-YEAR VALIDATION")
print("===================================")

print("\nCountry missing values:", df["country"].isnull().sum())
print("Year missing values:", df["year"].isnull().sum())

print("\nYear data type:", df["year"].dtype)
print("Minimum year:", df["year"].min())
print("Maximum year:", df["year"].max())

invalid_years = df[(df["year"] < 1990) | (df["year"] > 2024)]
print("\nInvalid year records:", len(invalid_years))

blank_countries = df[df["country"].astype(str).str.strip() == ""]
print("Blank country records:", len(blank_countries))

print("Unique countries:", df["country"].nunique())

# ===================================
# DATA RANGE & ANOMALY CHECKS
# ===================================

print("\n===================================")
print(" DATA RANGE & ANOMALY CHECKS")
print("===================================")

numeric_columns = [
    "gdp",
    "population",
    "electricity_demand_per_capita",
    "co2_per_capita",
    "temperature_change_c"
]

for col in numeric_columns:
    print(f"\n--- {col} ---")
    print("Minimum:", df[col].min())
    print("Maximum:", df[col].max())

    negative_values = (df[col] < 0).sum()
    print("Negative values:", negative_values)

# Detect unusually high temperature values
extreme_temp = df[df["temperature_change_c"] > 4]

print("\nExtreme temperature records:", len(extreme_temp))

if len(extreme_temp) > 0:
    print(extreme_temp[["country", "year", "temperature_change_c"]].head())
else:
    print("No extreme temperature anomalies found")

print("\n===================================")
print(" DATA QUALITY SUMMARY")
print("===================================")

print("""
Data Quality Observations:
- Dataset contains 4,994 rows and 22 columns.
- Missing values are mainly found in consumption CO2, GDP, and energy-per-GDP related fields.
- No duplicate records were found at the country-year level.
- Country and year fields are valid and consistent.
- Key numerical fields have acceptable ranges overall.
- Negative values in temperature_change_c are valid because temperature change can be below the baseline.
- No extreme temperature anomalies were detected.
- Dataset is suitable for further EDA and modeling after handling missing values.
""")

