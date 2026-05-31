import pandas as pd
from pathlib import Path

# ============================================================
# VERIFY EDA OUTPUTS
# Project: Impact of Climate Change on Energy Demand
# File Name: verify_eda_outputs.py
# Purpose:
# This script verifies dataset quality, EDA outputs,
# hypothesis testing outputs, and required notebook files.
# ============================================================


# ============================================================
# 1. LOAD FINAL DATASET
# ============================================================

DATA_PATH = Path("data/processed/final_merged.csv")

if not DATA_PATH.exists():
    raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

df = pd.read_csv(DATA_PATH)

print("===================================")
print(" DATASET OVERVIEW")
print("===================================")

print("\nDataset loaded successfully")
print("File:", DATA_PATH)
print("Shape:", df.shape)

print("\nColumns:")
print(df.columns.tolist())

print("\nPreview:")
print(df.head())


# ============================================================
# 2. DATA TYPES AND SUMMARY STATISTICS
# ============================================================

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


# ============================================================
# 3. DATA QUALITY CHECKS
# ============================================================

print("\n===================================")
print(" DATA QUALITY CHECKS")
print("===================================")

print("\n--- Missing Values (%) ---")
missing_df = (df.isnull().sum() / len(df)) * 100
print(missing_df.sort_values(ascending=False))

print("\n--- Duplicate Check: Full Rows ---")
full_duplicate_count = df.duplicated().sum()
print("Duplicate full rows:", full_duplicate_count)

print("\n--- Duplicate Check: Country-Year ---")
country_year_duplicate_count = df.duplicated(subset=["country", "year"]).sum()
print("Duplicate country-year records:", country_year_duplicate_count)

if country_year_duplicate_count > 0:
    duplicate_rows = df[df.duplicated(subset=["country", "year"], keep=False)]
    print("\nDuplicate Country-Year Rows:")
    print(duplicate_rows.head())
else:
    print("No duplicate country-year records found")


# ============================================================
# 4. COUNTRY-YEAR VALIDATION
# ============================================================

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


# ============================================================
# 5. NUMERIC RANGE AND ANOMALY CHECKS
# ============================================================

print("\n===================================")
print(" NUMERIC RANGE & ANOMALY CHECKS")
print("===================================")

numeric_columns = [
    "gdp",
    "population",
    "electricity_demand_per_capita",
    "co2_per_capita",
    "temperature_change_c",
    "renewables_share_elec",
    "fossil_share_elec"
]

for col in numeric_columns:
    if col in df.columns:
        print(f"\n--- {col} ---")
        print("Minimum:", df[col].min())
        print("Maximum:", df[col].max())
        print("Negative values:", (df[col] < 0).sum())
    else:
        print(f"\n[MISSING COLUMN] {col}")

extreme_temp = df[df["temperature_change_c"] > 4]

print("\nExtreme temperature records:", len(extreme_temp))

if len(extreme_temp) > 0:
    print(extreme_temp[["country", "year", "temperature_change_c"]].head())
else:
    print("No extreme temperature anomalies found")


# ============================================================
# 6. REQUIRED ANALYSIS COLUMN CHECK
# ============================================================

print("\n===================================")
print(" REQUIRED ANALYSIS COLUMN VALIDATION")
print("===================================")

required_columns = [
    "country",
    "year",
    "electricity_demand_per_capita",
    "temperature_change_c",
    "co2_per_capita",
    "gdp",
    "renewables_share_elec",
    "fossil_share_elec",
    "population"
]

missing_columns = [
    col for col in required_columns
    if col not in df.columns
]

if len(missing_columns) == 0:
    print("All required analysis columns exist.")
else:
    print("Missing required columns:", missing_columns)


# ============================================================
# 7. US-1 EDA OUTPUT VALIDATION
# ============================================================

print("\n===================================")
print(" US-1 EDA OUTPUT VALIDATION")
print("===================================")

us1_reports = Path("reports/us1_eda_outputs")

required_us1_files = [
    "missing_values_summary.csv",
    "numeric_summary_statistics.csv",
    "negative_value_check.csv",
    "range_check_summary.csv",
    "us1_validation_summary.csv"
]

if us1_reports.exists():
    print("[FOUND] reports/us1_eda_outputs")

    for file_name in required_us1_files:
        file_path = us1_reports / file_name

        if file_path.exists():
            print("[FOUND]", file_path)
        else:
            print("[MISSING]", file_path)
else:
    print("[MISSING] reports/us1_eda_outputs")


# ============================================================
# 8. MAIN EDA / US-3 TREND ANALYSIS VALIDATION
# ============================================================

print("\n===================================")
print(" MAIN EDA / US-3 TREND ANALYSIS VALIDATION")
print("===================================")

trend_files = [
    "data/processed/us3_yearly_trend_dataset.csv",
    "data/processed/us3_selected_trend_data.csv",
    "data/processed/us3_correlation_matrix.csv"
]

for file_path in trend_files:
    file = Path(file_path)

    if file.exists():
        print("[FOUND]", file)
    else:
        print("[MISSING]", file)

us3_reports = Path("reports/us3_eda_outputs")

if us3_reports.exists():
    png_files = list(us3_reports.glob("*.png"))

    print("\nUS-3 EDA charts generated:", len(png_files))

    for file in png_files:
        print("[CHART]", file.name)
else:
    print("[MISSING] reports/us3_eda_outputs")


# ============================================================
# 9. US-4 RELATIONSHIP ANALYSIS VALIDATION
# ============================================================

print("\n===================================")
print(" US-4 RELATIONSHIP ANALYSIS VALIDATION")
print("===================================")

relationship_files = [
    "data/processed/us4_advanced_correlation_matrix.csv",
    "data/processed/us4_country_relationship_analysis.csv"
]

for file_path in relationship_files:
    file = Path(file_path)

    if file.exists():
        print("[FOUND]", file)
    else:
        print("[MISSING]", file)

relationship_reports = Path("reports/us4_relationship_analysis")

if relationship_reports.exists():
    png_files = list(relationship_reports.glob("*.png"))

    print("\nUS-4 relationship charts generated:", len(png_files))

    for file in png_files:
        print("[CHART]", file.name)
else:
    print("[MISSING] reports/us4_relationship_analysis")


# ============================================================
# 10. US-5 FEATURE ENGINEERING VALIDATION
# ============================================================

print("\n===================================")
print(" US-5 FEATURE ENGINEERING VALIDATION")
print("===================================")

feature_file = Path("data/processed/us5_feature_engineered_dataset.csv")

if feature_file.exists():
    print("[FOUND]", feature_file)

    feature_df = pd.read_csv(feature_file)

    print("\nFeature dataset shape:", feature_df.shape)

    engineered_features = [
        "electricity_to_co2_ratio",
        "renewable_fossil_ratio",
        "gdp_per_population",
        "climate_pressure_index",
        "energy_sustainability_score"
    ]

    print("\nChecking engineered features:")

    for feature in engineered_features:
        if feature in feature_df.columns:
            print("[FOUND]", feature)
        else:
            print("[MISSING]", feature)

    print("\nMissing values in engineered features:")
    existing_engineered_features = [
        feature for feature in engineered_features
        if feature in feature_df.columns
    ]

    if existing_engineered_features:
        print(feature_df[existing_engineered_features].isnull().sum())

else:
    print("[MISSING]", feature_file)

insights_file = Path("reports/us5_insights/us5_key_insights.txt")

if insights_file.exists():
    print("[FOUND]", insights_file)
else:
    print("[MISSING]", insights_file)


# ============================================================
# 11. HYPOTHESIS TESTING VALIDATION
# ============================================================

print("\n===================================")
print(" HYPOTHESIS TESTING VALIDATION")
print("===================================")

hypothesis_reports = Path("reports/hypothesis_testing")

required_hypothesis_files = [
    "country_hypothesis_results.csv",
    "ols_summary.txt",
    "01_temperature_vs_demand.png",
    "02_co2_vs_demand.png",
    "03_regression_coefficients.png",
    "04_country_temperature_comparison.png"
]

if hypothesis_reports.exists():
    print("[FOUND] reports/hypothesis_testing")

    for file_name in required_hypothesis_files:
        file_path = hypothesis_reports / file_name

        if file_path.exists():
            print("[FOUND]", file_path)
        else:
            print("[MISSING]", file_path)
else:
    print("[MISSING] reports/hypothesis_testing")


# ============================================================
# 12. EDA NOTEBOOK FILE VALIDATION
# ============================================================

print("\n===================================")
print(" EDA NOTEBOOK FILE VALIDATION")
print("===================================")

eda_notebooks = [
    "notebooks/eda/eda.ipynb",
    "notebooks/eda/eda_us1.ipynb",
    "notebooks/eda/Hypothesys-testing-EDA.ipynb"
]

for notebook in eda_notebooks:
    notebook_path = Path(notebook)

    if notebook_path.exists():
        print("[FOUND]", notebook_path)
    else:
        print("[MISSING]", notebook_path)


# ============================================================
# 13. FINAL EDA VERIFICATION SUMMARY
# ============================================================

print("\n===================================")
print(" FINAL EDA VERIFICATION SUMMARY")
print("===================================")

print("""
EDA Verification Completed.

This script checked:
- Final merged dataset
- Missing values
- Duplicate records
- Country-year validity
- Numeric ranges and anomalies
- Required analysis columns
- US-1 EDA outputs
- Main EDA / US-3 outputs
- US-4 relationship analysis outputs
- US-5 feature engineering outputs
- Hypothesis testing outputs
- EDA notebook files

If all required files show [FOUND], the EDA workflow is properly organized and ready for commit.
""")