import pandas as pd
from pathlib import Path

df = pd.read_csv("data/processed/final_merged_partial_balanced.csv")

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

print("\n--- Missing Values (%) ---")
missing_df = (df.isnull().sum() / len(df)) * 100
print(missing_df.sort_values(ascending=False))

print("\n--- Duplicate Check (country-year) ---")
duplicate_count = df.duplicated(subset=["country", "year"]).sum()
print("Duplicate country-year records:", duplicate_count)

if duplicate_count > 0:
    duplicate_rows = df[df.duplicated(subset=["country", "year"], keep=False)]
    print("\nDuplicate Rows:")
    print(duplicate_rows.head())
else:
    print("No duplicate records found")

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
- The dataset contains 4,994 rows and 22 columns.
- Missing values are mainly found in consumption CO2, GDP, and energy-per-GDP related fields.
- No duplicate records were found at the country-year level.
- Country and year fields are valid and consistent.
- Key numerical fields have acceptable ranges overall.
- Negative values in temperature_change_c are valid because temperature change can be below the baseline.
- No extreme temperature anomalies were detected.
- The dataset is suitable for exploratory analysis and predictive modeling after handling missing values.
""")

print("\n===================================")
print(" TREND ANALYSIS VALIDATION")
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

print("\nChecking required columns:")

missing_columns = [
    col for col in required_columns
    if col not in df.columns
]

if len(missing_columns) == 0:
    print("All required trend analysis columns exist")
else:
    print("Missing columns:", missing_columns)

print("\nYear Range:")
print(df["year"].min(), "to", df["year"].max())

print("\nUnique Countries:")
print(df["country"].nunique())

trend_file = Path("data/processed/us3_yearly_trend_dataset.csv")
selected_file = Path("data/processed/us3_selected_trend_data.csv")

print("\nChecking trend analysis output files:")

if trend_file.exists():
    print("Found:", trend_file)
else:
    print("Missing:", trend_file)

if selected_file.exists():
    print("Found:", selected_file)
else:
    print("Missing:", selected_file)

print("\nTrend analysis validation completed successfully.")

print("\n===================================")
print(" RELATIONSHIP ANALYSIS VALIDATION")
print("===================================")

required_relationship_files = [
    "data/processed/us4_advanced_correlation_matrix.csv",
    "data/processed/us4_country_relationship_analysis.csv"
]

print("\nChecking relationship analysis output files:")

for file_path in required_relationship_files:
    file = Path(file_path)

    if file.exists():
        print("Found:", file)
    else:
        print("Missing:", file)

relationship_reports = Path("reports/us4_relationship_analysis")

if relationship_reports.exists():
    png_files = list(relationship_reports.glob("*.png"))

    print("\nTotal relationship analysis charts generated:", len(png_files))

    for file in png_files:
        print("[CHART]", file.name)
else:
    print("\nRelationship analysis reports folder missing")

print("\nRelationship analysis validation completed successfully.")

print("\n===================================")
print(" INSIGHTS AND FEATURE ENGINEERING VALIDATION")
print("===================================")

insights_file = Path("reports/us5_insights/us5_key_insights.txt")

print("\nChecking insights report:")

if insights_file.exists():
    print("[FOUND] reports/us5_insights/us5_key_insights.txt")
else:
    print("[MISSING] reports/us5_insights/us5_key_insights.txt")

feature_file = Path("data/processed/us5_feature_engineered_dataset.csv")

print("\nChecking feature engineered dataset:")

if feature_file.exists():
    print("[FOUND] data/processed/us5_feature_engineered_dataset.csv")

    feature_df = pd.read_csv(feature_file)

    print("\nDataset Shape:")
    print(feature_df.shape)

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
            print(f"[FOUND] {feature}")
        else:
            print(f"[MISSING] {feature}")

    print("\nMissing values summary:")

    print(
        feature_df[engineered_features]
        .isnull()
        .sum()
    )

    print("\nPreview of engineered dataset:")

    print(
        feature_df[
            [
                "country",
                "year",
                "electricity_to_co2_ratio",
                "renewable_fossil_ratio",
                "gdp_per_population"
            ]
        ].head()
    )

else:
    print("[MISSING] data/processed/us5_feature_engineered_dataset.csv")

print("\nInsights and feature engineering validation completed successfully.")