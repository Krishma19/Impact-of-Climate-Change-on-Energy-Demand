import pandas as pd
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = _REPO_ROOT / "data" / "raw"
DATA_PROCESSED = _REPO_ROOT / "data" / "processed"


# ============================================================
# STEP 0: File paths
# ============================================================
faostat_path = DATA_RAW / "FAOSTAT_data_en_4-29-2026.csv"
co2_path = DATA_RAW / "owid-co2-data.csv"
energy_path = DATA_RAW / "owid-energy-data.csv"
output_path = DATA_PROCESSED / "final_merged.csv"


# ============================================================
# STEP 1: INITIAL DATA INSPECTION
# - Read each dataset separately
# - Show columns, dtypes, sample rows
# - Identify key columns
# ============================================================
print("\n" + "=" * 80)
print("STEP 1: INITIAL DATA INSPECTION")
print("=" * 80)

faostat_raw = pd.read_csv(faostat_path)
co2_raw = pd.read_csv(co2_path)
energy_raw = pd.read_csv(energy_path)

datasets = {
    "FAOSTAT": faostat_raw,
    "OWID CO2": co2_raw,
    "OWID ENERGY": energy_raw,
}

for name, df in datasets.items():
    print(f"\n--- {name} ---")
    print(f"Shape: {df.shape}")
    print("Columns:")
    print(list(df.columns))
    print("\nDtypes:")
    print(df.dtypes)
    print("\nSample rows:")
    print(df.head(3))

print("\nKey columns identified:")
print("- FAOSTAT: Area (country), Year, Element, Months, Value")
print("- OWID CO2: country, year, indicators such as co2_per_capita, gdp, population")
print("- OWID ENERGY: country, year, indicators such as electricity_demand_per_capita, gdp, population")


# ============================================================
# STEP 2: STANDARDIZE COUNTRY NAMES
# - Normalize country names across all datasets
# - Create a mapping dictionary where needed
# ============================================================
print("\n" + "=" * 80)
print("STEP 2: STANDARDIZE COUNTRY NAMES")
print("=" * 80)

# Prepare country columns
faostat_raw["country"] = faostat_raw["Area"].astype(str).str.strip()
co2_raw["country"] = co2_raw["country"].astype(str).str.strip()
energy_raw["country"] = energy_raw["country"].astype(str).str.strip()

# Manual country mapping for common naming differences
country_mapping = {
    "United States of America": "United States",
    "Viet Nam": "Vietnam",
    "Iran (Islamic Republic of)": "Iran",
    "Russian Federation": "Russia",
    "Czechia": "Czech Republic",
    "Republic of Korea": "South Korea",
    "Türkiye": "Turkey",
    "United Republic of Tanzania": "Tanzania",
    "Syrian Arab Republic": "Syria",
    "Bolivia (Plurinational State of)": "Bolivia",
    "Venezuela (Bolivarian Republic of)": "Venezuela",
    "Democratic Republic of the Congo": "Democratic Republic of Congo",
    "Congo": "Republic of Congo",
    "Lao People's Democratic Republic": "Laos",
    "China, Taiwan Province of China": "Taiwan",
    "China, Hong Kong SAR": "Hong Kong",
    "China, Macao SAR": "Macao",
    "Côte d'Ivoire": "Cote d'Ivoire",
    "State of Palestine": "Palestine",
    "Bahamas": "The Bahamas",
    "Gambia": "The Gambia",
}

faostat_raw["country"] = faostat_raw["country"].replace(country_mapping)
co2_raw["country"] = co2_raw["country"].replace(country_mapping)
energy_raw["country"] = energy_raw["country"].replace(country_mapping)

print("Country mapping dictionary used (manual harmonization):")
print(country_mapping)

# Check overlap after harmonization
faostat_countries = set(faostat_raw["country"].unique())
co2_countries = set(co2_raw["country"].unique())
energy_countries = set(energy_raw["country"].unique())

common_countries = faostat_countries.intersection(co2_countries).intersection(energy_countries)
print(f"\nCountries in FAOSTAT: {len(faostat_countries)}")
print(f"Countries in OWID CO2: {len(co2_countries)}")
print(f"Countries in OWID ENERGY: {len(energy_countries)}")
print(f"Common countries across all datasets: {len(common_countries)}")


# ============================================================
# STEP 3: ALIGN TIME DIMENSION
# - Ensure year is integer
# - Find overlapping year range
# - Keep common years only
# ============================================================
print("\n" + "=" * 80)
print("STEP 3: ALIGN TIME DIMENSION")
print("=" * 80)

faostat_raw["year"] = pd.to_numeric(faostat_raw["Year"], errors="coerce").astype("Int64")
co2_raw["year"] = pd.to_numeric(co2_raw["year"], errors="coerce").astype("Int64")
energy_raw["year"] = pd.to_numeric(energy_raw["year"], errors="coerce").astype("Int64")

faostat_years = set(faostat_raw["year"].dropna().astype(int).unique())
co2_years = set(co2_raw["year"].dropna().astype(int).unique())
energy_years = set(energy_raw["year"].dropna().astype(int).unique())

common_years = sorted(list(faostat_years.intersection(co2_years).intersection(energy_years)))
print(f"FAOSTAT year range: {min(faostat_years)} to {max(faostat_years)}")
print(f"OWID CO2 year range: {min(co2_years)} to {max(co2_years)}")
print(f"OWID ENERGY year range: {min(energy_years)} to {max(energy_years)}")
print(f"Common year range: {min(common_years)} to {max(common_years)}")
print(f"Number of common years: {len(common_years)}")

faostat_raw = faostat_raw[faostat_raw["year"].isin(common_years)].copy()
co2_raw = co2_raw[co2_raw["year"].isin(common_years)].copy()
energy_raw = energy_raw[energy_raw["year"].isin(common_years)].copy()


# ============================================================
# STEP 4: TEMPERATURE DATA FILTERING (FAOSTAT)
# - Keep annual temperature change metric only
# - Drop unnecessary columns
# - Aggregate to country-year if duplicates exist
# ============================================================
print("\n" + "=" * 80)
print("STEP 4: TEMPERATURE DATA FILTERING")
print("=" * 80)

temp_df = faostat_raw[
    (faostat_raw["Element"] == "Temperature change")
    & (faostat_raw["Months"] == "Meteorological year")
].copy()

temp_df["temperature_change_c"] = pd.to_numeric(temp_df["Value"], errors="coerce")

temp_df = temp_df[["country", "year", "temperature_change_c"]].copy()

# Aggregate to one row per country-year (safety in case of duplicates)
temp_df = (
    temp_df.groupby(["country", "year"], as_index=False)["temperature_change_c"]
    .mean()
)

print(f"Filtered temperature dataset shape: {temp_df.shape}")
print("Temperature sample:")
print(temp_df.head(5))


# ============================================================
# STEP 5: MERGING STRATEGY
# - Explain join logic
# - Merge step-by-step using INNER JOIN for clean dataset
# ============================================================
print("\n" + "=" * 80)
print("STEP 5: MERGING STRATEGY")
print("=" * 80)
print("Join logic:")
print("1) Start with OWID ENERGY because it contains target variable electricity_demand_per_capita.")
print("2) Inner join with OWID CO2 on [country, year] to keep records present in both.")
print("3) Inner join with filtered FAOSTAT temperature on [country, year] for complete climate linkage.")

energy_keep_cols = [
    "country",
    "year",
    "iso_code",
    "population",
    "gdp",
    "electricity_demand_per_capita",
    "electricity_demand",
    "electricity_generation",
    "energy_per_capita",
    "energy_per_gdp",
    "fossil_share_elec",
    "renewables_share_elec",
    "net_elec_imports_share_demand",
]

co2_keep_cols = [
    "country",
    "year",
    "iso_code",
    "co2",
    "co2_per_capita",
    "co2_per_gdp",
    "co2_per_unit_energy",
    "consumption_co2",
    "consumption_co2_per_capita",
    "temperature_change_from_co2",
    "temperature_change_from_ghg",
]

energy_df = energy_raw[energy_keep_cols].copy()
co2_df = co2_raw[co2_keep_cols].copy()

rows_energy_before = len(energy_df)
rows_co2_before = len(co2_df)
rows_temp_before = len(temp_df)

merged_ec = pd.merge(
    energy_df,
    co2_df,
    on=["country", "year"],
    how="inner",
    suffixes=("_energy", "_co2"),
)

merged_all = pd.merge(
    merged_ec,
    temp_df,
    on=["country", "year"],
    how="inner",
)

print(f"Rows in energy before merge: {rows_energy_before}")
print(f"Rows in co2 before merge: {rows_co2_before}")
print(f"Rows in temp before merge: {rows_temp_before}")
print(f"Rows after energy + co2 merge: {len(merged_ec)}")
print(f"Rows after adding temperature: {len(merged_all)}")


# ============================================================
# STEP 6: DATA QUALITY CHECKS
# - Missing values
# - Duplicate rows
# - Unexpected row drops
# - Retention %
# ============================================================
print("\n" + "=" * 80)
print("STEP 6: DATA QUALITY CHECKS")
print("=" * 80)

missing_summary = merged_all.isna().sum().sort_values(ascending=False)
missing_pct = (merged_all.isna().mean() * 100).sort_values(ascending=False)
missing_report = pd.DataFrame({
    "missing_count": missing_summary,
    "missing_pct": missing_pct.round(2),
})

duplicates_count = merged_all.duplicated(subset=["country", "year"]).sum()

retention_vs_energy = (len(merged_all) / rows_energy_before) * 100 if rows_energy_before > 0 else 0
retention_vs_ec = (len(merged_all) / len(merged_ec)) * 100 if len(merged_ec) > 0 else 0

print("Top columns by missing values:")
print(missing_report.head(15))
print(f"\nDuplicate country-year rows: {duplicates_count}")
print(f"Retention vs ENERGY base: {retention_vs_energy:.2f}%")
print(f"Retention vs ENERGY+CO2 stage: {retention_vs_ec:.2f}%")


# ============================================================
# STEP 7: FINAL CLEANING
# - Remove irrelevant columns
# - Rename columns clearly
# - Ensure consistent dtypes
# - Sort by country and year
# ============================================================
print("\n" + "=" * 80)
print("STEP 7: FINAL CLEANING")
print("=" * 80)

# Resolve iso_code duplication and retain one clean iso_code
if "iso_code_energy" in merged_all.columns and "iso_code_co2" in merged_all.columns:
    merged_all["iso_code"] = merged_all["iso_code_energy"].combine_first(merged_all["iso_code_co2"])
    merged_all.drop(columns=["iso_code_energy", "iso_code_co2"], inplace=True)

# Keep rows where the modeling target exists
final_df = merged_all[merged_all["electricity_demand_per_capita"].notna()].copy()

# Ensure numeric types where expected
numeric_cols = [
    "year",
    "population",
    "gdp",
    "electricity_demand_per_capita",
    "electricity_demand",
    "electricity_generation",
    "energy_per_capita",
    "energy_per_gdp",
    "fossil_share_elec",
    "renewables_share_elec",
    "net_elec_imports_share_demand",
    "co2",
    "co2_per_capita",
    "co2_per_gdp",
    "co2_per_unit_energy",
    "consumption_co2",
    "consumption_co2_per_capita",
    "temperature_change_from_co2",
    "temperature_change_from_ghg",
    "temperature_change_c",
]

for col in numeric_cols:
    if col in final_df.columns:
        final_df[col] = pd.to_numeric(final_df[col], errors="coerce")

final_df["year"] = final_df["year"].astype(int)
final_df = final_df.sort_values(["country", "year"]).reset_index(drop=True)


# ============================================================
# STEP 8: FINAL OUTPUT
# - Show final columns
# - Shape
# - Sample rows
# - Save merged master dataset under data/processed/
# ============================================================
print("\n" + "=" * 80)
print("STEP 8: FINAL OUTPUT")
print("=" * 80)

print("Final column list:")
print(list(final_df.columns))
print(f"\nFinal dataset shape: {final_df.shape}")
print("\nFinal sample rows:")
print(final_df.head(10))

output_path.parent.mkdir(parents=True, exist_ok=True)
final_df.to_csv(output_path, index=False)
print(f"\nSaved cleaned master dataset to: {output_path}")


# ============================================================
# STEP 9: ASSUMPTIONS USED
# ============================================================
print("\n" + "=" * 80)
print("ASSUMPTIONS USED")
print("=" * 80)
print("1) FAOSTAT 'Temperature change' at 'Meteorological year' is treated as annual climate signal.")
print("2) Inner joins are used to keep only country-year rows present across all three sources.")
print("3) Electricity-demand modeling readiness requires non-null electricity_demand_per_capita.")
print("4) Country harmonization uses a manual mapping for common naming differences.")
print("\nBalanced-panel exports (usable data, 2001-2022): python src/build_balanced_alternatives.py")
print(f"- {(DATA_PROCESSED / 'final_merged_balanced_adjusted.csv').name}")
print(f"- {(DATA_PROCESSED / 'final_merged_partial_balanced.csv').name}")
