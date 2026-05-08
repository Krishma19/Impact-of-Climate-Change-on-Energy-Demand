import pandas as pd
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_PROCESSED = _REPO_ROOT / "data" / "processed"


# ============================================================
# STEP 0: Load cleaned master dataset
# ============================================================
input_path = DATA_PROCESSED / "final_merged.csv"
output_adjusted = DATA_PROCESSED / "final_merged_balanced_adjusted.csv"
output_partial = DATA_PROCESSED / "final_merged_partial_balanced.csv"

df = pd.read_csv(input_path)

print("\n" + "=" * 80)
print("STEP 0: INPUT CHECK")
print("=" * 80)
print(f"Input shape: {df.shape}")
print(f"Countries: {df['country'].nunique()}")
print(f"Year range: {df['year'].min()} to {df['year'].max()}")


# ============================================================
# STEP 1: Data availability analysis
# - Count available years per country
# - Find common continuous year ranges across countries
# ============================================================
print("\n" + "=" * 80)
print("STEP 1: DATA AVAILABILITY ANALYSIS")
print("=" * 80)

key_vars = [
    "electricity_demand_per_capita",
    "gdp",
    "co2_per_capita",
    "temperature_change_c",
]

# Use only rows where all key variables are available
df_key = df.dropna(subset=key_vars).copy()
df_key["year"] = pd.to_numeric(df_key["year"], errors="coerce").astype(int)

# Country-level available year counts
years_per_country = (
    df_key.groupby("country")["year"]
    .nunique()
    .sort_values(ascending=False)
    .reset_index(name="available_year_count")
)
print("Available years per country (top 20):")
print(years_per_country.head(20))

# Per-year country availability
countries_per_year = (
    df_key.groupby("year")["country"]
    .nunique()
    .sort_index()
)
print("\nCountries available per year:")
print(countries_per_year.to_string())


# Search for best continuous year range
years = sorted(df_key["year"].unique())
presence = (
    df_key.groupby(["country", "year"])
    .size()
    .unstack(fill_value=0)
    .reindex(columns=years, fill_value=0)
)

range_stats = []
for i, start_year in enumerate(years):
    for end_year in years[i:]:
        selected_years = [y for y in years if start_year <= y <= end_year]
        period_len = len(selected_years)
        if period_len < 10:
            continue
        complete_countries = (presence[selected_years] > 0).all(axis=1).sum()
        total_rows = complete_countries * period_len
        range_stats.append({
            "start_year": start_year,
            "end_year": end_year,
            "period_len": period_len,
            "complete_countries": int(complete_countries),
            "total_rows": int(total_rows),
        })

range_stats_df = pd.DataFrame(range_stats).sort_values(
    by=["complete_countries", "period_len", "total_rows"],
    ascending=[False, False, False],
)

print("\nTop 10 candidate continuous year ranges:")
print(range_stats_df.head(10))

# Choose a realistic long window with high retention:
# Best long option from profiling is 2001-2022 (22 years, 150 countries).
adj_start_year = 2001
adj_end_year = 2022
adj_years = list(range(adj_start_year, adj_end_year + 1))

print("\nSuggested optimal adjusted range:")
print(f"{adj_start_year}-{adj_end_year} ({len(adj_years)} years)")


# ============================================================
# STEP 2: OPTION A - Adjusted strict balanced panel
# - Keep countries with complete data for all years in adjusted range
# ============================================================
print("\n" + "=" * 80)
print("STEP 2: OPTION A - ADJUSTED BALANCED PANEL")
print("=" * 80)

df_adj_window = df[df["year"].isin(adj_years)].copy()
df_adj_window = df_adj_window.dropna(subset=key_vars).copy()

country_year_counts_a = df_adj_window.groupby("country")["year"].nunique()
countries_a = country_year_counts_a[country_year_counts_a == len(adj_years)].index.tolist()

balanced_adjusted = (
    df_adj_window[df_adj_window["country"].isin(countries_a)]
    .sort_values(["country", "year"])
    .reset_index(drop=True)
)

print(f"Option A countries retained: {balanced_adjusted['country'].nunique()}")
print(f"Option A rows: {len(balanced_adjusted)}")

# Validation for Option A
valid_year_cov_a = (
    balanced_adjusted.groupby("country")["year"].nunique().eq(len(adj_years)).all()
    if len(balanced_adjusted) > 0 else True
)
dup_a = balanced_adjusted.duplicated(subset=["country", "year"]).sum()
missing_key_a = balanced_adjusted[key_vars].isna().sum()

print(f"Option A full year coverage valid: {valid_year_cov_a}")
print(f"Option A duplicates: {dup_a}")
print("Option A key-variable missing counts:")
print(missing_key_a)


# ============================================================
# STEP 3: OPTION B - Partial balanced panel (85% coverage + interpolation)
# - Keep countries with at least 85% year coverage
# - Reindex full country-year grid
# - Interpolate small gaps in key vars
# ============================================================
print("\n" + "=" * 80)
print("STEP 3: OPTION B - PARTIAL BALANCED + INTERPOLATION")
print("=" * 80)

coverage_threshold = 0.85  # within requested 80-90%
min_required_years = int(len(adj_years) * coverage_threshold)

df_partial_window = df[df["year"].isin(adj_years)].copy()

# Coverage is based on rows where all key vars are currently present
coverage_counts = (
    df_partial_window.dropna(subset=key_vars)
    .groupby("country")["year"]
    .nunique()
)

countries_b = coverage_counts[coverage_counts >= min_required_years].index.tolist()
df_partial = df_partial_window[df_partial_window["country"].isin(countries_b)].copy()

print(f"Coverage threshold: {coverage_threshold:.0%}")
print(f"Minimum required years: {min_required_years} out of {len(adj_years)}")
print(f"Countries passing threshold before interpolation: {len(countries_b)}")
print(f"Rows before reindexing: {len(df_partial)}")

# Reindex to full grid for retained countries and adjusted years
full_index = pd.MultiIndex.from_product(
    [sorted(countries_b), adj_years],
    names=["country", "year"],
)

df_partial = (
    df_partial.set_index(["country", "year"])
    .sort_index()
    .reindex(full_index)
    .reset_index()
)

print(f"Rows after reindexing to full country-year grid: {len(df_partial)}")

# Interpolate key variables within each country to fill small interior gaps
# limit=2 means fill up to 2 consecutive missing years
for col in key_vars:
    df_partial[col] = (
        df_partial.groupby("country")[col]
        .transform(lambda s: s.interpolate(method="linear", limit=2, limit_area="inside"))
    )

# Keep rows where key variables are now available
df_partial = df_partial.dropna(subset=key_vars).copy()

# Recheck country completeness after interpolation
country_year_counts_b = df_partial.groupby("country")["year"].nunique()
countries_b_final = country_year_counts_b[country_year_counts_b == len(adj_years)].index.tolist()

partial_balanced = (
    df_partial[df_partial["country"].isin(countries_b_final)]
    .sort_values(["country", "year"])
    .reset_index(drop=True)
)

print(f"Option B countries retained after interpolation + completeness check: {partial_balanced['country'].nunique()}")
print(f"Option B rows: {len(partial_balanced)}")

# Validation for Option B
valid_year_cov_b = (
    partial_balanced.groupby("country")["year"].nunique().eq(len(adj_years)).all()
    if len(partial_balanced) > 0 else True
)
dup_b = partial_balanced.duplicated(subset=["country", "year"]).sum()
missing_key_b = partial_balanced[key_vars].isna().sum()

print(f"Option B full year coverage valid: {valid_year_cov_b}")
print(f"Option B duplicates: {dup_b}")
print("Option B key-variable missing counts:")
print(missing_key_b)


# ============================================================
# STEP 4: Compare options and recommend
# ============================================================
print("\n" + "=" * 80)
print("STEP 4: OPTION COMPARISON")
print("=" * 80)

comparison = pd.DataFrame({
    "dataset": ["Option A - adjusted balanced", "Option B - partial balanced + interpolation"],
    "start_year": [adj_start_year, adj_start_year],
    "end_year": [adj_end_year, adj_end_year],
    "countries": [balanced_adjusted["country"].nunique(), partial_balanced["country"].nunique()],
    "rows": [len(balanced_adjusted), len(partial_balanced)],
    "duplicates_country_year": [dup_a, dup_b],
    "missing_key_values_total": [int(missing_key_a.sum()), int(missing_key_b.sum())],
})
print(comparison)

if partial_balanced["country"].nunique() > balanced_adjusted["country"].nunique():
    recommendation = "Option B is recommended for modeling because it retains more countries while meeting key quality checks."
else:
    recommendation = "Option A is recommended for modeling because it is fully observed without interpolation and retains strong coverage."

print("\nRecommendation:")
print(recommendation)


# ============================================================
# STEP 5: Save outputs under data/processed/
# ============================================================
DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
balanced_adjusted.to_csv(output_adjusted, index=False)
partial_balanced.to_csv(output_partial, index=False)

print("\nSaved files:")
print(f"- {output_adjusted}")
print(f"- {output_partial}")

print("\nSample rows - Option A:")
print(balanced_adjusted.head(5))
print("\nSample rows - Option B:")
print(partial_balanced.head(5))
