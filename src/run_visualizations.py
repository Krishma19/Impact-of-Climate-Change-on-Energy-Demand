from pathlib import Path
import matplotlib
matplotlib.use("Agg")

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================================
# Paths
# ==========================================================

REPO_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = REPO_ROOT / "data" / "processed" / "modeling_ready.csv"

FIGURES = REPO_ROOT / "reports" / "figures"

# ==========================================================
# Style
# ==========================================================

sns.set_style("whitegrid")

plt.rcParams["figure.figsize"] = (10, 6)

# ==========================================================
# Load dataset
# ==========================================================

df = pd.read_csv(DATA_PATH)

print("Dataset Loaded:", df.shape)

# ==========================================================
# Electricity Demand Trend
# ==========================================================

yearly_demand = (
    df.groupby("year")["electricity_demand_per_capita"]
    .mean()
)

plt.plot(yearly_demand.index, yearly_demand.values)

plt.title("Average Electricity Demand Over Time")

plt.xlabel("Year")

plt.ylabel("Electricity Demand Per Capita")

plt.savefig(FIGURES / "electricity_trend.png")

plt.close()

print("Saved electricity_trend.png")

# ==========================================================
# Temperature Trend
# ==========================================================

yearly_temp = (
    df.groupby("year")["temperature_change_c"]
    .mean()
)

plt.plot(yearly_temp.index, yearly_temp.values)

plt.title("Average Temperature Change Over Time")

plt.xlabel("Year")

plt.ylabel("Temperature Change (°C)")

plt.savefig(FIGURES / "temperature_trend.png")

plt.close()

print("Saved temperature_trend.png")

# ==========================================================
# CO2 vs Demand
# ==========================================================

sns.regplot(
    data=df,
    x="co2_per_capita",
    y="electricity_demand_per_capita"
)

plt.title("CO₂ Emissions vs Electricity Demand")

plt.xlabel("CO₂ Per Capita")

plt.ylabel("Electricity Demand Per Capita")

plt.savefig(FIGURES / "co2_vs_demand.png")

plt.close()

print("Saved co2_vs_demand.png")

# ==========================================================
# Renewable vs Demand
# ==========================================================

sns.regplot(
    data=df,
    x="renewables_share_elec",
    y="electricity_demand_per_capita"
)

plt.title("Renewable Energy vs Electricity Demand")

plt.xlabel("Renewable Electricity Share")

plt.ylabel("Electricity Demand Per Capita")

plt.savefig(FIGURES / "renewable_vs_demand.png")

plt.close()

print("Saved renewable_vs_demand.png")

# ==========================================================
# Correlation Heatmap
# ==========================================================

corr_columns = [
    "electricity_demand_per_capita",
    "temperature_change_c",
    "co2_per_capita",
    "gdp",
    "renewables_share_elec",
    "fossil_share_elec",
    "population"
]

corr_matrix = df[corr_columns].corr()

plt.figure(figsize=(10, 6))

sns.heatmap(
    corr_matrix,
    annot=True,
    cmap="coolwarm"
)

plt.title("Correlation Heatmap")

plt.savefig(FIGURES / "correlation_heatmap.png")

plt.close()

print("Saved correlation_heatmap.png")

# ==========================================================
# Model Comparison
# ==========================================================

models = [
    "Linear Regression",
    "Ridge Regression",
    "Random Forest"
]

scores = [0.42, 0.46, 0.83]

plt.bar(models, scores)

plt.title("Model Comparison")

plt.ylabel("R² Score")

plt.savefig(FIGURES / "model_comparison.png")

plt.close()

print("Saved model_comparison.png")

print("\nUS-14 visualization pipeline completed successfully.")