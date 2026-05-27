from pathlib import Path
import matplotlib

matplotlib.use("Agg")

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = REPO_ROOT / "data" / "processed" / "modeling_ready.csv"
FIGURES = REPO_ROOT / "reports" / "figures"
METRICS_PATH = REPO_ROOT / "reports" / "us12_baseline_comparison.csv"

FIGURES.mkdir(parents=True, exist_ok=True)

if not DATA_PATH.exists():
    raise FileNotFoundError(
        "Missing modeling_ready.csv. Run: python src/run_modeling.py"
    )

df = pd.read_csv(DATA_PATH)

print("Dataset Loaded:", df.shape)

sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (10, 6)

# 1. Electricity Demand Trend
yearly_demand = df.groupby("year")["electricity_demand_per_capita"].mean()

plt.plot(yearly_demand.index, yearly_demand.values, marker="o")
plt.title("Average Electricity Demand Over Time")
plt.xlabel("Year")
plt.ylabel("Electricity Demand Per Capita")
plt.savefig(FIGURES / "electricity_trend.png", bbox_inches="tight")
plt.close()
print("Saved electricity_trend.png")

# 2. Temperature Trend
yearly_temp = df.groupby("year")["temperature_change_c"].mean()

plt.plot(yearly_temp.index, yearly_temp.values, marker="o")
plt.title("Average Temperature Change Over Time")
plt.xlabel("Year")
plt.ylabel("Temperature Change (°C)")
plt.savefig(FIGURES / "temperature_trend.png", bbox_inches="tight")
plt.close()
print("Saved temperature_trend.png")

# 3. CO2 vs Electricity Demand
sns.regplot(
    data=df,
    x="co2_per_capita",
    y="electricity_demand_per_capita"
)

plt.title("CO2 Emissions vs Electricity Demand")
plt.xlabel("CO2 Per Capita")
plt.ylabel("Electricity Demand Per Capita")
plt.savefig(FIGURES / "co2_vs_demand.png", bbox_inches="tight")
plt.close()
print("Saved co2_vs_demand.png")

# 4. Renewable Energy vs Electricity Demand
sns.regplot(
    data=df,
    x="renewables_share_elec",
    y="electricity_demand_per_capita"
)

plt.title("Renewable Energy vs Electricity Demand")
plt.xlabel("Renewable Electricity Share")
plt.ylabel("Electricity Demand Per Capita")
plt.savefig(FIGURES / "renewable_vs_demand.png", bbox_inches="tight")
plt.close()
print("Saved renewable_vs_demand.png")

# 5. Correlation Heatmap
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
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.savefig(FIGURES / "correlation_heatmap.png", bbox_inches="tight")
plt.close()
print("Saved correlation_heatmap.png")

# 6. Model Comparison from actual metrics
if METRICS_PATH.exists():
    metrics_df = pd.read_csv(METRICS_PATH)

    model_labels = {
        "linear_regression": "Linear Regression",
        "ridge_linear": "Ridge Regression",
        "random_forest": "Random Forest"
    }

    metrics_df["model_label"] = metrics_df["model"].replace(model_labels)

    plt.bar(
        metrics_df["model_label"],
        metrics_df["R2"]
    )

    plt.title("Model Comparison")
    plt.ylabel("R2 Score")
    plt.xticks(rotation=15)
    plt.savefig(FIGURES / "model_comparison.png", bbox_inches="tight")
    plt.close()

    print("Saved model_comparison.png")
else:
    print("Skipping model comparison chart: reports/us12_baseline_comparison.csv not found")

print("\nUS-14 visualization pipeline completed successfully.")