from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st

# Page setup
st.set_page_config(
    page_title="Climate Change & Energy Dashboard",
    layout="wide"
)

# Paths
REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = REPO_ROOT / "data" / "processed" / "modeling_ready.csv"
FIGURES = REPO_ROOT / "reports" / "figures"

# Load data
df = pd.read_csv(DATA_PATH)

# Title
st.title("Climate Change Impact on Energy Demand")

st.markdown("""
This dashboard analyzes climate change indicators, electricity demand,
renewable energy trends, and machine learning model performance.
""")

# Sidebar filters
st.sidebar.header("Filters")

countries = sorted(df["country"].unique())

selected_country = st.sidebar.selectbox(
    "Select Country",
    countries
)

year_range = st.sidebar.slider(
    "Select Year Range",
    int(df["year"].min()),
    int(df["year"].max()),
    (
        int(df["year"].min()),
        int(df["year"].max())
    )
)

filtered_df = df[
    (df["country"] == selected_country)
    & (df["year"] >= year_range[0])
    & (df["year"] <= year_range[1])
]

# Tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "Overview",
    "Trends",
    "Relationships",
    "Model Results"
])

# =========================
# TAB 1: OVERVIEW
# =========================
with tab1:
    st.header("Project Overview")

    st.write("""
    This project investigates how climate change indicators such as temperature change,
    CO₂ emissions, renewable electricity share, and economic factors influence
    electricity demand per capita across countries.
    """)

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Total Countries", df["country"].nunique())
    col2.metric("Year Range", f"{df['year'].min()} - {df['year'].max()}")
    col3.metric(
        "Avg Electricity Demand",
        round(df["electricity_demand_per_capita"].mean(), 2)
    )
    col4.metric("Selected Country Records", filtered_df.shape[0])

    st.subheader("Top 10 Countries by Average Electricity Demand")

    top10 = (
        df.groupby("country")["electricity_demand_per_capita"]
        .mean()
        .sort_values(ascending=False)
        .head(10)
    )

    fig, ax = plt.subplots(figsize=(10, 5))
    top10.sort_values().plot(kind="barh", ax=ax)
    ax.set_xlabel("Average Electricity Demand Per Capita")
    ax.set_ylabel("Country")
    ax.set_title("Top 10 Countries by Electricity Demand")
    st.pyplot(fig)

# =========================
# TAB 2: TRENDS
# =========================
with tab2:
    st.header("Trend Analysis")

    st.subheader(f"Electricity Demand Trend: {selected_country}")

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(
        filtered_df["year"],
        filtered_df["electricity_demand_per_capita"],
        marker="o"
    )
    ax.set_xlabel("Year")
    ax.set_ylabel("Electricity Demand Per Capita")
    ax.set_title(f"Electricity Demand Over Time - {selected_country}")
    st.pyplot(fig)

    st.write("""
    Insight: This chart shows how electricity demand per capita changes over time
    for the selected country.
    """)

    st.subheader(f"Temperature Change Trend: {selected_country}")

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(
        filtered_df["year"],
        filtered_df["temperature_change_c"],
        marker="o"
    )
    ax.set_xlabel("Year")
    ax.set_ylabel("Temperature Change (°C)")
    ax.set_title(f"Temperature Change Over Time - {selected_country}")
    st.pyplot(fig)

    st.write("""
    Insight: This chart shows how temperature change varies over time for the selected country.
    """)

# =========================
# TAB 3: RELATIONSHIPS
# =========================
with tab3:
    st.header("Relationship Analysis")

    st.subheader("Temperature Change vs Electricity Demand")

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.regplot(
        data=df,
        x="temperature_change_c",
        y="electricity_demand_per_capita",
        ax=ax
    )
    ax.set_title("Temperature Change vs Electricity Demand")
    ax.set_xlabel("Temperature Change (°C)")
    ax.set_ylabel("Electricity Demand Per Capita")
    st.pyplot(fig)

    st.subheader("CO₂ Emissions vs Electricity Demand")
    st.image(str(FIGURES / "co2_vs_demand.png"))

    st.subheader("Renewable Energy vs Electricity Demand")
    st.image(str(FIGURES / "renewable_vs_demand.png"))

    st.subheader("Correlation Heatmap")
    st.image(str(FIGURES / "correlation_heatmap.png"))

    st.write("""
    Insight: These visualizations help evaluate relationships between climate,
    emissions, renewable energy, and electricity demand.
    """)

# =========================
# TAB 4: MODEL RESULTS
# =========================
with tab4:
    st.header("Model Results")

    st.subheader("Model Comparison")
    st.image(str(FIGURES / "model_comparison.png"))

    st.subheader("Random Forest Feature Importance")
    st.image(str(FIGURES / "us11_feature_importance.png"))

    st.markdown("""
    Key Findings:
    - Random Forest performed better than linear baseline models.
    - Feature importance helps identify which variables contribute most to prediction.
    - Model results support the predictive goal of estimating electricity demand.
    """)