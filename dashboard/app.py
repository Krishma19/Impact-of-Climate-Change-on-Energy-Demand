from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.linear_model import LinearRegression

st.set_page_config(
    page_title="Climate Change & Energy Dashboard",
    layout="wide"
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = REPO_ROOT / "data" / "processed" / "modeling_ready.csv"
FIGURES = REPO_ROOT / "reports" / "figures"
METRICS_PATH = REPO_ROOT / "reports" / "us12_baseline_comparison.csv"

st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #eef7ff 0%, #f7fff4 45%, #fff7ed 100%);
}
.big-title {
    font-size: 42px;
    font-weight: 800;
    color: #1f2937;
}
.subtitle {
    font-size: 18px;
    color: #4b5563;
}
</style>
""", unsafe_allow_html=True)

st.markdown(
    '<div class="big-title">Climate Change Impact on Energy Demand</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Interactive dashboard for climate trends, energy demand, model results, and forecasting.</div>',
    unsafe_allow_html=True
)

if not DATA_PATH.exists():
    st.error(
        "Missing modeling_ready.csv. Please run the modeling pipeline first: python src/run_modeling.py"
    )
    st.stop()

df = pd.read_csv(DATA_PATH)


def show_image(path: Path, caption: str):
    if path.exists():
        st.image(str(path), caption=caption, use_container_width=True)
    else:
        st.warning(f"Missing figure: {path.name}. Please run python src/run_visualizations.py")


st.sidebar.header("Dashboard Filters")

countries = sorted(df["country"].unique())

selected_countries = st.sidebar.multiselect(
    "Select Country/Countries",
    countries,
    default=["Canada"] if "Canada" in countries else countries[:1]
)

if not selected_countries:
    st.warning("Please select at least one country.")
    st.stop()

year_range = st.sidebar.slider(
    "Select Year Range",
    int(df["year"].min()),
    int(df["year"].max()),
    (int(df["year"].min()), int(df["year"].max()))
)

top_n = st.sidebar.slider("Top N Countries", 5, 20, 10)

x_var = st.sidebar.selectbox(
    "Scatter X Variable",
    [
        "temperature_change_c",
        "co2_per_capita",
        "gdp",
        "population",
        "renewables_share_elec",
        "fossil_share_elec"
    ]
)

forecast_country = st.sidebar.selectbox(
    "Forecast Country",
    countries,
    index=countries.index("Canada") if "Canada" in countries else 0
)

forecast_years = st.sidebar.slider("Forecast Years Ahead", 1, 10, 5)

filtered_df = df[
    (df["country"].isin(selected_countries)) &
    (df["year"] >= year_range[0]) &
    (df["year"] <= year_range[1])
].copy()

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Overview",
    "Trends",
    "Relationships",
    "Model Results",
    "Forecasting"
])

with tab1:
    st.header("Overview")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Total Countries", df["country"].nunique())
    col2.metric("Year Range", f"{df['year'].min()} - {df['year'].max()}")
    col3.metric("Filtered Records", filtered_df.shape[0])
    col4.metric(
        "Avg Demand",
        round(filtered_df["electricity_demand_per_capita"].mean(), 2)
    )

    st.subheader(f"Top {top_n} Countries by Average Electricity Demand")

    top_countries = (
        df.groupby("country", as_index=False)["electricity_demand_per_capita"]
        .mean()
        .sort_values("electricity_demand_per_capita", ascending=False)
        .head(top_n)
    )

    fig = px.bar(
        top_countries,
        x="electricity_demand_per_capita",
        y="country",
        orientation="h",
        text="electricity_demand_per_capita",
        title=f"Top {top_n} Countries by Average Electricity Demand",
        labels={
            "electricity_demand_per_capita": "Avg Electricity Demand Per Capita",
            "country": "Country"
        }
    )

    fig.update_traces(texttemplate="%{text:.2f}", textposition="outside")
    fig.update_layout(yaxis={"categoryorder": "total ascending"})
    st.plotly_chart(fig, use_container_width=True)

    st.download_button(
        "Download Filtered Data",
        filtered_df.to_csv(index=False),
        "filtered_dashboard_data.csv",
        "text/csv"
    )

with tab2:
    st.header("Trend Analysis")

    yearly = (
        filtered_df.groupby(["country", "year"], as_index=False)
        .agg({
            "electricity_demand_per_capita": "mean",
            "temperature_change_c": "mean",
            "co2_per_capita": "mean",
            "renewables_share_elec": "mean"
        })
    )

    fig = px.line(
        yearly,
        x="year",
        y="electricity_demand_per_capita",
        color="country",
        markers=True,
        text=yearly["electricity_demand_per_capita"].round(2),
        title="Electricity Demand Over Time",
        labels={
            "year": "Year",
            "electricity_demand_per_capita": "Electricity Demand Per Capita"
        }
    )

    fig.update_traces(textposition="top center")
    st.plotly_chart(fig, use_container_width=True)

    fig = px.line(
        yearly,
        x="year",
        y="temperature_change_c",
        color="country",
        markers=True,
        text=yearly["temperature_change_c"].round(2),
        title="Temperature Change Over Time",
        labels={
            "year": "Year",
            "temperature_change_c": "Temperature Change (°C)"
        }
    )

    fig.update_traces(textposition="top center")
    st.plotly_chart(fig, use_container_width=True)

    fig = px.line(
        yearly,
        x="year",
        y="renewables_share_elec",
        color="country",
        markers=True,
        text=yearly["renewables_share_elec"].round(2),
        title="Renewable Electricity Share Over Time",
        labels={
            "year": "Year",
            "renewables_share_elec": "Renewable Electricity Share"
        }
    )

    fig.update_traces(textposition="top center")
    st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.header("Relationship Analysis")

    fig = px.scatter(
        filtered_df,
        x=x_var,
        y="electricity_demand_per_capita",
        color="country",
        hover_data=["year", "country"],
        trendline="ols",
        title=f"{x_var.replace('_', ' ').title()} vs Electricity Demand",
        labels={
            x_var: x_var.replace("_", " ").title(),
            "electricity_demand_per_capita": "Electricity Demand Per Capita"
        }
    )

    fig.update_traces(marker=dict(size=9, opacity=0.75))
    st.plotly_chart(fig, use_container_width=True)

    show_image(FIGURES / "correlation_heatmap.png", "Correlation Heatmap")

    st.info(
        "These charts help evaluate relationships between climate, emissions, renewable energy, and electricity demand."
    )

with tab4:
    st.header("Model Results")

    col1, col2 = st.columns(2)

    with col1:
        show_image(FIGURES / "model_comparison.png", "Model Comparison")

    with col2:
        show_image(FIGURES / "us11_feature_importance.png", "Random Forest Feature Importance")

    if METRICS_PATH.exists():
        metrics_df = pd.read_csv(METRICS_PATH)

        st.subheader("Model Metrics Table")
        st.dataframe(metrics_df)

        fig = px.bar(
            metrics_df,
            x="model",
            y=["RMSE", "MAE", "R2"],
            barmode="group",
            text_auto=True,
            title="Model Metrics Comparison",
            labels={"value": "Metric Value", "model": "Model"}
        )

        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Missing model metrics file: reports/us12_baseline_comparison.csv")

with tab5:
    st.header("Future Electricity Demand Forecasting")

    country_df = df[df["country"] == forecast_country].sort_values("year").copy()

    if len(country_df) < 2:
        st.warning("Not enough data available for forecasting.")
        st.stop()

    X = country_df[["year"]]
    y = country_df["electricity_demand_per_capita"]

    model = LinearRegression()
    model.fit(X, y)

    last_year = int(country_df["year"].max())

    future_df = pd.DataFrame({
        "year": list(range(last_year + 1, last_year + forecast_years + 1))
    })

    future_df["electricity_demand_per_capita"] = model.predict(future_df[["year"]])
    future_df["country"] = forecast_country
    future_df["type"] = "Forecast"

    actual_df = country_df[[
        "country",
        "year",
        "electricity_demand_per_capita"
    ]].copy()

    actual_df["type"] = "Actual"

    combined = pd.concat([actual_df, future_df], ignore_index=True)

    fig = px.line(
        combined,
        x="year",
        y="electricity_demand_per_capita",
        color="type",
        markers=True,
        text="electricity_demand_per_capita",
        title=f"Electricity Demand Forecast for {forecast_country}",
        labels={
            "year": "Year",
            "electricity_demand_per_capita": "Electricity Demand Per Capita",
            "type": "Data Type"
        }
    )

    fig.update_traces(texttemplate="%{text:.2f}", textposition="top center")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Forecast Table")
    st.dataframe(future_df[["country", "year", "electricity_demand_per_capita"]])

    st.warning(
        "Forecast uses a simple linear trend model and should be interpreted as exploratory, not causal."
    )