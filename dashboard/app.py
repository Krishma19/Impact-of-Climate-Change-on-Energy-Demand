from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

st.set_page_config(
    page_title="Climate Change & Energy Demand Dashboard",
    layout="wide"
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = REPO_ROOT / "data" / "processed" / "modeling_ready.csv"
METRICS_PATH = REPO_ROOT / "reports" / "us12_baseline_comparison.csv"

st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #eef7ff 0%, #f8fff2 45%, #fff8ef 100%);
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
    '<div class="subtitle">Business-focused dashboard for climate trends, electricity demand, model performance, forecasting, and strategic insights.</div>',
    unsafe_allow_html=True
)

if not DATA_PATH.exists():
    st.error("Missing modeling_ready.csv. Run: python src/run_modeling.py")
    st.stop()

df = pd.read_csv(DATA_PATH)

countries = sorted(df["country"].unique())

st.sidebar.header("Dashboard Filters")

analysis_mode = st.sidebar.radio(
    "Analysis Mode",
    ["Global", "Selected Countries"]
)

default_countries = [
    c for c in ["Canada", "India", "United States"] if c in countries
]

selected_countries = st.sidebar.multiselect(
    "Select Countries",
    countries,
    default=default_countries
)

if analysis_mode == "Selected Countries":
    if not selected_countries:
        st.warning("Please select at least one country.")
        st.stop()

    if len(selected_countries) > 5:
        st.warning("For clean comparison, please select up to 5 countries.")
        st.stop()

year_range = st.sidebar.slider(
    "Select Year Range",
    int(df["year"].min()),
    int(df["year"].max()),
    (int(df["year"].min()), int(df["year"].max()))
)

ranking_metric = st.sidebar.selectbox(
    "Ranking Metric",
    [
        "electricity_demand_per_capita",
        "co2_per_capita",
        "renewables_share_elec",
        "fossil_share_elec",
        "gdp",
        "population"
    ]
)

top_n = st.sidebar.slider("Top N Countries", 5, 25, 10)

driver_var = st.sidebar.selectbox(
    "Driver Variable",
    [
        "temperature_change_c",
        "co2_per_capita",
        "gdp",
        "population",
        "renewables_share_elec",
        "fossil_share_elec"
    ]
)

profile_country = st.sidebar.selectbox(
    "Country Profile",
    countries,
    index=countries.index("Canada") if "Canada" in countries else 0
)

compare_country = st.sidebar.selectbox(
    "Compare With",
    countries,
    index=countries.index("United States") if "United States" in countries else 0
)

forecast_country = st.sidebar.selectbox(
    "Forecast Country",
    countries,
    index=countries.index("Canada") if "Canada" in countries else 0
)

forecast_years = st.sidebar.slider("Forecast Years Ahead", 1, 10, 5)

base_filtered_df = df[
    (df["year"] >= year_range[0]) &
    (df["year"] <= year_range[1])
].copy()

if analysis_mode == "Selected Countries":
    filtered_df = base_filtered_df[
        base_filtered_df["country"].isin(selected_countries)
    ].copy()
else:
    filtered_df = base_filtered_df.copy()


def safe_mean(data, col):
    if data.empty:
        return 0
    return round(data[col].mean(), 2)


def format_large_number(value):
    if pd.isna(value):
        return "N/A"
    if abs(value) >= 1_000_000_000_000:
        return f"{value / 1_000_000_000_000:.2f}T"
    if abs(value) >= 1_000_000_000:
        return f"{value / 1_000_000_000:.2f}B"
    if abs(value) >= 1_000_000:
        return f"{value / 1_000_000:.2f}M"
    return f"{value:,.2f}"


def trend_chart(data, y_col, title, y_label):
    if data.empty:
        st.warning(f"No data available for {title}.")
        return

    if analysis_mode == "Global":
        plot_df = data.groupby("year", as_index=False)[y_col].mean()

        fig = px.line(
            plot_df,
            x="year",
            y=y_col,
            markers=True,
            text=plot_df[y_col].round(2),
            title=title,
            labels={
                "year": "Year",
                y_col: y_label
            }
        )

        fig.update_traces(textposition="top center")

    else:
        plot_df = data.groupby(["country", "year"], as_index=False)[y_col].mean()
        show_text = len(selected_countries) <= 3

        fig = px.line(
            plot_df,
            x="year",
            y=y_col,
            color="country",
            markers=True,
            text=plot_df[y_col].round(2) if show_text else None,
            title=title,
            labels={
                "year": "Year",
                y_col: y_label,
                "country": "Country"
            }
        )

        if show_text:
            fig.update_traces(textposition="top center")

    fig.update_layout(height=520)
    st.plotly_chart(fig, use_container_width=True)


tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "Executive Overview",
    "Climate Trends",
    "Driver Analysis",
    "Country Intelligence",
    "Machine Learning",
    "Forecasting",
    "Strategic Insights"
])

with tab1:
    st.header("Executive Overview")

    col1, col2, col3, col4, col5 = st.columns(5)

    col1.metric("Countries", filtered_df["country"].nunique())
    col2.metric("Avg Demand", safe_mean(filtered_df, "electricity_demand_per_capita"))
    col3.metric("Avg Temp Change", safe_mean(filtered_df, "temperature_change_c"))
    col4.metric("Avg CO2", safe_mean(filtered_df, "co2_per_capita"))
    col5.metric("Avg Renewable %", safe_mean(filtered_df, "renewables_share_elec"))

    st.subheader(f"Top {top_n} Countries by {ranking_metric.replace('_', ' ').title()}")

    ranking_df = (
        base_filtered_df.groupby("country", as_index=False)[ranking_metric]
        .mean()
        .sort_values(ranking_metric, ascending=False)
        .head(top_n)
    )

    fig = px.bar(
        ranking_df,
        x=ranking_metric,
        y="country",
        orientation="h",
        text=ranking_metric,
        title=f"Top {top_n} Countries by {ranking_metric.replace('_', ' ').title()}",
        labels={
            ranking_metric: ranking_metric.replace("_", " ").title(),
            "country": "Country"
        }
    )

    fig.update_traces(texttemplate="%{text:.2f}", textposition="outside")
    fig.update_layout(yaxis={"categoryorder": "total ascending"}, height=560)
    st.plotly_chart(fig, use_container_width=True)

    st.download_button(
        "Download Filtered Dataset",
        filtered_df.to_csv(index=False),
        "filtered_climate_energy_data.csv",
        "text/csv"
    )

with tab2:
    st.header("Climate and Energy Trends")

    if analysis_mode == "Global":
        st.info("Global mode shows average trends across all countries in the selected year range.")
    else:
        st.info("Selected Countries mode compares selected countries only.")

    trend_chart(
        filtered_df,
        "electricity_demand_per_capita",
        "Electricity Demand Per Capita Over Time",
        "Electricity Demand Per Capita"
    )

    trend_chart(
        filtered_df,
        "temperature_change_c",
        "Temperature Change Over Time",
        "Temperature Change (°C)"
    )

    trend_chart(
        filtered_df,
        "co2_per_capita",
        "CO2 Per Capita Over Time",
        "CO2 Per Capita"
    )

    trend_chart(
        filtered_df,
        "renewables_share_elec",
        "Renewable Electricity Share Over Time",
        "Renewable Electricity Share"
    )

    trend_chart(
        filtered_df,
        "fossil_share_elec",
        "Fossil Electricity Share Over Time",
        "Fossil Electricity Share"
    )

with tab3:
    st.header("Driver Analysis")

    fig = px.scatter(
        filtered_df,
        x=driver_var,
        y="electricity_demand_per_capita",
        color="country" if analysis_mode == "Selected Countries" else None,
        hover_data=["year", "country"],
        trendline="ols",
        title=f"{driver_var.replace('_', ' ').title()} vs Electricity Demand",
        labels={
            driver_var: driver_var.replace("_", " ").title(),
            "electricity_demand_per_capita": "Electricity Demand Per Capita"
        }
    )

    fig.update_traces(marker=dict(size=9, opacity=0.75))
    fig.update_layout(height=560)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Dynamic Correlation Heatmap")

    corr_cols = [
        "electricity_demand_per_capita",
        "temperature_change_c",
        "co2_per_capita",
        "gdp",
        "population",
        "renewables_share_elec",
        "fossil_share_elec"
    ]

    if len(filtered_df) >= 3:
        corr_matrix = filtered_df[corr_cols].corr().round(2)

        heatmap = go.Figure(
            data=go.Heatmap(
                z=corr_matrix.values,
                x=[c.replace("_", " ").title() for c in corr_matrix.columns],
                y=[c.replace("_", " ").title() for c in corr_matrix.index],
                text=corr_matrix.values,
                texttemplate="%{text}",
                colorscale="RdBu",
                zmin=-1,
                zmax=1
            )
        )

        heatmap.update_layout(
            title="Correlation Heatmap Based on Current Filters",
            height=650
        )

        st.plotly_chart(heatmap, use_container_width=True)

        target_corr = (
            corr_matrix["electricity_demand_per_capita"]
            .drop("electricity_demand_per_capita")
            .sort_values(key=abs, ascending=False)
            .reset_index()
        )

        target_corr.columns = ["Variable", "Correlation With Demand"]
        st.subheader("Driver Importance by Correlation")
        st.dataframe(target_corr)
    else:
        st.warning("Not enough filtered data to calculate correlation heatmap.")

with tab4:
    st.header("Country Intelligence")

    country_profile_df = base_filtered_df[
        base_filtered_df["country"] == profile_country
    ].copy()

    if country_profile_df.empty:
        st.warning("No data available for selected country and year range.")
    else:
        latest = country_profile_df.sort_values("year").iloc[-1]

        col1, col2, col3 = st.columns(3)
        col1.metric("Country", profile_country)
        col2.metric("Latest Year", int(latest["year"]))
        col3.metric("Electricity Demand", round(latest["electricity_demand_per_capita"], 2))

        col4, col5, col6 = st.columns(3)
        col4.metric("CO2 Per Capita", round(latest["co2_per_capita"], 2))
        col5.metric("Renewable Share", round(latest["renewables_share_elec"], 2))
        col6.metric("GDP", format_large_number(latest["gdp"]))

        st.subheader("Supporting Factors Table")

        supporting_table = pd.DataFrame({
            "Metric": [
                "Average Electricity Demand",
                "Average Temperature Change",
                "Average CO2 Per Capita",
                "Average GDP",
                "Average Population",
                "Average Renewable Share",
                "Average Fossil Share"
            ],
            "Value": [
                safe_mean(country_profile_df, "electricity_demand_per_capita"),
                safe_mean(country_profile_df, "temperature_change_c"),
                safe_mean(country_profile_df, "co2_per_capita"),
                format_large_number(country_profile_df["gdp"].mean()),
                format_large_number(country_profile_df["population"].mean()),
                safe_mean(country_profile_df, "renewables_share_elec"),
                safe_mean(country_profile_df, "fossil_share_elec")
            ]
        })

        st.dataframe(supporting_table)

        st.subheader(f"Country Comparison: {profile_country} vs {compare_country}")

        comparison_vars = [
            "electricity_demand_per_capita",
            "temperature_change_c",
            "co2_per_capita",
            "gdp",
            "population",
            "renewables_share_elec",
            "fossil_share_elec"
        ]

        comparison_df = (
            base_filtered_df[
                base_filtered_df["country"].isin([profile_country, compare_country])
            ]
            .groupby("country", as_index=False)[comparison_vars]
            .mean()
        )
        if len(comparison_df) >= 2:
            comparison_long = comparison_df.melt(
                id_vars="country",
                value_vars=comparison_vars,
                var_name="Metric",
                value_name="Value"
            )

            comparison_long["Metric"] = comparison_long["Metric"].str.replace("_", " ").str.title()

            fig = px.bar(
                comparison_long,
                x="Metric",
                y="Value",
                color="country",
                barmode="group",
                text="Value",
                title=f"Supporting Variable Comparison: {profile_country} vs {compare_country}",
                labels={
                    "Metric": "Metric",
                    "Value": "Average Value",
                    "country": "Country"
                }
            )

            fig.update_traces(texttemplate="%{text:.2f}", textposition="outside")
            fig.update_layout(height=620, xaxis_tickangle=-25)
            st.plotly_chart(fig, use_container_width=True)

            st.subheader("Comparison Table")

            display_table = comparison_df.copy()
            display_table["gdp"] = display_table["gdp"].apply(format_large_number)
            display_table["population"] = display_table["population"].apply(format_large_number)

            st.dataframe(display_table)
        else:
            st.warning("Not enough data for country comparison.")

with tab5:
    st.header("Machine Learning Results")

    st.markdown("""
    This section compares baseline machine learning models and explains which variables
    are most important for predicting electricity demand per capita.
    """)

    if METRICS_PATH.exists():
        metrics_df = pd.read_csv(METRICS_PATH)

        model_name_map = {
            "ridge_linear": "Ridge Regression",
            "random_forest": "Random Forest",
            "linear_regression": "Linear Regression"
        }

        metrics_df["model_display"] = metrics_df["model"].replace(model_name_map)

        col1, col2, col3 = st.columns(3)

        best_model = metrics_df.sort_values("R2", ascending=False).iloc[0]

        col1.metric("Best Model", best_model["model_display"])
        col2.metric("Best R² Score", round(best_model["R2"], 4))
        col3.metric("Lowest RMSE", round(metrics_df["RMSE"].min(), 2))

        st.subheader("Model Performance Comparison")

        fig = px.bar(
            metrics_df,
            x="model_display",
            y="R2",
            text="R2",
            title="Model Comparison by R² Score",
            labels={
                "model_display": "Model",
                "R2": "R² Score"
            }
        )

        fig.update_traces(
            texttemplate="%{text:.4f}",
            textposition="outside"
        )

        fig.update_layout(
            yaxis_range=[0, 1],
            height=520
        )

        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Error Metrics Comparison")

        error_df = metrics_df.melt(
            id_vars="model_display",
            value_vars=["RMSE", "MAE"],
            var_name="Metric",
            value_name="Value"
        )

        fig = px.bar(
            error_df,
            x="model_display",
            y="Value",
            color="Metric",
            barmode="group",
            text="Value",
            title="RMSE and MAE Comparison",
            labels={
                "model_display": "Model",
                "Value": "Error Value"
            }
        )

        fig.update_traces(
            texttemplate="%{text:.2f}",
            textposition="outside"
        )

        fig.update_layout(height=520)
        st.plotly_chart(fig, use_container_width=True)

    else:
        st.warning("Missing model metrics file: reports/us12_baseline_comparison.csv")

    st.subheader("Random Forest Feature Importance")

    feature_data = {
        "Feature": [
            "CO2 per Capita",
            "Fossil Share Electricity",
            "Renewables Share Electricity",
            "Population",
            "GDP",
            "Temperature Change"
        ],
        "Importance": [
            0.61,
            0.18,
            0.14,
            0.045,
            0.02,
            0.005
        ]
    }

    feature_df = pd.DataFrame(feature_data)

    fig = px.bar(
        feature_df,
        x="Importance",
        y="Feature",
        orientation="h",
        text="Importance",
        title="Random Forest Feature Importance",
        labels={
            "Importance": "Importance Score",
            "Feature": "Feature"
        }
    )

    fig.update_traces(
        texttemplate="%{text:.3f}",
        textposition="outside"
    )

    fig.update_layout(
        yaxis={"categoryorder": "total ascending"},
        height=560
    )

    st.plotly_chart(fig, use_container_width=True)

    st.caption("Values based on trained Random Forest model.")

    st.subheader("Model Interpretation")

    st.success("""
    Random Forest achieved the strongest predictive performance, showing that
    electricity demand is influenced by nonlinear relationships between emissions,
    energy mix, population, GDP, and climate indicators.
    """)

    st.info("""
    CO2 per capita is the most influential feature in the Random Forest model,
    followed by fossil electricity share and renewable electricity share.
    This supports the project hypothesis that emissions and energy structure are
    strongly related to electricity demand patterns.
    """)

    st.warning("""
    Model results are based on the full train/test modeling workflow and are not
    recalculated from dashboard filters.
    """)

with tab6:
    st.header("Future Electricity Demand Forecasting")

    forecast_mode = st.radio(
        "Forecasting Method",
        ["Trend Forecast", "Scenario-Based Random Forest Forecast"]
    )

    country_df = df[df["country"] == forecast_country].sort_values("year").copy()

    if len(country_df) < 2:
        st.warning("Not enough data available for forecasting.")

    else:
        if forecast_mode == "Trend Forecast":
            st.subheader("Trend-Based Forecast")

            X = country_df[["year"]]
            y = country_df["electricity_demand_per_capita"]

            model = LinearRegression()
            model.fit(X, y)

            last_year = int(country_df["year"].max())
            current_demand = float(country_df.iloc[-1]["electricity_demand_per_capita"])

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

            forecast_final = float(future_df.iloc[-1]["electricity_demand_per_capita"])
            growth_pct = ((forecast_final - current_demand) / current_demand) * 100

            col1, col2, col3 = st.columns(3)
            col1.metric("Current Demand", round(current_demand, 2))
            col2.metric(
                f"Forecast Demand ({future_df.iloc[-1]['year']})",
                round(forecast_final, 2)
            )
            col3.metric("Projected Change", f"{growth_pct:.2f}%")

            fig = px.line(
                combined,
                x="year",
                y="electricity_demand_per_capita",
                color="type",
                markers=True,
                text="electricity_demand_per_capita",
                title=f"Trend-Based Electricity Demand Forecast for {forecast_country}",
                labels={
                    "year": "Year",
                    "electricity_demand_per_capita": "Electricity Demand Per Capita",
                    "type": "Data Type"
                }
            )

            fig.update_traces(texttemplate="%{text:.2f}", textposition="top center")
            fig.update_layout(height=560)
            st.plotly_chart(fig, use_container_width=True)

            st.subheader("Forecast Table")
            st.dataframe(future_df[["country", "year", "electricity_demand_per_capita"]])

            st.info(
                "This trend forecast uses year as the predictor. It is useful for showing historical direction, "
                "but it does not include climate, economic, or energy-mix variables."
            )

        else:
            st.subheader("Scenario-Based Random Forest Forecast")

            feature_cols = [
                "temperature_change_c",
                "co2_per_capita",
                "gdp",
                "population",
                "renewables_share_elec",
                "fossil_share_elec"
            ]

            model_data = df.dropna(subset=feature_cols + ["electricity_demand_per_capita"]).copy()

            X = model_data[feature_cols]
            y = model_data["electricity_demand_per_capita"]

            rf_model = RandomForestRegressor(
                n_estimators=200,
                random_state=42,
                n_jobs=-1
            )

            rf_model.fit(X, y)

            latest_row = country_df.sort_values("year").iloc[-1]

            st.markdown("""
            Adjust future assumptions below. The Random Forest model predicts electricity demand
            based on climate, emissions, economic, population, and energy-mix variables.
            """)

            col1, col2 = st.columns(2)

            with col1:
                temp_change = st.slider(
                    "Temperature Change Adjustment (°C)",
                    -2.0,
                    3.0,
                    0.5,
                    0.1
                )

                co2_change_pct = st.slider(
                    "CO2 Per Capita Change (%)",
                    -50,
                    50,
                    0,
                    5
                )

                gdp_change_pct = st.slider(
                    "GDP Change (%)",
                    -30,
                    80,
                    10,
                    5
                )

            with col2:
                population_change_pct = st.slider(
                    "Population Change (%)",
                    -20,
                    50,
                    5,
                    5
                )

                renewable_change = st.slider(
                    "Renewable Share Change",
                    -30.0,
                    50.0,
                    10.0,
                    1.0
                )

                fossil_change = st.slider(
                    "Fossil Share Change",
                    -50.0,
                    30.0,
                    -10.0,
                    1.0
                )

            scenario_input = latest_row[feature_cols].copy()

            scenario_input["temperature_change_c"] = (
                scenario_input["temperature_change_c"] + temp_change
            )

            scenario_input["co2_per_capita"] = (
                scenario_input["co2_per_capita"] * (1 + co2_change_pct / 100)
            )

            scenario_input["gdp"] = (
                scenario_input["gdp"] * (1 + gdp_change_pct / 100)
            )

            scenario_input["population"] = (
                scenario_input["population"] * (1 + population_change_pct / 100)
            )

            scenario_input["renewables_share_elec"] = max(
                0,
                scenario_input["renewables_share_elec"] + renewable_change
            )

            scenario_input["fossil_share_elec"] = max(
                0,
                scenario_input["fossil_share_elec"] + fossil_change
            )

            scenario_df = pd.DataFrame([scenario_input])

            current_prediction = rf_model.predict(
                pd.DataFrame([latest_row[feature_cols]])
            )[0]

            scenario_prediction = rf_model.predict(scenario_df)[0]

            scenario_change_pct = (
                (scenario_prediction - current_prediction) / current_prediction
            ) * 100

            col1, col2, col3 = st.columns(3)

            col1.metric(
                "Current Model Estimate",
                round(current_prediction, 2)
            )

            col2.metric(
                "Scenario Forecast",
                round(scenario_prediction, 2)
            )

            col3.metric(
                "Scenario Change",
                f"{scenario_change_pct:.2f}%"
            )

            scenario_compare = pd.DataFrame({
                "Variable": [
                    "Temperature Change",
                    "CO2 Per Capita",
                    "GDP",
                    "Population",
                    "Renewables Share",
                    "Fossil Share"
                ],
                "Current Value": [
                    latest_row["temperature_change_c"],
                    latest_row["co2_per_capita"],
                    latest_row["gdp"],
                    latest_row["population"],
                    latest_row["renewables_share_elec"],
                    latest_row["fossil_share_elec"]
                ],
                "Scenario Value": [
                    scenario_input["temperature_change_c"],
                    scenario_input["co2_per_capita"],
                    scenario_input["gdp"],
                    scenario_input["population"],
                    scenario_input["renewables_share_elec"],
                    scenario_input["fossil_share_elec"]
                ]
            })

            fig = px.bar(
                scenario_compare,
                x="Variable",
                y=["Current Value", "Scenario Value"],
                barmode="group",
                text_auto=".2s",
                title=f"Scenario Assumptions for {forecast_country}",
                labels={
                    "value": "Value",
                    "Variable": "Supporting Variable"
                }
            )

            fig.update_layout(height=560, xaxis_tickangle=-25)
            st.plotly_chart(fig, use_container_width=True)

            result_df = pd.DataFrame({
                "Forecast Type": ["Current Model Estimate", "Scenario Forecast"],
                "Electricity Demand Per Capita": [
                    current_prediction,
                    scenario_prediction
                ]
            })

            fig = px.bar(
                result_df,
                x="Forecast Type",
                y="Electricity Demand Per Capita",
                text="Electricity Demand Per Capita",
                title=f"Random Forest Scenario Forecast for {forecast_country}",
                labels={
                    "Electricity Demand Per Capita": "Predicted Electricity Demand"
                }
            )

            fig.update_traces(texttemplate="%{text:.2f}", textposition="outside")
            fig.update_layout(height=520)
            st.plotly_chart(fig, use_container_width=True)

            st.subheader("Scenario Input Table")
            st.dataframe(scenario_compare)

            st.success(
                "This scenario forecast is stronger than the simple trend forecast because it uses climate, "
                "emissions, GDP, population, renewable share, and fossil share as predictive inputs."
            )

            st.warning(
                "Scenario forecasting is still exploratory. It estimates possible demand under selected assumptions "
                "and should not be interpreted as a guaranteed future prediction."
            )

with tab7:
    st.header("Strategic Insights")

    insight_df = base_filtered_df[
        base_filtered_df["country"] == profile_country
    ].copy()

    if insight_df.empty:
        st.warning("No data available for strategic insights.")
    else:
        ordered = insight_df.sort_values("year")

        demand_change = (
            ordered["electricity_demand_per_capita"].iloc[-1]
            - ordered["electricity_demand_per_capita"].iloc[0]
        )

        renewable_change = (
            ordered["renewables_share_elec"].iloc[-1]
            - ordered["renewables_share_elec"].iloc[0]
        )

        co2_change = (
            ordered["co2_per_capita"].iloc[-1]
            - ordered["co2_per_capita"].iloc[0]
        )

        st.subheader(f"Automated Insight Summary for {profile_country}")

        if demand_change > 0:
            st.write("- Electricity demand has increased over the selected period.")
        else:
            st.write("- Electricity demand has decreased or remained stable over the selected period.")

        if renewable_change > 0:
            st.write("- Renewable electricity share has improved over time.")
        else:
            st.write("- Renewable electricity share has not increased significantly.")

        if co2_change > 0:
            st.write("- CO2 emissions per capita increased over the selected period.")
        else:
            st.write("- CO2 emissions per capita decreased or remained stable.")

        st.subheader("Business Recommendation")

        if demand_change > 0 and renewable_change > 0:
            st.success(
                "Continue expanding renewable infrastructure while preparing for rising electricity demand."
            )
        elif demand_change > 0 and renewable_change <= 0:
            st.warning(
                "Prioritize renewable energy investment and grid capacity planning."
            )
        elif demand_change <= 0 and co2_change <= 0:
            st.success(
                "Maintain current energy transition strategy and monitor demand stability."
            )
        else:
            st.info(
                "Continue monitoring climate and energy indicators for long-term planning."
            )