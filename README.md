# Climate Change Impact on Energy Demand

## Project Overview

This capstone project analyzes the relationship between climate change indicators and electricity demand across approximately 150 countries using yearly data from 2001 to 2022.

The project investigates how climate-related variables such as temperature change, CO₂ emissions, renewable energy usage, fossil fuel electricity share, GDP, and population influence electricity demand patterns over time.

The project combines:
* Exploratory Data Analysis (EDA)
* Statistical Analysis
* Predictive Modeling
* Machine Learning
* Interactive Dashboard Visualization
* Forecasting

---

# Research Questions

1. How does climate change influence electricity demand across countries between 2001 and 2022?
2. What is the relationship between temperature change, CO₂ emissions, and electricity demand per capita?
3. Can climate and economic indicators be used to predict future electricity demand patterns?

---

# Hypotheses

* H1: There is a positive relationship between temperature change and electricity demand per capita.
* H2: Countries with higher CO₂ emissions per capita tend to have higher electricity demand per capita.
* H3: Renewable energy usage and GDP significantly influence electricity demand patterns across countries.

---

# Data Sources
The project uses publicly available datasets from:
* Our World in Data (OWID)
* FAOSTAT
* World Bank

---

# Final Dataset Variables

| Variable                      | Description                   |
| ----------------------------- | ----------------------------- |
| country                       | Country name                  |
| year                          | Observation year              |
| electricity_demand_per_capita | Electricity demand per person |
| temperature_change_c          | Annual temperature change     |
| co2_per_capita                | CO₂ emissions per person      |
| gdp                           | Gross Domestic Product        |
| renewables_share_elec         | Renewable electricity share   |
| fossil_share_elec             | Fossil fuel electricity share |
| population                    | Country population            |

---

# Project Structure

```text
Impact-of-Climate-Change-on-Energy-Demand/
│
├── dashboard/
│   └── app.py
│
├── data/
│   ├── raw/
│   └── processed/
│
├── notebooks/
│   ├── eda/
│   ├── modeling/
│   └── visualization/
│
├── reports/
│   ├── figures/
│   └── *.csv
│
├── src/
│   ├── run_modeling.py
│   └── run_visualizations.py
│
├── requirements.txt
└── README.md
```

---

# Analytical Workflow

## 1. Data Collection and Preparation

* Collected multi-country climate and energy datasets
* Cleaned and merged country-year data
* Created modeling-ready dataset

## 2. Exploratory Data Analysis (EDA)

* Trend analysis
* Correlation analysis
* Statistical summaries
* Cross-country comparisons

## 3. Statistical Analysis

* Pearson correlation
* Regression analysis
* Hypothesis testing

## 4. Predictive Modeling

The following machine learning models were implemented:

* Linear Regression
* Ridge Regression
* Random Forest Regression

## 5. Visualization and Dashboard

The project includes:

* Climate and energy visualization charts
* Correlation heatmap
* Model comparison charts
* Feature importance analysis
* Interactive Streamlit dashboard
* Forecasting section

---

# Dashboard Features

The Streamlit dashboard includes:

* Interactive country filters
* Year range filters
* KPI metrics
* Trend analysis charts
* Relationship analysis charts
* Correlation heatmap
* Model comparison section
* Feature importance visualization
* Future forecasting
* Downloadable filtered data

---

# Installation

## Clone Repository

```bash
git clone https://github.com/Krishma19/Impact-of-Climate-Change-on-Energy-Demand.git
cd Impact-of-Climate-Change-on-Energy-Demand
```

---

# Install Dependencies

```bash
pip install -r requirements.txt
```

---

# Run Modeling Pipeline

```bash
python src/run_modeling.py
```

---

# Run Visualization Pipeline

```bash
python src/run_visualizations.py
```

---

# Run Dashboard

```bash
streamlit run dashboard/app.py
```

---

# Key Findings

* Climate change indicators influence electricity demand patterns.
* Temperature changes and CO₂ emissions show relationships with electricity demand.
* Renewable energy and GDP contribute to electricity demand variations.
* Random Forest achieved the strongest predictive performance among baseline models.
* Forecasting analysis demonstrates future electricity demand trends using historical data.

---

# Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Plotly
* Streamlit
* Scikit-learn
* Statsmodels
* GitHub
* Taiga

---

# Future Improvements

Potential future enhancements include:

* Advanced time-series forecasting models
* Regional climate segmentation
* Deep learning approaches
* Real-time energy data integration
* Deployment to cloud platforms

---

# Team

Master of Data Analytics Capstone Project  
University of Niagara Falls Canada  

## Team Members

- Krishma Patel-(https://github.com/Krishma19)
- Mohammed Shoaib-(https://github.com/mshoaib239)
- Ritu Patel-(https://github.com/RituPatel21)
- Sunny Garasiya-(https://github.com/SunnyGarasiya0001)
- Yuvraj Thakur-(https://github.com/yuvrajthakur2709-maker)
---

# License

This project is developed for academic and educational purposes.
