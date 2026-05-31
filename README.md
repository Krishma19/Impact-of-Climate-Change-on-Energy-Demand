# 🌍 Impact of Climate Change on Energy Demand

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Random%20Forest-green)
![Dashboard](https://img.shields.io/badge/Streamlit-Dashboard-red)
![Status](https://img.shields.io/badge/Project-Completed-success)
![Capstone](https://img.shields.io/badge/MDA-Capstone-orange)

---

## 📖 Project Overview

This capstone project analyzes the relationship between climate change indicators and electricity demand across approximately **192 countries** using yearly data from **2001–2022**.

In this study, **energy demand refers specifically to electricity demand per capita**, representing the average electricity consumption per person within a country.

The analysis investigates how climate-related and economic variables influence electricity demand patterns over time, including:

* 🌡️ Temperature Change
* 🌍 CO₂ Emissions
* ♻️ Renewable Energy Usage
* 🏭 Fossil Fuel Electricity Share
* 💰 Gross Domestic Product (GDP)
* 👥 Population

The project combines:

* 📊 Exploratory Data Analysis (EDA)
* 📈 Statistical Analysis
* 🧪 Hypothesis Testing
* 🤖 Machine Learning Modeling
* 🔮 Forecasting
* 🖥️ Interactive Dashboard Development

---

# 🔄 Project Workflow

Raw Climate & Energy Data
        ↓
Data Cleaning & Integration
        ↓
Exploratory Data Analysis (EDA)
        ↓
Hypothesis Testing
        ↓
Feature Engineering
        ↓
Machine Learning Models
        ↓
Visualization & Dashboard
        ↓
Forecasting & Insights

---

# 🎯 Research Questions

### RQ1

How does climate change influence electricity demand across countries between 2001 and 2022?

### RQ2

What is the relationship between temperature change, CO₂ emissions, and electricity demand per capita?

### RQ3

Can climate and economic indicators be used to predict future electricity demand patterns?

---

# 🧪 Hypotheses

### H1

There is a positive relationship between temperature change and electricity demand per capita.

### H0

There is no significant relationship between temperature change and electricity demand per capita.

### H2

Countries with higher CO₂ emissions per capita tend to have higher electricity demand per capita.

### H3

Renewable energy usage and GDP significantly influence electricity demand patterns across countries.

---

# 🌐 Data Sources

The project utilizes publicly available datasets from trusted international organizations and research platforms.

| Source                | Description                                                               |
| ------------------------------- | ------------------------------------------------------------------------- |
| 🌍 **Our World in Data (OWID)** | Electricity demand, CO₂ emissions, GDP, population, and energy indicators |
| 🌾 **FAOSTAT**                  | Annual temperature change and climate indicators                          |
| 🏦 **World Bank**               | Renewable energy and economic indicators                                  |

### Dataset Links

🔗 **Our World in Data (OWID)**
https://github.com/owid/energy-data

🔗 **FAOSTAT Climate Change Indicators**
https://www.fao.org/faostat/en/#data/ET

🔗 **World Bank Open Data**
https://data.worldbank.org/

These datasets were cleaned, validated, and integrated into a unified country-year analytical dataset covering approximately **192 countries from 2001–2022**.


---

# 📋 Final Dataset Variables

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

# 📂 Project Structure

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
│   ├── hypothesis_testing/
│   └── outputs/
│
├── src/
│   ├── run_modeling.py
│   ├── run_visualizations.py
│   └── verify_eda_outputs.py
│
├── requirements.txt
└── README.md
```

---

# 🔬 Analytical Methodology

## 1️⃣ Data Collection & Preparation

* Collected multi-country climate and energy datasets
* Cleaned and integrated country-year records
* Created modeling-ready dataset

## 2️⃣ Exploratory Data Analysis (EDA)

* Trend Analysis
* Correlation Analysis
* Statistical Summaries
* Cross-Country Comparisons

## 3️⃣ Statistical Analysis

* Pearson Correlation
* Multiple Regression Analysis
* Hypothesis Testing

## 4️⃣ Machine Learning Modeling

Implemented baseline predictive models:

* Linear Regression
* Ridge Regression
* Random Forest Regression

## 5️⃣ Visualization & Dashboard Development

- Climate Trend Visualizations
- Correlation Heatmaps
- Climate Impact Ranking Analysis
- Renewable Impact Ranking Analysis
- Model Comparison Charts
- Feature Importance Analysis
- Interactive Streamlit Dashboard
- Forecasting Analysis
---

# 🏆 Model Performance

| Model                    | Test R² Score |
| ------------------------ | ------------- |
| Ridge Regression         | 0.4599        |
| Random Forest Regression | 0.8335        |

✅ Random Forest achieved the strongest predictive performance and was selected as the best-performing baseline model.

✅ Random Forest outperformed Ridge Regression by a substantial margin, indicating that nonlinear relationships between climate and energy variables are important for electricity demand prediction.

---

## 📦 Project Deliverables

✅ Integrated Climate-Energy Dataset

✅ Exploratory Data Analysis (EDA)

✅ Hypothesis Testing

✅ Feature Engineering

✅ Ridge Regression Baseline Model

✅ Random Forest Baseline Model

✅ Model Comparison Analysis

✅ Climate Impact Ranking Analysis

✅ Renewable Energy Impact Ranking Analysis

✅ Visualization Suite

✅ Interactive Streamlit Dashboard

✅ Forecasting Analysis

✅ Final Capstone Report

---

# 🖥️ Dashboard Features

* 🌎 Interactive Country Filters
* 📅 Year Range Filters
* 📈 Climate Trend Analysis
* ⚡ Electricity Demand Trends
* 🔥 Relationship Analysis
* 📊 Correlation Heatmaps
* 🤖 Model Performance Comparison
* 🌲 Feature Importance Visualization
* 🔮 Forecasting Module
* 📥 Downloadable Filtered Data

---

# 📸 Dashboard Preview

### Executive Overview & Climate Trends

<p align="center">
  <img src="https://github.com/user-attachments/assets/ae148ce4-c7d8-4ad9-9fc8-f5bd25a5cd19" width="48%" />
  <img src="https://github.com/user-attachments/assets/091f0ccf-1331-49c1-9ee7-1a8b9514b82d" width="48%" />
</p>

### Model Performance & Forecasting Analysis

<p align="center">
  <img src="https://github.com/user-attachments/assets/f20ef7c8-0161-48a2-ab82-1ab40dd70eae" width="48%" />
  <img src="https://github.com/user-attachments/assets/14762729-1b52-443b-8571-0e3f65d56c72" width="48%" />
</p>

---

# ⚙️ Installation & Setup

## Clone Repository

```bash
git clone https://github.com/Krishma19/Impact-of-Climate-Change-on-Energy-Demand.git

cd Impact-of-Climate-Change-on-Energy-Demand
```

## Install Dependencies

```bash
pip install -r requirements.txt
```

## Run Modeling Pipeline

```bash
python src/run_modeling.py
```

## Run Visualization Pipeline

```bash
python src/run_visualizations.py
```

## Verify EDA Outputs

```bash
python src/verify_eda_outputs.py
```

## Launch Dashboard

```bash
streamlit run dashboard/app.py
```

---

# 💡 Key Findings

✅ Global electricity demand increased substantially between 2001 and 2022.

✅ Temperature change demonstrated measurable relationships with electricity demand across multiple countries.

✅ GDP and population emerged as major drivers of electricity demand.

✅ Several countries showed strong climate sensitivity, with temperature-demand correlations above 0.70.

✅ Renewable energy adoption aligned positively with electricity demand growth in many countries.

✅ Random Forest Regression achieved the highest predictive performance (R² = 0.8335).

✅ Machine learning models outperformed traditional linear approaches for electricity demand forecasting.

---

# 💼 Business Applications

- Long-term electricity demand forecasting
- Energy infrastructure and grid planning
- Climate adaptation strategy development
- Renewable energy investment planning
- Utility resource allocation
- Sustainability and carbon-reduction planning
- Climate-aware policy decision support

---

# 🛠️ Technologies Used

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

## 🔮 Future Research Directions

🌍 Expand analysis to include precipitation, humidity, drought severity, and extreme weather events.

📈 Apply advanced forecasting models such as ARIMA, Prophet, LSTM, and Transformer-based approaches.

🗺️ Conduct region-specific climate-energy analyses.

⚡ Incorporate real-time energy consumption and smart-grid data.

🤖 Explore deep learning and ensemble learning techniques.

🏛️ Investigate the impact of government energy policies and carbon pricing mechanisms.

🌱 Evaluate renewable energy transitions and electrification strategies.

☁️ Deploy the framework as a cloud-based decision-support system.

---


# 📚 References

- NASA Climate Data: https://data.giss.nasa.gov/gistemp/
- Our World in Data Energy Dataset: https://github.com/owid/energy-data
- FAOSTAT Climate Indicators: https://www.fao.org/faostat/en/#data/ET
- World Bank Open Data: https://data.worldbank.org/
- International Energy Agency (IEA): https://www.iea.org/

---

# 👥 Team Members

| Name            | GitHub                                    |
| --------------- | ----------------------------------------- |
| Krishma Patel   | https://github.com/Krishma19              |
| Mohammed Shoaib | https://github.com/mshoaib239             |
| Ritu Patel      | https://github.com/RituPatel21            |
| Sunny Garasiya  | https://github.com/SunnyGarasiya0001      |
| Yuvraj Thakur   | https://github.com/yuvrajthakur2709-maker |

---

# 🎓 Academic Information

**Master of Data Analytics (MDA)**
**University of Niagara Falls Canada**

**Capstone Project:** Analyzing the Impact of Climate Change on Energy Demand

---

# 📄 License

This project was developed for academic and educational purposes.

---

# ⭐ About

This project analyzes how climate change affects electricity demand by studying temperature change, CO₂ emissions, renewable energy usage, and economic indicators. Through data analytics, machine learning, forecasting, and dashboard visualization, the project provides actionable insights for sustainable energy planning and climate-aware decision-making.
