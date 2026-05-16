# Impact-of-Climate-Change-on-Energy-Demand

This project analyzes how climate change relates to energy demand using CO₂, temperature, and electricity indicators. The **`feature/master-dataset-pipeline`** branch carries the reproducible dataset build.

## Repository layout

| Path | Purpose |
|------|---------|
| `data/raw/` | Source extracts (FAOSTAT temperature, OWID CO₂ & energy). |
| `data/processed/` | Merged and balanced-panel outputs from the pipeline. |
| `src/` | Executable Python pipeline scripts. |
| `notebooks/` | Notebook versions of the same pipeline. |
| `docs/` | Project planning and documentation. |
| `reports/` | Reserved for exported figures / write-ups. |
| `requirements.txt` | Python dependencies. |

## Running the pipeline

From the repository root:

```bash
pip install -r requirements.txt
python src/build_master_dataset.py
python src/build_balanced_alternatives.py
```

Artifacts are written to `data/processed/` (`final_merged.csv`, balanced alternatives).

## Exploratory Data Analysis and Validation

The project includes an initial exploratory data analysis (EDA) and validation phase to better understand the climate-energy dataset before moving into visualization and predictive modeling.

During this stage, the dataset structure, column types, and summary statistics were reviewed to understand the overall data distribution and identify important variables related to climate change and energy demand.

Several validation checks were also performed, including:

- Missing value analysis across all columns
- Duplicate country-year record validation
- Country and year consistency checks
- Numerical range and anomaly detection
- Data quality observation reporting

Key observations from the analysis:

- The dataset contains 4,994 rows and 22 columns.
- No duplicate country-year records were found.
- Country and year fields are complete and consistent.
- Missing values are mainly present in some CO₂ and GDP-related variables.
- Population and electricity demand fields contain mostly complete data.
- No major anomalies or invalid records were identified during validation.

These checks helped ensure that the dataset is reliable and suitable for further analysis, dashboard development, and machine learning tasks.
