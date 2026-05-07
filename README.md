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
