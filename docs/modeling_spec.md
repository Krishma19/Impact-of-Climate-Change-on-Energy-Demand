# Modeling specification — Sprint 5

**Project:** Impact of Climate Change on Energy Demand  
**Sprint:** Modeling & baseline predictions  
**User story:** US 7 — Modeling specification  
**Status:** Draft — both teammates should confirm before US 8+

---

## 1. Primary dataset

| Item | Value |
|------|--------|
| **File** | `data/processed/final_merged_balanced_adjusted.csv` |
| **Panel keys** | `country`, `year` |
| **Rows × columns** | 3,300 × 22 (verified May 2026) |
| **Countries** | 150 |
| **Year range** | 2001–2022 (22 years per country, strict balanced panel) |
| **Notes** | No interpolation; all modeling columns 100% non-null on this file |

**Robustness (optional, US 13):** `data/processed/final_merged_partial_balanced.csv`

**Not for primary modeling:** `data/processed/final_merged.csv` (unbalanced; EDA only)

---

## 2. Target variable

| Role | Column |
|------|--------|
| **Target (y)** | `electricity_demand_per_capita` |

**Interpretation:** Per-capita electricity demand (OWID energy merge).  
**Units:** As in source CSV (do not rescale without documenting in notebooks).

---

## 3. Feature variables (v1)

| Feature | Column | Role |
|---------|--------|------|
| Climate | `temperature_change_c` | Primary climate signal (FAOSTAT) |
| Emissions | `co2_per_capita` | Economic/emissions context |
| Economy | `gdp` | Scale / development |
| Demographics | `population` | Scale |
| Energy mix | `renewables_share_elec` | Supply structure |
| Energy mix | `fossil_share_elec` | Supply structure |

**Identifiers (not features in baseline v1):** `country`, `year`, `iso_code`

---

## 4. Evaluation metrics

Report on **held-out test** data only:

| Metric | Use |
|--------|-----|
| **RMSE** | Primary error scale |
| **MAE** | Robust to outliers |
| **R²** | Explained variance (interpret with caution on panel data) |

---

## 5. Train / test split (agreed for US 9)

**Method:** Time-based split (no random row shuffle).

| Set | Years | Purpose |
|-----|-------|---------|
| **Train** | 2001–2018 | Fit models and preprocessing |
| **Test** | 2019–2022 | Final metrics for sprint baselines |

**Rationale:** Mimics forecasting on recent years; avoids treating future years as known during training.

**Alternative (if team changes):** Country holdout — document here before US 9 implementation.

---

## 6. Modeling scope and caveats

- **In scope (Sprint 5):** Baseline linear/Ridge and tree model (e.g. Random Forest); same split for both.
- **Out of scope:** Causal claims, country fixed effects, hyperparameter tuning, production deployment.
- **Caveat:** Pooled correlations and global models do **not** control for unobserved country heterogeneity; phrase results as **associative / predictive**, not causal.

---

## 7. Reproducibility

- Run notebooks/scripts from **repository root**.
- Branch: `feature/modeling-baseline`
- Python deps: `requirements.txt` (add `scikit-learn` when US 10+ starts)

---

## Sign-off

| Teammate | Confirmed (date) |
|----------|------------------|
| | |
| | |

---

*After both sign off, mark US 7 Done in Taiga and start US 8 / US 9.*
