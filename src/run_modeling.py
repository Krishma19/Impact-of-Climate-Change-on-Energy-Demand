"""Sprint 5 modeling pipeline — same logic as notebooks/modeling/01–05."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_PROCESSED = REPO_ROOT / "data" / "processed"
REPORTS = REPO_ROOT / "reports"
FIGURES = REPORTS / "figures"

TARGET = "electricity_demand_per_capita"
FEATURES = [
    "temperature_change_c",
    "co2_per_capita",
    "gdp",
    "population",
    "renewables_share_elec",
    "fossil_share_elec",
]
ID_COLS = ["country", "year"]
TRAIN_YEAR_MAX = 2018
TEST_YEAR_MIN = 2019


def prep_modeling_data() -> pd.DataFrame:
    data_path = DATA_PROCESSED / "final_merged_balanced_adjusted.csv"
    out_path = DATA_PROCESSED / "modeling_ready.csv"
    keep_cols = ID_COLS + [TARGET] + FEATURES

    df = pd.read_csv(data_path)
    model_df = df[keep_cols].copy()

    if model_df[keep_cols].isnull().any().any():
        raise ValueError("Modeling columns contain null values.")
    if model_df.duplicated(subset=ID_COLS).any():
        raise ValueError("Duplicate country-year rows in modeling frame.")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    model_df.to_csv(out_path, index=False)
    print(f"modeling_ready.csv saved: {model_df.shape}")
    return model_df


def split_train_test(model_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = model_df[model_df["year"] <= TRAIN_YEAR_MAX].copy()
    test = model_df[model_df["year"] >= TEST_YEAR_MIN].copy()

    for name, frame in ("train", train), ("test", test):
        dupes = frame.duplicated(subset=ID_COLS).sum()
        if dupes:
            raise ValueError(f"{name} set has {dupes} duplicate country-year rows.")

    train_path = DATA_PROCESSED / "train.csv"
    test_path = DATA_PROCESSED / "test.csv"
    train.to_csv(train_path, index=False)
    test.to_csv(test_path, index=False)
    print(f"train.csv {train.shape} | test.csv {test.shape}")
    return train, test


def train_ridge(train: pd.DataFrame, test: pd.DataFrame) -> dict[str, float]:
    X_train = train[FEATURES]
    y_train = train[TARGET]
    X_test = test[FEATURES]
    y_test = test[TARGET]

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    model = Ridge(alpha=1.0)
    model.fit(X_train_s, y_train)
    y_pred = model.predict(X_test_s)

    metrics = _test_metrics(y_test, y_pred)
    _save_metrics(REPORTS / "us10_linear_metrics.csv", "ridge_linear", metrics)
    print(f"Ridge test R²: {metrics['R2']:.4f}")
    return metrics


def train_random_forest(train: pd.DataFrame, test: pd.DataFrame) -> dict[str, float]:
    X_train = train[FEATURES]
    y_train = train[TARGET]
    X_test = test[FEATURES]
    y_test = test[TARGET]

    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    metrics = _test_metrics(y_test, y_pred)
    _save_metrics(REPORTS / "us11_tree_metrics.csv", "random_forest", metrics)
    _save_feature_importance(model)
    print(f"Random Forest test R²: {metrics['R2']:.4f}")
    return metrics


def compare_models() -> pd.DataFrame:
    linear_path = REPORTS / "us10_linear_metrics.csv"
    tree_path = REPORTS / "us11_tree_metrics.csv"
    comparison_path = REPORTS / "us12_baseline_comparison.csv"
    fig_path = FIGURES / "us12_metrics_comparison.png"
    summary_path = REPORTS / "us12_model_comparison_summary.txt"

    comparison = pd.concat(
        [pd.read_csv(linear_path), pd.read_csv(tree_path)],
        ignore_index=True,
    )[["model", "RMSE", "MAE", "R2"]]

    _save_comparison_chart(comparison, fig_path)
    comparison.to_csv(comparison_path, index=False)

    ridge = comparison.loc[comparison["model"] == "ridge_linear"].iloc[0]
    rf = comparison.loc[comparison["model"] == "random_forest"].iloc[0]
    summary = "\n".join([
        "US 12 — BASELINE MODEL COMPARISON SUMMARY",
        "=" * 60,
        "",
        "Setup:",
        "- Same train/test split as US 9 (train 2001–2018, test 2019–2022)",
        "- Target: electricity_demand_per_capita",
        "- Six features per docs/modeling_spec.md",
        "",
        "Test metrics:",
        f"- Ridge (US 10): RMSE={ridge['RMSE']:.2f}, MAE={ridge['MAE']:.2f}, R²={ridge['R2']:.4f}",
        f"- Random Forest (US 11): RMSE={rf['RMSE']:.2f}, MAE={rf['MAE']:.2f}, R²={rf['R2']:.4f}",
        "",
        "Findings:",
        "1. Random Forest outperforms Ridge on all three test metrics.",
        "2. Ridge is a simpler linear baseline; tree model captures non-linear patterns better.",
        "3. US 11 feature importance: co2_per_capita and electricity mix shares ranked highest.",
        "4. temperature_change_c had low importance in the pooled Random Forest model.",
        "",
        "Caveat:",
        "- Results are predictive/associative, not causal (see docs/modeling_spec.md).",
        "- Pooled global models do not control for country fixed effects.",
        "",
    ])
    summary_path.write_text(summary, encoding="utf-8")
    print(f"Comparison saved: {comparison_path.name}, {fig_path.name}, {summary_path.name}")
    return comparison


def run_modeling_pipeline() -> None:
    print("\n" + "=" * 60)
    print("MODELING PIPELINE (US 8–12)")
    print("=" * 60)

    model_df = prep_modeling_data()
    train, test = split_train_test(model_df)
    train_ridge(train, test)
    train_random_forest(train, test)
    compare_models()

    print("\nModeling pipeline complete.")


def _test_metrics(y_test: pd.Series, y_pred) -> dict[str, float]:
    mse = mean_squared_error(y_test, y_pred)
    return {
        "RMSE": mse ** 0.5,
        "MAE": mean_absolute_error(y_test, y_pred),
        "R2": r2_score(y_test, y_pred),
    }


def _save_metrics(path: Path, model_name: str, metrics: dict[str, float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{"model": model_name, **metrics}]).to_csv(path, index=False)


def _save_feature_importance(model: RandomForestRegressor) -> None:
    fig_path = FIGURES / "us11_feature_importance.png"
    fig_path.parent.mkdir(parents=True, exist_ok=True)

    imp = pd.Series(model.feature_importances_, index=FEATURES).sort_values()
    fig, ax = plt.subplots(figsize=(8, 5))
    imp.plot(kind="barh", ax=ax, color="steelblue")
    ax.set_title("Random Forest — feature importance (US 11)")
    ax.set_xlabel("Importance")
    plt.tight_layout()
    fig.savefig(fig_path, dpi=120)
    plt.close(fig)


def _save_comparison_chart(comparison: pd.DataFrame, fig_path: Path) -> None:
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    labels = {
        "ridge_linear": "Ridge (US 10)",
        "random_forest": "Random Forest (US 11)",
    }
    plot_df = comparison.set_index("model")[["RMSE", "MAE", "R2"]]
    plot_df.index = [labels.get(m, m) for m in plot_df.index]

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    colors = ["#4C72B0", "#55A868"]
    for ax, metric in zip(axes, ["RMSE", "MAE", "R2"]):
        plot_df[metric].plot(kind="bar", ax=ax, color=colors, rot=0)
        ax.set_title(metric)
        ax.set_xlabel("")
        ax.set_ylabel(metric)

    fig.suptitle("Baseline model comparison — test set (US 12)", y=1.02)
    plt.tight_layout()
    fig.savefig(fig_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    run_modeling_pipeline()
