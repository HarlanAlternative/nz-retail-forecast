"""Compare DL baselines against classical models and produce summary figure.

Loads metrics from both MLflow experiments and saves a ranked comparison
table + bar chart to models/dl_vs_classical_comparison.png.

Usage:
    python -m deep_learning.compare_dl_vs_classical
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import pandas as pd

from forecasting.config import get_project_root, load_config

logger = logging.getLogger(__name__)

_MODEL_ORDER = ["Prophet", "Ridge", "LightGBM", "GRU", "LSTM",
                "Holt-Winters", "SARIMA", "ElasticNet"]

_CLASSICAL_METRIC_KEYS = {
    "Prophet":     ("prophet_mape", "prophet_rmse", "prophet_da"),
    "Ridge":       ("ridge_mape",   "ridge_rmse",   "ridge_da"),
    "LightGBM":    ("lgbm_mape",    "lgbm_rmse",    "lgbm_da"),
    "Holt-Winters":("hw_mape",      "hw_rmse",      "hw_da"),
    "SARIMA":      ("sarima_mape",  "sarima_rmse",  "sarima_da"),
    "ElasticNet":  ("elasticnet_mape", "elasticnet_rmse", "elasticnet_da"),
}

_DL_METRIC_KEYS = {
    "LSTM": ("lstm_mape", "lstm_rmse", "lstm_da"),
    "GRU":  ("gru_mape",  "gru_rmse",  "gru_da"),
}


def _load_latest_run_metrics(experiment_name: str, tracking_uri: str) -> dict[str, float]:
    mlflow.set_tracking_uri(tracking_uri)
    client = mlflow.tracking.MlflowClient()
    exp = client.get_experiment_by_name(experiment_name)
    if exp is None:
        logger.warning("Experiment '%s' not found in MLflow.", experiment_name)
        return {}
    runs = client.search_runs(
        experiment_ids=[exp.experiment_id],
        order_by=["start_time DESC"],
        max_results=1,
    )
    if not runs:
        return {}
    return runs[0].data.metrics


def load_all_metrics(cfg: dict) -> pd.DataFrame:
    tracking_uri = cfg["mlflow"]["tracking_uri"]

    classical = _load_latest_run_metrics(cfg["mlflow"]["experiment_name"], tracking_uri)
    dl        = _load_latest_run_metrics(cfg["deep_learning"]["experiment_name"], tracking_uri)

    rows = []
    for model, (mape_k, rmse_k, da_k) in _CLASSICAL_METRIC_KEYS.items():
        if mape_k in classical:
            rows.append({
                "Model": model,
                "MAPE%": classical[mape_k],
                "RMSE":  classical[rmse_k],
                "DA%":   classical.get(da_k, float("nan")) * 100,
                "Type":  "Classical",
            })

    for model, (mape_k, rmse_k, da_k) in _DL_METRIC_KEYS.items():
        if mape_k in dl:
            rows.append({
                "Model": model,
                "MAPE%": dl[mape_k],
                "RMSE":  dl[rmse_k],
                "DA%":   dl.get(da_k, float("nan")) * 100,
                "Type":  "Deep Learning",
            })

    df = pd.DataFrame(rows)
    df["_order"] = df["Model"].map(
        {m: i for i, m in enumerate(_MODEL_ORDER)}
    ).fillna(99)
    return df.sort_values("MAPE%").drop(columns="_order").reset_index(drop=True)


def plot_comparison(df: pd.DataFrame, save_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    colors = ["#2196F3" if t == "Classical" else "#FF5722" for t in df["Type"]]

    # MAPE bar chart
    ax = axes[0]
    bars = ax.barh(df["Model"], df["MAPE%"], color=colors)
    ax.set_xlabel("MAPE (%)")
    ax.set_title("Model Comparison — MAPE (lower is better)")
    ax.invert_yaxis()
    for bar, val in zip(bars, df["MAPE%"], strict=True):
        ax.text(bar.get_width() + 0.05, bar.get_y() + bar.get_height() / 2,
                f"{val:.2f}%", va="center", fontsize=9)

    # RMSE bar chart
    ax2 = axes[1]
    bars2 = ax2.barh(df["Model"], df["RMSE"], color=colors)
    ax2.set_xlabel("RMSE")
    ax2.set_title("Model Comparison — RMSE (lower is better)")
    ax2.invert_yaxis()
    for bar, val in zip(bars2, df["RMSE"], strict=True):
        ax2.text(bar.get_width() + 5, bar.get_y() + bar.get_height() / 2,
                 f"{val:.0f}", va="center", fontsize=9)

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#2196F3", label="Classical"),
        Patch(facecolor="#FF5722", label="Deep Learning"),
    ]
    fig.legend(handles=legend_elements, loc="lower center", ncol=2, fontsize=10)
    fig.suptitle(
        "NZ Retail Sales Forecasting — Classical vs Deep Learning\n"
        "Test set: 8 quarters (2023 Q1 – 2024 Q4)  |  Train: 102 quarters",
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0.06, 1, 1])
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    logger.info("Saved comparison figure → %s", save_path)


def run_comparison() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    cfg = load_config()

    df = load_all_metrics(cfg)
    if df.empty:
        print("No metrics found. Run train.py and train_dl.py first.")
        return

    out_dir = get_project_root() / "models"
    out_dir.mkdir(exist_ok=True)
    plot_comparison(df, out_dir / "dl_vs_classical_comparison.png")

    print("\n=== Full Model Comparison (ranked by MAPE) ===")
    print(df.to_string(
        index=False,
        formatters={
            "MAPE%": "{:.2f}%".format,
            "RMSE":  "{:.0f}".format,
            "DA%":   "{:.1f}%".format,
        }
    ))


if __name__ == "__main__":
    run_comparison()
