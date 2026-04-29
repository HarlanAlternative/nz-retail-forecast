"""PCA feature ablation: does dimensionality reduction improve Ridge / LightGBM?

Key question: Does any PCA configuration recover the (false) 0.95% MAPE from
before the leakage fix? If not, this is independent confirmation that the 0.95%
was a genuine leakage artifact — not a compressible signal that PCA could surface.

Usage:
    python -m dimensionality.pca_features
"""

from __future__ import annotations

import logging
import warnings
from typing import Any

import lightgbm as lgb
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from dimensionality._utils import (
    eval_ridge,
    load_featured_data,
    prepare_arrays,
)
from forecasting.config import get_project_root, load_config
from forecasting.evaluate import directional_accuracy, mape, rmse

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)

_MODELS_DIR    = get_project_root() / "models"
_BASELINE_MAPE = 2.58   # corrected Ridge MAPE after leakage fix
_LEAKED_MAPE   = 0.95   # the inflated value before the fix
_EXPERIMENT    = "pca_ablation"


# ---------------------------------------------------------------------------
# LightGBM helper (fixed params — no Optuna, for speed)
# ---------------------------------------------------------------------------

def _eval_lgbm(
    X_tr: np.ndarray, y_tr: np.ndarray,
    X_te: np.ndarray, y_te: np.ndarray,
    cfg: dict[str, Any],
) -> dict[str, float]:
    params = {
        "n_estimators":     200,
        "learning_rate":    0.05,
        "num_leaves":       8,
        "min_child_samples": 15,
        "subsample":        0.8,
        "colsample_bytree": 0.8,
        "reg_alpha":        2.0,
        "reg_lambda":       2.0,
        "random_state":     cfg["model"]["random_seed"],
        "verbosity":        -1,
    }
    model = lgb.LGBMRegressor(**params)
    model.fit(X_tr, y_tr)
    preds = model.predict(X_te)
    return {
        "mape": mape(y_te, preds),
        "rmse": rmse(y_te, preds),
        "da":   directional_accuracy(y_te, preds),
    }


# ---------------------------------------------------------------------------
# PCA sweep
# ---------------------------------------------------------------------------

def run_pca_sweep(
    X_tr: np.ndarray,
    X_te: np.ndarray,
    y_tr: np.ndarray,
    y_te: np.ndarray,
    cfg: dict[str, Any],
) -> pd.DataFrame:
    p = X_tr.shape[1]
    n_components_list = [2, 5, 10, 15, min(20, p), p]
    n_components_list = sorted(set(n_components_list))

    rows = []
    for n in n_components_list:
        pca = PCA(n_components=n, random_state=cfg["model"]["random_seed"])
        X_tr_pca = pca.fit_transform(X_tr)
        X_te_pca = pca.transform(X_te)
        exp_var   = float(pca.explained_variance_ratio_.sum()) * 100

        ridge_m = eval_ridge(X_tr_pca, y_tr, X_te_pca, y_te,
                             alpha=cfg["model"]["ridge"]["alpha"])
        lgbm_m  = _eval_lgbm(X_tr_pca, y_tr, X_te_pca, y_te, cfg)

        row = {
            "n_components":  n,
            "explained_var": exp_var,
            "ridge_mape":    ridge_m["mape"],
            "ridge_rmse":    ridge_m["rmse"],
            "ridge_da":      ridge_m["da"],
            "lgbm_mape":     lgbm_m["mape"],
            "lgbm_rmse":     lgbm_m["rmse"],
            "lgbm_da":       lgbm_m["da"],
        }
        rows.append(row)

        mlflow.log_metrics({
            f"pca{n}_ridge_mape": ridge_m["mape"],
            f"pca{n}_ridge_rmse": ridge_m["rmse"],
            f"pca{n}_lgbm_mape":  lgbm_m["mape"],
            f"pca{n}_lgbm_rmse":  lgbm_m["rmse"],
            f"pca{n}_explained_var": exp_var,
        })

        logger.info(
            "  n=%2d  var=%.1f%%  Ridge MAPE=%.2f%%  LightGBM MAPE=%.2f%%",
            n, exp_var, ridge_m["mape"], lgbm_m["mape"],
        )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def _plot_mape_sweep(df: pd.DataFrame, save_path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df["n_components"], df["ridge_mape"],
            "o-", color="#2196F3", linewidth=2, label="Ridge (PCA features)")
    ax.plot(df["n_components"], df["lgbm_mape"],
            "s-", color="#FF9800", linewidth=2, label="LightGBM (PCA features)")
    ax.axhline(_BASELINE_MAPE, color="#4CAF50", linestyle="--", linewidth=2,
               label=f"Corrected Ridge baseline ({_BASELINE_MAPE:.2f}%)")
    ax.axhline(_LEAKED_MAPE, color="red", linestyle=":", linewidth=1.5,
               label=f"Pre-fix Ridge (leaked, {_LEAKED_MAPE:.2f}%)")
    ax.set_xlabel("PCA components (n)")
    ax.set_ylabel("MAPE (%)")
    ax.set_title("PCA Ablation: Does Dimensionality Reduction Improve Forecasting?\n"
                 "Red dotted line = pre-fix MAPE (data leakage artifact)")
    ax.legend()
    ax.set_ylim(bottom=0)

    # Mark the closest PCA approach to the leaked value
    min_ridge = df["ridge_mape"].min()
    ax.annotate(f"Best Ridge+PCA: {min_ridge:.2f}%",
                xy=(df.loc[df["ridge_mape"].idxmin(), "n_components"], min_ridge),
                xytext=(df["n_components"].mean(), min_ridge + 1.0),
                arrowprops={"arrowstyle": "->", "color": "#2196F3"},
                fontsize=9, color="#2196F3")

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    logger.info("Saved → %s", save_path)
    plt.close(fig)


def _leakage_verdict(df: pd.DataFrame) -> str:
    min_mape = df["ridge_mape"].min()
    gap = abs(min_mape - _LEAKED_MAPE)
    if gap < 0.3:
        return (
            f"WARNING: PCA+Ridge achieved {min_mape:.2f}% MAPE — suspiciously close "
            f"to the leaked value ({_LEAKED_MAPE}%). Investigate component mixing."
        )
    return (
        f"CONFIRMED: Best PCA+Ridge = {min_mape:.2f}% MAPE. "
        f"The leaked {_LEAKED_MAPE}% was NOT a compressible signal PCA could "
        f"recover. This is independent confirmation of the leakage diagnosis."
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_pca_features() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    _MODELS_DIR.mkdir(exist_ok=True)

    cfg = load_config()
    train_df, test_df, feat_cols = load_featured_data(cfg)

    X_tr, X_te, y_tr, y_te, *_ = prepare_arrays(train_df, test_df, feat_cols)

    # ── Sanity check: reproduce corrected Ridge baseline ────────────────────
    baseline = eval_ridge(X_tr, y_tr, X_te, y_te, alpha=cfg["model"]["ridge"]["alpha"])
    logger.info(
        "Sanity check — full-feature Ridge MAPE=%.2f%%  (expected ~%.2f%%)",
        baseline["mape"], _BASELINE_MAPE,
    )
    if abs(baseline["mape"] - _BASELINE_MAPE) > 0.5:
        logger.warning(
            "Baseline deviation %.2fpp — check feature engineering or data.",
            abs(baseline["mape"] - _BASELINE_MAPE),
        )

    # ── PCA sweep with MLflow logging ───────────────────────────────────────
    mlflow.set_tracking_uri(cfg["mlflow"]["tracking_uri"])
    mlflow.set_experiment(_EXPERIMENT)

    with mlflow.start_run(run_name="pca_ridge_lgbm_sweep"):
        mlflow.log_params({
            "baseline_ridge_mape": baseline["mape"],
            "n_train_quarters":    len(train_df),
            "n_test_quarters":     len(test_df),
            "n_features":          len(feat_cols),
        })

        logger.info("Running PCA sweep (n_components ∈ {2,5,10,15,20,full})...")
        df = run_pca_sweep(X_tr, X_te, y_tr, y_te, cfg)

        verdict = _leakage_verdict(df)
        mlflow.log_param("leakage_verdict", verdict)

    _plot_mape_sweep(df, _MODELS_DIR / "pca_mape_sweep.png")

    # ── Print results table ─────────────────────────────────────────────────
    print("\n=== PCA Ablation Results ===")
    print(f"{'n':>4}  {'var%':>6}  {'Ridge MAPE':>11}  {'LGBM MAPE':>10}  {'vs baseline':>12}")
    print("-" * 52)
    for _, r in df.iterrows():
        delta = r["ridge_mape"] - _BASELINE_MAPE
        print(f"{int(r['n_components']):>4}  {r['explained_var']:>5.1f}%"
              f"  {r['ridge_mape']:>9.2f}%  {r['lgbm_mape']:>8.2f}%"
              f"  {delta:>+10.2f}pp")

    print(f"\n{verdict}")
    print(f"\nFull-feature Ridge baseline (corrected): {_BASELINE_MAPE:.2f}%")
    print(f"Pre-fix Ridge (leaked):                  {_LEAKED_MAPE:.2f}%")
    print("\nFigure saved → models/pca_mape_sweep.png")
    print("MLflow run logged to experiment: pca_ablation")


if __name__ == "__main__":
    run_pca_features()
