"""Ablation experiment: LightGBM with vs without exogenous features.

Tests the hypothesis that low-coverage exogenous variables (CPI, unemployment,
interest rate) act as noise for tree models on small quarterly datasets.

Usage:
    python -m forecasting.ablation
"""

from __future__ import annotations

import logging
import warnings

import lightgbm as lgb
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.model_selection import TimeSeriesSplit

from forecasting.config import load_config
from forecasting.data import build_merged_dataset
from forecasting.evaluate import mae, mape, rmse
from forecasting.features import TimeSeriesFeatureEngineer

logger = logging.getLogger(__name__)


def _train_lgbm(X_train: np.ndarray, y_train: np.ndarray,
                X_test: np.ndarray, y_test: np.ndarray,
                params: dict, n_splits: int) -> dict[str, float]:
    imp = SimpleImputer(strategy="median")
    X_tr = imp.fit_transform(X_train)
    X_te = imp.transform(X_test)

    tscv = TimeSeriesSplit(n_splits=n_splits)
    cv_rmses = []
    for tr_i, val_i in tscv.split(X_tr):
        m = lgb.LGBMRegressor(**params)
        m.fit(X_tr[tr_i], y_train[tr_i])
        cv_rmses.append(rmse(y_train[val_i], m.predict(X_tr[val_i])))

    final = lgb.LGBMRegressor(**params)
    final.fit(X_tr, y_train)
    preds = final.predict(X_te)

    return {
        "rmse": rmse(y_test, preds),
        "mae":  mae(y_test, preds),
        "mape": mape(y_test, preds),
        "cv_rmse_mean": float(np.mean(cv_rmses)),
    }


def run_ablation() -> None:
    """Run the exogenous-feature ablation and print a comparison table."""
    logging.basicConfig(level=logging.WARNING)
    warnings.filterwarnings("ignore")

    cfg = load_config()
    merged = build_merged_dataset(cfg["data"]["start_date"])

    fe = TimeSeriesFeatureEngineer(
        lag_months=cfg["features"]["lag_periods"],
        rolling_windows=cfg["features"]["rolling_windows"],
        nan_strategy=cfg["features"]["nan_fill_strategy"],
    )
    featured = fe.fit_transform(merged)

    test_months = cfg["model"]["test_size_months"]
    train_df = featured.iloc[:-test_months]
    test_df  = featured.iloc[-test_months:]
    y_train  = train_df["retail_sales"].values
    y_test   = test_df["retail_sales"].values

    all_cols         = [c for c in featured.columns if c not in ("date", "retail_sales")]
    retail_only_cols = [c for c in all_cols if c.startswith(("retail_", "month_"))]
    exog_cols        = [c for c in all_cols if c not in retail_only_cols]

    params = {
        "n_estimators": 200,
        "learning_rate": 0.05,
        "num_leaves": 8,
        "min_child_samples": 15,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 2.0,
        "reg_lambda": 2.0,
        "random_state": cfg["model"]["random_seed"],
        "verbosity": -1,
    }
    n_splits = cfg["model"]["cv_folds"]

    print(f"\n训练集: {len(y_train)} 季度 | 测试集: {len(y_test)} 季度")
    print(f"完整特征集: {len(all_cols)} 个  |  零售自身: {len(retail_only_cols)} 个")
    print(f"外生特征 ({len(exog_cols)}): {exog_cols}\n")

    runs = [
        ("LightGBM + 全部特征（含外生）", all_cols),
        ("LightGBM + 仅零售自身特征",     retail_only_cols),
    ]

    results = []
    for label, cols in runs:
        r = _train_lgbm(
            train_df[cols].values, y_train,
            test_df[cols].values,  y_test,
            params, n_splits,
        )
        r["label"] = label
        results.append(r)
        print(f"  [{label}]  MAPE={r['mape']:.2f}%  RMSE={r['rmse']:.1f}  CV-RMSE={r['cv_rmse_mean']:.1f}")

    delta = results[1]["mape"] - results[0]["mape"]
    direction = "下降" if delta < 0 else "上升"
    verdict = (
        "外生变量对树模型是噪声（低覆盖率 + 插值数据稀释了有效特征的分裂权重）"
        if delta < 0
        else "样本量是根本原因，外生变量影响有限"
    )
    print(f"\n结论: 去掉外生变量后 MAPE {direction} {abs(delta):.2f}pp → {verdict}")


if __name__ == "__main__":
    run_ablation()
