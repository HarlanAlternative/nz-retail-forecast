"""Shared helpers for dimensionality analysis modules."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from forecasting.config import load_config
from forecasting.data import build_merged_dataset
from forecasting.evaluate import directional_accuracy, mape, rmse
from forecasting.features import TimeSeriesFeatureEngineer

_EXCLUDE = {"date", "retail_sales"}
_RETAIL_PREFIXES = (
    "retail_lag_", "retail_roll_", "retail_yoy_", "month_sin", "month_cos",
)


def load_featured_data(cfg: dict[str, Any] | None = None):
    """Return (train_df, test_df, all_feat_cols)."""
    if cfg is None:
        cfg = load_config()
    merged = build_merged_dataset(cfg["data"]["start_date"])
    fe = TimeSeriesFeatureEngineer(
        lag_months=cfg["features"]["lag_periods"],
        rolling_windows=cfg["features"]["rolling_windows"],
        nan_strategy=cfg["features"]["nan_fill_strategy"],
    )
    featured = fe.fit_transform(merged)
    test_size = cfg["model"]["test_size_months"]
    train_df = featured.iloc[:-test_size].reset_index(drop=True)
    test_df  = featured.iloc[-test_size:].reset_index(drop=True)
    feat_cols = [c for c in featured.columns if c not in _EXCLUDE]
    return train_df, test_df, feat_cols


def retail_feature_cols(df: pd.DataFrame) -> list[str]:
    """Return only retail-intrinsic feature columns (no exogenous)."""
    return [c for c in df.columns if any(c.startswith(p) for p in _RETAIL_PREFIXES)]


def prepare_arrays(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feat_cols: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, SimpleImputer, StandardScaler]:
    """Impute → scale on train, transform test. Returns X_tr, X_te, y_tr, y_te, imputer, scaler."""
    imputer = SimpleImputer(strategy="median")
    scaler  = StandardScaler()
    X_tr = scaler.fit_transform(imputer.fit_transform(train_df[feat_cols].values))
    X_te = scaler.transform(imputer.transform(test_df[feat_cols].values))
    y_tr = train_df["retail_sales"].values
    y_te = test_df["retail_sales"].values
    return X_tr, X_te, y_tr, y_te, imputer, scaler


def eval_ridge(
    X_tr: np.ndarray, y_tr: np.ndarray,
    X_te: np.ndarray, y_te: np.ndarray,
    alpha: float = 1.0,
) -> dict[str, float]:
    model = Ridge(alpha=alpha)
    model.fit(X_tr, y_tr)
    preds = model.predict(X_te)
    return {
        "mape": mape(y_te, preds),
        "rmse": rmse(y_te, preds),
        "da":   directional_accuracy(y_te, preds),
    }
