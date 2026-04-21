"""Time series feature engineering for NZ retail sales forecasting."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

logger = logging.getLogger(__name__)


class TimeSeriesFeatureEngineer(BaseEstimator, TransformerMixin):
    """Stateless sklearn-compatible transformer for time series features.

    Creates lag features, rolling statistics, YoY change, cyclical
    month encoding, CPI-adjusted values, and employment exogenous feature.

    Args:
        lag_months: List of lag periods in months.
        rolling_windows: List of rolling window sizes in months.
        nan_strategy: How to handle NaN from lags — 'drop' or 'fill_zero'.
    """

    def __init__(
        self,
        lag_months: list[int] | None = None,
        rolling_windows: list[int] | None = None,
        nan_strategy: str = "drop",
    ) -> None:
        self.lag_months = lag_months or [1, 3, 6, 12]
        self.rolling_windows = rolling_windows or [3, 6, 12]
        self.nan_strategy = nan_strategy

    def fit(self, X: pd.DataFrame, y: Any = None) -> "TimeSeriesFeatureEngineer":
        """No-op: transformer is stateless.

        Args:
            X: Input DataFrame (unused).
            y: Ignored.

        Returns:
            self
        """
        return self

    def transform(self, X: pd.DataFrame, y: Any = None) -> pd.DataFrame:
        """Apply all feature engineering steps.

        Args:
            X: DataFrame with columns [date, retail_sales] and optionally
               [cpi, employment_rate]. Must be sorted by date ascending.

        Returns:
            Feature DataFrame. Rows with NaN lags are dropped or zero-filled
            depending on nan_strategy.

        Raises:
            ValueError: If required columns are missing or date is not sorted.
        """
        self._validate_input(X)
        df = X.copy().sort_values("date").reset_index(drop=True)

        df = self._add_lag_features(df)
        df = self._add_rolling_features(df)
        df = self._add_yoy_change(df)
        df = self._add_cyclical_month(df)

        if "cpi" in df.columns:
            df = self._add_cpi_adjusted(df)

        # Exogenous lags — only add if coverage exceeds threshold to avoid imputer noise
        _MIN_COVERAGE = 0.20  # require at least 20% non-null before adding lag features
        for exog_col, lag_periods in [
            ("unemployment_rate", [1, 4]),   # lag 1q and 1yr
            ("interest_rate_90d", [1, 4]),   # lag 1q and 1yr (skipped if sparse)
            ("employment_count", [1]),
        ]:
            if exog_col in df.columns:
                coverage = df[exog_col].notna().mean()
                if coverage >= _MIN_COVERAGE:
                    for lag in lag_periods:
                        df[f"{exog_col}_lag{lag}"] = df[exog_col].shift(lag)
                else:
                    logger.debug(
                        "Skipping lag features for '%s' (coverage %.1f%% < %.0f%%)",
                        exog_col, coverage * 100, _MIN_COVERAGE * 100,
                    )

        df = self._handle_nan(df)
        logger.debug("Feature engineering complete: %d rows × %d cols", *df.shape)
        return df

    # ------------------------------------------------------------------
    # Feature builders
    # ------------------------------------------------------------------

    def _add_lag_features(self, df: pd.DataFrame) -> pd.DataFrame:
        for lag in self.lag_months:
            df[f"retail_lag_{lag}m"] = df["retail_sales"].shift(lag)
        return df

    def _add_rolling_features(self, df: pd.DataFrame) -> pd.DataFrame:
        for window in self.rolling_windows:
            # shift(1) ensures we never include the current observation (no leakage)
            rolled = df["retail_sales"].shift(1).rolling(window=window, min_periods=window)
            df[f"retail_roll_mean_{window}m"] = rolled.mean()
            df[f"retail_roll_std_{window}m"] = rolled.std()
        return df

    def _add_yoy_change(self, df: pd.DataFrame) -> pd.DataFrame:
        # 4 periods = 1 year for quarterly data; 12 periods for monthly
        yoy_periods = 4
        yoy_base = df["retail_sales"].shift(yoy_periods)
        df["retail_yoy_pct"] = (df["retail_sales"] - yoy_base) / yoy_base * 100
        return df

    @staticmethod
    def _add_cyclical_month(df: pd.DataFrame) -> pd.DataFrame:
        month = df["date"].dt.month
        df["month_sin"] = np.sin(2 * np.pi * month / 12)
        df["month_cos"] = np.cos(2 * np.pi * month / 12)
        return df

    @staticmethod
    def _add_cpi_adjusted(df: pd.DataFrame) -> pd.DataFrame:
        # Normalise CPI so it doesn't dominate scale; use first non-null as base
        base_cpi = df["cpi"].dropna().iloc[0] if not df["cpi"].dropna().empty else 1.0
        df["retail_real"] = df["retail_sales"] / (df["cpi"] / base_cpi)
        return df

    def _handle_nan(self, df: pd.DataFrame) -> pd.DataFrame:
        # Only drop/fill NaN based on engineered feature columns (lags, rolling,
        # yoy, cyclical), not on optional exogenous columns like employment_count
        # which may be entirely NaN but are still useful via imputation downstream.
        # Only drop/fill NaN based on core retail engineered features; exogenous
        # lag columns (unemployment, interest rate) may have partial NaN coverage
        # and are handled downstream by the imputer in the sklearn pipeline.
        engineered_cols = [
            c for c in df.columns
            if any(c.startswith(p) for p in [
                "retail_lag_", "retail_roll_", "retail_yoy_", "month_sin", "month_cos",
            ])
        ]

        if self.nan_strategy == "drop":
            before = len(df)
            df = df.dropna(subset=engineered_cols).reset_index(drop=True)
            dropped = before - len(df)
            if dropped:
                logger.debug("Dropped %d rows with NaN in lag/rolling features", dropped)
        elif self.nan_strategy == "fill_zero":
            df[engineered_cols] = df[engineered_cols].fillna(0)
        else:
            raise ValueError(f"Unknown nan_strategy: '{self.nan_strategy}'. Use 'drop' or 'fill_zero'.")
        return df

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_input(df: pd.DataFrame) -> None:
        if "retail_sales" not in df.columns:
            raise ValueError("Input DataFrame must contain 'retail_sales' column.")
        if "date" not in df.columns:
            raise ValueError("Input DataFrame must contain 'date' column.")
        if not pd.api.types.is_datetime64_any_dtype(df["date"]):
            raise ValueError("'date' column must be datetime dtype.")

    def get_feature_names_out(self, input_features: Any = None) -> list[str]:
        """Return list of feature column names produced by transform().

        Useful for compatibility with sklearn pipelines and feature importance.
        """
        names = ["date", "retail_sales"]
        for lag in self.lag_months:
            names.append(f"retail_lag_{lag}m")
        for w in self.rolling_windows:
            names += [f"retail_roll_mean_{w}m", f"retail_roll_std_{w}m"]
        names += ["retail_yoy_pct", "month_sin", "month_cos"]
        return names
