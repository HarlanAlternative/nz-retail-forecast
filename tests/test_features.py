"""Tests for TimeSeriesFeatureEngineer: correctness, leakage, and output shape."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from forecasting.features import TimeSeriesFeatureEngineer


@pytest.fixture()
def fe() -> TimeSeriesFeatureEngineer:
    return TimeSeriesFeatureEngineer(
        lag_months=[1, 3],
        rolling_windows=[3],
        nan_strategy="drop",
    )


class TestLagFeatures:
    def test_lag1_correct_value(self, fe, retail_df):
        result = fe.transform(retail_df)
        # After dropping NaN rows, first valid row has lag1 == original row before it
        orig = retail_df["retail_sales"].values
        lag1_col = result["retail_lag_1m"].values
        result_dates = result["date"].values
        for i, date in enumerate(result_dates):
            idx = retail_df[retail_df["date"] == date].index[0]
            expected = orig[idx - 1]
            assert abs(lag1_col[i] - expected) < 1e-6

    def test_lag3_correct_value(self, fe, retail_df):
        result = fe.transform(retail_df)
        orig = retail_df["retail_sales"].values
        for i, date in enumerate(result["date"].values):
            idx = retail_df[retail_df["date"] == date].index[0]
            expected = orig[idx - 3]
            assert abs(result["retail_lag_3m"].values[i] - expected) < 1e-6

    def test_no_nan_in_output_with_drop_strategy(self, fe, retail_df):
        result = fe.transform(retail_df)
        assert not result.isnull().any().any()

    def test_no_data_leakage_rolling(self, fe, retail_df):
        """Rolling mean at position t must NOT use observation at t (only t-1 and before)."""
        result = fe.transform(retail_df)
        for i, date in enumerate(result["date"].values):
            idx = retail_df[retail_df["date"] == date].index[0]
            if idx < 3:
                continue
            expected_mean = retail_df["retail_sales"].iloc[idx - 3: idx].mean()
            actual_mean = result["retail_roll_mean_3m"].values[i]
            assert abs(actual_mean - expected_mean) < 1e-6, (
                f"Leakage detected at index {idx}: expected {expected_mean}, got {actual_mean}"
            )


class TestOutputShape:
    def test_output_has_more_columns_than_input(self, fe, retail_df):
        result = fe.transform(retail_df)
        assert result.shape[1] > retail_df.shape[1]

    def test_output_rows_less_than_input_due_to_drop(self, fe, retail_df):
        result = fe.transform(retail_df)
        assert len(result) < len(retail_df)

    def test_fill_zero_strategy_preserves_all_rows(self, retail_df):
        fe_fill = TimeSeriesFeatureEngineer(
            lag_months=[1], rolling_windows=[3], nan_strategy="fill_zero"
        )
        result = fe_fill.transform(retail_df)
        assert len(result) == len(retail_df)

    def test_cyclical_month_columns_present(self, fe, retail_df):
        result = fe.transform(retail_df)
        assert "month_sin" in result.columns
        assert "month_cos" in result.columns

    def test_cyclical_month_range(self, fe, retail_df):
        result = fe.transform(retail_df)
        assert result["month_sin"].between(-1, 1).all()
        assert result["month_cos"].between(-1, 1).all()

    def test_cpi_adjusted_column_added_when_cpi_present(self, fe, merged_df):
        result = fe.transform(merged_df)
        assert "retail_real" in result.columns

    def test_invalid_nan_strategy_raises(self, retail_df):
        fe_bad = TimeSeriesFeatureEngineer(nan_strategy="invalid_strategy")
        with pytest.raises(ValueError, match="nan_strategy"):
            fe_bad.transform(retail_df)

    def test_missing_retail_sales_raises(self, fe):
        bad_df = pd.DataFrame({"date": pd.date_range("2023-01-01", periods=3, freq="MS"),
                                "cpi": [100.0, 101.0, 102.0]})
        with pytest.raises(ValueError, match="retail_sales"):
            fe.transform(bad_df)

    def test_fit_returns_self(self, fe, retail_df):
        result = fe.fit(retail_df)
        assert result is fe
