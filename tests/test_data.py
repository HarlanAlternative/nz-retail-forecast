"""Tests for ADEClient: SDMX parsing, pandera validation, and caching."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pandera.pandas as pa
import pytest

from forecasting.data import ADEClient, RETAIL_SCHEMA


# ---------------------------------------------------------------------------
# Helper: expose private static method for testing without full client setup
# ---------------------------------------------------------------------------

def _parse(response: dict, col: str = "retail_sales") -> pd.DataFrame:
    return ADEClient._parse_sdmx_json(response, col)


# ---------------------------------------------------------------------------
# SDMX parsing tests
# ---------------------------------------------------------------------------

class TestSDMXParsing:
    def test_parse_returns_correct_columns(self, sample_sdmx_response):
        df = _parse(sample_sdmx_response)
        assert "date" in df.columns
        assert "retail_sales" in df.columns

    def test_parse_correct_row_count(self, sample_sdmx_response):
        df = _parse(sample_sdmx_response)
        assert len(df) == 6

    def test_parse_dates_are_datetime(self, sample_sdmx_response):
        df = _parse(sample_sdmx_response)
        assert pd.api.types.is_datetime64_any_dtype(df["date"])

    def test_parse_dates_ascending(self, sample_sdmx_response):
        df = _parse(sample_sdmx_response)
        assert df["date"].is_monotonic_increasing

    def test_parse_values_correct(self, sample_sdmx_response):
        df = _parse(sample_sdmx_response)
        expected = [5000.0, 5100.0, 5200.0, 5300.0, 5400.0, 5500.0]
        assert list(df["retail_sales"]) == expected

    def test_parse_empty_response_raises(self, empty_sdmx_response):
        with pytest.raises(ValueError, match="No observations"):
            _parse(empty_sdmx_response)

    def test_parse_malformed_response_raises(self, malformed_sdmx_response):
        with pytest.raises(KeyError):
            _parse(malformed_sdmx_response)

    def test_parse_month_format_converted(self, sample_sdmx_response):
        df = _parse(sample_sdmx_response)
        assert df["date"].dt.month.tolist() == [1, 2, 3, 4, 5, 6]
        assert df["date"].dt.year.unique().tolist() == [2023]


# ---------------------------------------------------------------------------
# Pandera validation tests
# ---------------------------------------------------------------------------

class TestSchemaValidation:
    def test_valid_df_passes(self, retail_df):
        validated = RETAIL_SCHEMA.validate(retail_df)
        assert len(validated) == len(retail_df)

    def test_negative_sales_fails(self, retail_df):
        bad_df = retail_df.copy()
        bad_df.loc[0, "retail_sales"] = -100.0
        with pytest.raises(pa.errors.SchemaError):
            RETAIL_SCHEMA.validate(bad_df)

    def test_zero_sales_fails(self, retail_df):
        bad_df = retail_df.copy()
        bad_df.loc[0, "retail_sales"] = 0.0
        with pytest.raises(pa.errors.SchemaError):
            RETAIL_SCHEMA.validate(bad_df)

    def test_future_date_fails(self, retail_df):
        bad_df = retail_df.copy()
        bad_df.loc[0, "date"] = pd.Timestamp("2099-01-01")
        with pytest.raises(pa.errors.SchemaError, match="future dates"):
            RETAIL_SCHEMA.validate(bad_df)

    def test_non_datetime_date_coerced(self, retail_df):
        df = retail_df.copy()
        df["date"] = df["date"].astype(str)
        validated = RETAIL_SCHEMA.validate(df)
        assert pd.api.types.is_datetime64_any_dtype(validated["date"])


# ---------------------------------------------------------------------------
# Cache read/write tests
# ---------------------------------------------------------------------------

class TestCaching:
    def test_cache_write_creates_parquet(self, retail_df, tmp_path):
        client = ADEClient(api_key="test_key", cache_dir=tmp_path)
        client._save_cache(retail_df, "test_key")
        assert (tmp_path / "test_key.parquet").exists()

    def test_cache_read_returns_dataframe(self, retail_df, tmp_path):
        client = ADEClient(api_key="test_key", cache_dir=tmp_path)
        client._save_cache(retail_df, "test_key")
        result = client._load_cache("test_key")
        assert result is not None
        assert len(result) == len(retail_df)

    def test_cache_miss_returns_none(self, tmp_path):
        client = ADEClient(api_key="test_key", cache_dir=tmp_path)
        result = client._load_cache("nonexistent_key")
        assert result is None

    def test_cache_roundtrip_preserves_dtypes(self, retail_df, tmp_path):
        client = ADEClient(api_key="test_key", cache_dir=tmp_path)
        client._save_cache(retail_df, "dtype_test")
        result = client._load_cache("dtype_test")
        assert pd.api.types.is_datetime64_any_dtype(result["date"])
        assert result["retail_sales"].dtype == float
