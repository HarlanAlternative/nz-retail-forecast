"""Shared pytest fixtures providing SDMX-like sample data."""

from __future__ import annotations

import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Sample SDMX-JSON response (mirrors real ADE API structure)
# ---------------------------------------------------------------------------

@pytest.fixture()
def sample_sdmx_response() -> dict:
    """Minimal valid SDMX-JSON response with 6 monthly retail observations."""
    time_values = [
        {"id": "2023-M01"}, {"id": "2023-M02"}, {"id": "2023-M03"},
        {"id": "2023-M04"}, {"id": "2023-M05"}, {"id": "2023-M06"},
    ]
    observations = {str(i): [float(5000 + i * 100)] for i in range(6)}
    return {
        "data": {
            "structure": {
                "dimensions": {
                    "observation": [{"values": time_values}]
                }
            },
            "dataSets": [
                {
                    "series": {
                        "0:0:0": {"observations": observations}
                    }
                }
            ],
        }
    }


@pytest.fixture()
def empty_sdmx_response() -> dict:
    """SDMX-JSON response with no observations — should raise ValueError."""
    return {
        "data": {
            "structure": {
                "dimensions": {
                    "observation": [{"values": []}]
                }
            },
            "dataSets": [{"series": {}}],
        }
    }


@pytest.fixture()
def malformed_sdmx_response() -> dict:
    """SDMX-JSON missing the 'data' key — should raise KeyError."""
    return {"meta": {"id": "broken"}}


# ---------------------------------------------------------------------------
# Sample DataFrames
# ---------------------------------------------------------------------------

@pytest.fixture()
def retail_df() -> pd.DataFrame:
    """24 months of valid retail sales data."""
    dates = pd.date_range("2022-01-01", periods=24, freq="MS")
    return pd.DataFrame({
        "date": dates,
        "retail_sales": [float(5000 + i * 50) for i in range(24)],
    })


@pytest.fixture()
def merged_df(retail_df: pd.DataFrame) -> pd.DataFrame:
    """Retail + CPI + employment merged DataFrame (24 months)."""
    df = retail_df.copy()
    df["cpi"] = [100.0 + i * 0.3 for i in range(len(df))]
    df["employment_rate"] = [67.0 + i * 0.05 for i in range(len(df))]
    return df


@pytest.fixture()
def short_retail_df() -> pd.DataFrame:
    """Only 5 rows — useful for edge case tests."""
    dates = pd.date_range("2023-01-01", periods=5, freq="MS")
    return pd.DataFrame({
        "date": dates,
        "retail_sales": [5100.0, 5200.0, 5150.0, 5300.0, 5250.0],
    })
