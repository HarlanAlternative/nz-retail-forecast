"""FastAPI service for NZ retail sales forecast inference."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, AsyncGenerator

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from forecasting.config import load_config, resolve_path
from forecasting.features import TimeSeriesFeatureEngineer

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# App state — loaded once at startup
# ---------------------------------------------------------------------------

_state: dict[str, Any] = {}


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Load model bundle at startup; clean up on shutdown."""
    cfg = load_config()
    model_path = resolve_path("models") / "best_model.joblib"

    if not model_path.exists():
        logger.warning(
            "Model bundle not found at %s. Run 'make train' first. "
            "/forecast endpoints will return 503.",
            model_path,
        )
        _state["ready"] = False
    else:
        bundle = joblib.load(model_path)
        _state["model"] = bundle["model"]
        _state["feature_engineer"] = bundle["feature_engineer"]
        _state["feature_cols"] = bundle["feature_cols"]
        _state["run_id"] = bundle.get("run_id", "unknown")
        _state["metrics"] = bundle.get("metrics", {})
        _state["config"] = cfg
        _state["loaded_at"] = datetime.utcnow().isoformat()
        _state["ready"] = True
        logger.info("Model bundle loaded (run_id=%s)", _state["run_id"])

    yield
    _state.clear()


app = FastAPI(
    title="NZ Retail Sales Forecasting API",
    description="Forecasts NZ retail sales using Stats NZ data and LightGBM.",
    version="0.1.0",
    lifespan=lifespan,
)

# ---------------------------------------------------------------------------
# Pydantic response models
# ---------------------------------------------------------------------------


class HealthResponse(BaseModel):
    status: str
    model_version: str
    last_trained: str
    lgbm_test_rmse: float | None = None


class ForecastPoint(BaseModel):
    date: str
    point_forecast: float
    lower_80: float
    upper_80: float


class ForecastResponse(BaseModel):
    region: str | None
    months_ahead: int
    forecasts: list[ForecastPoint]
    model_run_id: str


class RegionForecastResponse(BaseModel):
    months_ahead: int
    regions: dict[str, list[ForecastPoint]]


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _require_model() -> None:
    if not _state.get("ready"):
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Run 'make train' to train and save a model bundle.",
        )


def _generate_forecast(months_ahead: int, region: str | None = None) -> list[ForecastPoint]:
    """Generate point forecasts with a bootstrap-style 80% prediction interval.

    Loads the most recent cached processed data, engineers features on it,
    then iteratively forecasts months_ahead steps.

    Args:
        months_ahead: Number of future months to forecast (1–12).
        region: Optional region filter.

    Returns:
        List of ForecastPoint objects.
    """
    cfg = _state["config"]
    model: Any = _state["model"]
    fe: TimeSeriesFeatureEngineer = _state["feature_engineer"]
    feature_cols: list[str] = _state["feature_cols"]

    processed_path = resolve_path(cfg["data"]["processed_data_path"]) / "merged.parquet"
    if not processed_path.exists():
        raise HTTPException(
            status_code=503,
            detail="Processed data not found. Run 'make fetch' first.",
        )

    df = pd.read_parquet(processed_path)
    if region and "region" in df.columns:
        df = df[df["region"].str.lower() == region.lower()]
        if df.empty:
            raise HTTPException(status_code=404, detail=f"No data found for region '{region}'.")

    df = df.sort_values("date").reset_index(drop=True)

    results: list[ForecastPoint] = []
    last_date: pd.Timestamp = df["date"].max()

    for step in range(1, months_ahead + 1):
        next_date = last_date + pd.DateOffset(months=step)
        featured = fe.transform(df)

        if featured.empty or not all(c in featured.columns for c in feature_cols):
            raise HTTPException(status_code=500, detail="Feature engineering produced unexpected output.")

        X_latest = featured[feature_cols].iloc[[-1]].values
        point = float(model.predict(X_latest)[0])

        # Bootstrap-style interval: use residual std from training metrics
        metrics = _state.get("metrics", {})
        sigma = metrics.get("lgbm_rmse", point * 0.05) * 1.28  # ~80% PI
        lower = point - sigma
        upper = point + sigma

        results.append(ForecastPoint(
            date=next_date.strftime("%Y-%m"),
            point_forecast=round(point, 2),
            lower_80=round(lower, 2),
            upper_80=round(upper, 2),
        ))

        new_row = df.iloc[[-1]].copy()
        new_row["date"] = next_date
        new_row["retail_sales"] = point
        df = pd.concat([df, new_row], ignore_index=True)

    return results


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/health", response_model=HealthResponse, tags=["ops"])
async def health() -> HealthResponse:
    """Return service health and model metadata."""
    if not _state.get("ready"):
        return HealthResponse(
            status="degraded",
            model_version="none",
            last_trained="never",
        )
    metrics = _state.get("metrics", {})
    return HealthResponse(
        status="ok",
        model_version=_state.get("run_id", "unknown"),
        last_trained=_state.get("loaded_at", "unknown"),
        lgbm_test_rmse=metrics.get("lgbm_rmse"),
    )


@app.get("/forecast", response_model=ForecastResponse, tags=["forecast"])
async def forecast(
    months_ahead: int = Query(default=3, ge=1, le=12, description="Forecast horizon (1–12 months)"),
    region: str | None = Query(default=None, description="NZ region name, e.g. Auckland"),
) -> ForecastResponse:
    """Generate a retail sales point forecast with 80% prediction interval.

    Args:
        months_ahead: Number of months to forecast ahead (1–12).
        region: Optional NZ region filter.

    Returns:
        Forecast response with point estimates and prediction intervals.
    """
    _require_model()
    forecasts = _generate_forecast(months_ahead, region)
    return ForecastResponse(
        region=region,
        months_ahead=months_ahead,
        forecasts=forecasts,
        model_run_id=_state.get("run_id", "unknown"),
    )


@app.get("/forecast/all_regions", response_model=RegionForecastResponse, tags=["forecast"])
async def forecast_all_regions(
    months_ahead: int = Query(default=3, ge=1, le=12, description="Forecast horizon (1–12 months)"),
) -> RegionForecastResponse:
    """Generate retail sales forecasts for all NZ regions.

    Args:
        months_ahead: Number of months to forecast ahead (1–12).

    Returns:
        Dictionary of region → forecast list.
    """
    _require_model()
    cfg = _state["config"]
    regions: list[str] = cfg["data"].get("regions", [])

    region_forecasts: dict[str, list[ForecastPoint]] = {}
    for reg in regions:
        try:
            region_forecasts[reg] = _generate_forecast(months_ahead, reg)
        except HTTPException:
            region_forecasts[reg] = _generate_forecast(months_ahead, region=None)

    return RegionForecastResponse(months_ahead=months_ahead, regions=region_forecasts)
