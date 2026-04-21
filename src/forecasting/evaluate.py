"""Forecast evaluation metrics and diagnostic plots."""

from __future__ import annotations

import logging
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.stats.diagnostic import acorr_ljungbox

logger = logging.getLogger(__name__)


def rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Root Mean Squared Error.

    Args:
        actual: Ground-truth values.
        predicted: Model predictions.

    Returns:
        RMSE as a float.
    """
    return float(np.sqrt(np.mean((actual - predicted) ** 2)))


def mae(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Mean Absolute Error.

    Args:
        actual: Ground-truth values.
        predicted: Model predictions.

    Returns:
        MAE as a float.
    """
    return float(np.mean(np.abs(actual - predicted)))


def mape(actual: np.ndarray, predicted: np.ndarray, epsilon: float = 1e-8) -> float:
    """Mean Absolute Percentage Error.

    Args:
        actual: Ground-truth values.
        predicted: Model predictions.
        epsilon: Small value added to denominator to avoid division by zero.

    Returns:
        MAPE as a percentage float (e.g. 5.3 means 5.3%).
    """
    return float(np.mean(np.abs((actual - predicted) / (np.abs(actual) + epsilon))) * 100)


def directional_accuracy(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Fraction of steps where the forecast direction matches the actual direction.

    Direction is measured as the sign of the first-difference. The first
    element is excluded because there is no prior value to difference against.

    Args:
        actual: Ground-truth values (length ≥ 2).
        predicted: Model predictions (same length as actual).

    Returns:
        Directional accuracy between 0.0 and 1.0.

    Raises:
        ValueError: If arrays are shorter than 2 elements.
    """
    if len(actual) < 2:
        raise ValueError("Arrays must have at least 2 elements for directional accuracy.")
    actual_dir = np.sign(np.diff(actual))
    pred_dir = np.sign(np.diff(predicted))
    return float(np.mean(actual_dir == pred_dir))


def plot_forecast_vs_actual(
    dates: pd.Series | np.ndarray,
    actual: np.ndarray,
    predicted: np.ndarray,
    lower: np.ndarray | None = None,
    upper: np.ndarray | None = None,
    title: str = "Forecast vs Actual",
    save_path: str | None = None,
) -> plt.Figure:
    """Plot forecast against actual values with optional prediction interval.

    Args:
        dates: Sequence of date values for the x-axis.
        actual: Ground-truth observations.
        predicted: Point forecasts.
        lower: Lower bound of prediction interval (80% by convention).
        upper: Upper bound of prediction interval.
        title: Plot title.
        save_path: If provided, save figure to this path instead of showing.

    Returns:
        Matplotlib Figure object.
    """
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(dates, actual, label="Actual", color="#1f77b4", linewidth=2)
    ax.plot(dates, predicted, label="Forecast", color="#ff7f0e", linewidth=2, linestyle="--")

    if lower is not None and upper is not None:
        ax.fill_between(dates, lower, upper, alpha=0.25, color="#ff7f0e", label="80% PI")

    ax.set_title(title, fontsize=14)
    ax.set_xlabel("Date")
    ax.set_ylabel("Retail Sales")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)
        logger.info("Saved forecast plot → %s", save_path)
    return fig


def residual_analysis(
    residuals: np.ndarray,
    lags: int = 10,
    significance: float = 0.05,
) -> dict[str, Any]:
    """Run Ljung-Box autocorrelation test on forecast residuals.

    Args:
        residuals: Forecast errors (actual − predicted).
        lags: Number of lags for the Ljung-Box test.
        significance: p-value threshold for the autocorrelation test.

    Returns:
        Dictionary with keys:
            - lb_stat: Ljung-Box test statistics per lag.
            - lb_pvalue: p-values per lag.
            - autocorrelated: True if any p-value < significance level.
            - verdict: Human-readable string.
    """
    result = acorr_ljungbox(residuals, lags=lags, return_df=True)
    autocorrelated = bool((result["lb_pvalue"] < significance).any())
    verdict = (
        "Residuals show significant autocorrelation — consider adding more lags or a richer model."
        if autocorrelated
        else "No significant autocorrelation detected in residuals."
    )
    logger.info("Ljung-Box test: %s", verdict)
    return {
        "lb_stat": result["lb_stat"].tolist(),
        "lb_pvalue": result["lb_pvalue"].tolist(),
        "autocorrelated": autocorrelated,
        "verdict": verdict,
    }
