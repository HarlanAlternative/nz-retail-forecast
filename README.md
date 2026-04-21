# NZ Retail Sales Forecasting

Forecasts New Zealand quarterly retail sales using official government data, with a full MLOps pipeline: multi-source data ingestion → pandera validation → feature engineering → multi-model training with MLflow tracking → FastAPI inference service.

Best model: **Ridge, MAPE 0.95%** on a held-out 8-quarter test set (2023 Q1 – 2024 Q4).

## Business Problem

NZ retailers face demand uncertainty driven by inflation, employment shifts, and seasonal patterns. Accurate quarterly forecasts help procurement teams optimise inventory, reduce stockouts, and improve cash-flow planning.

## Data Sources

All data is publicly available — no paid licences required.

| Source | Content | Access |
|--------|---------|--------|
| Stats NZ Retail Trade Survey | Quarterly retail sales (RTTQ.SF11CA, from 1995 Q3) | Public CSV download |
| Stats NZ ADE API | Monthly CPI food price index (from 2000) | SDMX-JSON 2.0 |
| IMF DataMapper | NZ annual unemployment rate (from 1980 → quarterly interpolation) | Unauthenticated REST API |
| OECD MEI_FIN | NZ 3-month interbank rate (monthly, from 2018) | SDMX-JSON |

> **Note:** The original Stats NZ API (api.stats.govt.nz) was decommissioned in August 2024. This project has been fully migrated to the replacement data sources above.

## Architecture

```
Multi-source ingestion → pandera validation → TimeSeriesFeatureEngineer
    → 6 models + Stacking (Optuna + MLflow) → ablation experiment → FastAPI
```

| Stage | Tool | Output |
|-------|------|--------|
| Ingestion & caching | Multi-source clients + parquet | `data/raw/*.parquet` |
| Validation | pandera `DataFrameSchema` | `SchemaError` on bad data |
| Feature engineering | sklearn `TransformerMixin` | Lags, rolling stats, YoY, cyclical encoding, CPI-adjusted values, exogenous lags |
| Training | 6 models + Stacking + Optuna | `models/best_model.joblib` |
| Experiment tracking | MLflow | `mlruns/` |
| Ablation experiment | `ablation.py` | Exogenous variable signal vs noise test |
| Inference | FastAPI | `/forecast` |

## Results

Test set: 8 quarters (2023 Q1 – 2024 Q4) · Training set: 102 quarters (1995 Q3 – 2022 Q4)

| Model | RMSE | MAE | MAPE | Directional Accuracy |
|-------|-----:|----:|-----:|---------------------:|
| **Ridge (best)** | **240** | **222** | **0.95%** | 85.7% |
| Prophet | 643 | 471 | 2.00% | 71.4% |
| Holt-Winters | 1,708 | 1,474 | 6.23% | 85.7% |
| SARIMA(1,1,1)(1,1,0)[4] | 1,949 | 1,715 | 7.23% | 57.1% |
| LightGBM + Optuna | 4,068 | 3,939 | 16.67% | 71.4% |

### Why does Ridge outperform LightGBM?

This is the key empirical finding of the project, quantified by the ablation experiment:

1. **Small sample size:** 102 training quarters is insufficient for a tree model to learn high-dimensional feature interactions without overfitting. Ridge's L2 regularisation handles this naturally.
2. **Linearly-structured data:** NZ retail sales exhibit a strong linear trend plus quarterly seasonality — exactly the pattern Ridge exploits via lag features (lag 1q + lag 4q ≈ weighted linear extrapolation).
3. **Low-coverage exogenous features add noise to tree models (ablation-verified):** Removing sparse exogenous variables (CPI 85% coverage, unemployment from annual interpolation, interest rate 6%) drops LightGBM MAPE from 15.04% → 14.55% (−0.49pp). Ridge is largely unaffected because L2 regularisation suppresses low-signal features automatically.

## Quick Start

### 1. Install

```bash
git clone <repo>
cd nz_economy
pip install -e ".[dev]"

# Stats NZ ADE API key (free registration at portal.apis.stats.govt.nz)
cp .env.example .env
# Set ADE_API_KEY=your_key_here in .env
```

### 2. Fetch data

```bash
python -m forecasting.data
# Output: data/raw/*.parquet, data/processed/merged.parquet
# Coverage report: CPI 85% | Unemployment 100% | Interest rate 6%
```

### 3. Train all models

```bash
python -m forecasting.train
# Trains LightGBM + Ridge + ElasticNet + SARIMA + Holt-Winters + Prophet + Stacking
# Prints ranked leaderboard; all metrics logged to MLflow
mlflow ui --backend-store-uri mlruns/
```

### 4. Ablation experiment

```bash
python -m forecasting.ablation
# Question: does removing exogenous variables improve LightGBM MAPE?
# Answers: noise hypothesis vs sample-size hypothesis
```

### 5. Start inference API

```bash
uvicorn api.app:app --reload
# GET http://localhost:8000/health
# GET http://localhost:8000/forecast?quarters_ahead=4
# Docs: http://localhost:8000/docs
```

### 6. Tests & linting

```bash
pytest tests/ -v
ruff check src/
```

## Project Structure

```
nz_economy/
├── src/forecasting/
│   ├── config.py       # Config loading, path resolution
│   ├── data.py         # Multi-source clients (Stats NZ / ADE / IMF / OECD) + pandera validation
│   ├── features.py     # Time series feature engineering (sklearn TransformerMixin)
│   ├── evaluate.py     # RMSE / MAE / MAPE / directional accuracy / Ljung-Box / plots
│   ├── train.py        # 6 models + Stacking + Optuna + MLflow full pipeline
│   └── ablation.py     # Exogenous feature ablation experiment
├── api/app.py          # FastAPI inference service
├── tests/              # pytest unit tests
├── config/config.yaml  # All tunable parameters
└── models/             # Saved model bundles + feature importance + forecast plots
```

## Stack

Python · LightGBM · Prophet · statsmodels · scikit-learn · MLflow · Optuna · FastAPI · pandera · pytest · ruff

## Key Engineering Decisions

| Decision | Rationale |
|----------|-----------|
| Quarterly frequency | Stats NZ Retail Trade Survey is quarterly; forcing monthly disaggregation adds no information |
| Ridge as production model | Ablation experiment + full model comparison confirm linear models are optimal at this sample size |
| Exogenous coverage threshold (≥ 20%) | Sparse features hurt LightGBM via imputer noise; auto-skipped below threshold |
| Annual unemployment → quarterly interpolation | IMF DataMapper is the only freely accessible, long-history NZ unemployment series |
| parquet caching | Avoids re-downloading large Stats NZ files; supports offline development |
