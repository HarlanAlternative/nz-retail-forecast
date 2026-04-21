"""Model training: LightGBM + Ridge + ElasticNet + SARIMA + Holt-Winters + Stacking."""

from __future__ import annotations

import logging
import warnings
from pathlib import Path
from typing import Any

import joblib
import lightgbm as lgb
import mlflow
import mlflow.lightgbm
import numpy as np
import optuna
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from forecasting.config import load_config, resolve_path
from forecasting.data import build_merged_dataset
from forecasting.evaluate import mae, mape, rmse, directional_accuracy, plot_forecast_vs_actual
from forecasting.features import TimeSeriesFeatureEngineer

logger = logging.getLogger(__name__)
optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore", category=UserWarning, module="prophet")
warnings.filterwarnings("ignore", category=UserWarning, module="statsmodels")

_FEATURE_COLS_EXCLUDE = {"date", "retail_sales"}


def _get_feature_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in _FEATURE_COLS_EXCLUDE]


def _cv_score_lgbm(
    params: dict[str, Any],
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int,
) -> float:
    tscv = TimeSeriesSplit(n_splits=n_splits)
    fold_rmses: list[float] = []
    for train_idx, val_idx in tscv.split(X):
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]
        model = lgb.LGBMRegressor(**params)
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(period=-1)],
        )
        fold_rmses.append(rmse(y_val, model.predict(X_val)))
    return float(np.mean(fold_rmses))


def _build_optuna_objective(
    X: np.ndarray, y: np.ndarray, cfg: dict[str, Any]
) -> Any:
    n_splits = cfg["model"]["cv_folds"]

    def objective(trial: optuna.Trial) -> float:
        # Tighter search space for small datasets (~84 samples) to prevent overfitting
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 300),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 4, 24),          # was 15-127
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 40),  # was 5-50
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.1, 20.0, log=True),   # floor higher
            "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 20.0, log=True),
            "random_state": cfg["model"]["random_seed"],
            "verbosity": -1,
        }
        return _cv_score_lgbm(params, X, y, n_splits)

    return objective


def _train_lgbm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    cfg: dict[str, Any],
) -> tuple[lgb.LGBMRegressor, dict[str, float], dict[str, Any]]:
    n_trials = cfg["model"]["optuna_trials"]
    n_splits = cfg["model"]["cv_folds"]

    study = optuna.create_study(direction="minimize")
    study.optimize(
        _build_optuna_objective(X_train, y_train, cfg),
        n_trials=n_trials,
        show_progress_bar=False,
    )
    best_params = {**study.best_params, "random_state": cfg["model"]["random_seed"], "verbosity": -1}
    logger.info("Optuna best CV RMSE=%.4f  params=%s", study.best_value, best_params)

    tscv = TimeSeriesSplit(n_splits=n_splits)
    fold_rmses: list[float] = []
    for fold, (tr_idx, val_idx) in enumerate(tscv.split(X_train), 1):
        m = lgb.LGBMRegressor(**best_params)
        m.fit(X_train[tr_idx], y_train[tr_idx],
              eval_set=[(X_train[val_idx], y_train[val_idx])],
              callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(period=-1)])
        fold_rmse = rmse(y_train[val_idx], m.predict(X_train[val_idx]))
        fold_rmses.append(fold_rmse)
        mlflow.log_metric(f"lgbm_cv_rmse_fold{fold}", fold_rmse)

    final_model = lgb.LGBMRegressor(**best_params)
    final_model.fit(X_train, y_train)
    preds = final_model.predict(X_test)

    metrics = {
        "lgbm_rmse": rmse(y_test, preds),
        "lgbm_mae": mae(y_test, preds),
        "lgbm_mape": mape(y_test, preds),
        "lgbm_da": directional_accuracy(y_test, preds),
        "lgbm_cv_rmse_mean": float(np.mean(fold_rmses)),
    }
    return final_model, metrics, best_params


def _make_linear_pipeline(alpha_cfg: dict[str, Any], model_cls, **model_kwargs) -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", model_cls(**model_kwargs)),
    ])


def _train_ridge(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    cfg: dict[str, Any],
) -> tuple[Pipeline, dict[str, float]]:
    pipe = _make_linear_pipeline({}, Ridge, alpha=cfg["model"]["ridge"]["alpha"])
    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_test)
    metrics = {
        "ridge_rmse": rmse(y_test, preds),
        "ridge_mae": mae(y_test, preds),
        "ridge_mape": mape(y_test, preds),
        "ridge_da": directional_accuracy(y_test, preds),
    }
    return pipe, metrics


def _train_elasticnet(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    cfg: dict[str, Any],
) -> tuple[Pipeline, dict[str, float]]:
    en_cfg = cfg["model"].get("elasticnet", {"alpha": 1.0, "l1_ratio": 0.5})
    pipe = _make_linear_pipeline(
        {}, ElasticNet,
        alpha=en_cfg["alpha"],
        l1_ratio=en_cfg["l1_ratio"],
        max_iter=5000,
    )
    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_test)
    metrics = {
        "elasticnet_rmse": rmse(y_test, preds),
        "elasticnet_mae": mae(y_test, preds),
        "elasticnet_mape": mape(y_test, preds),
        "elasticnet_da": directional_accuracy(y_test, preds),
    }
    return pipe, metrics


def _train_sarima(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> tuple[Any, dict[str, float]]:
    """SARIMAX(1,1,1)(1,1,0)[4] — seasonal quarterly model."""
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    y_train = train_df["retail_sales"].values
    y_test = test_df["retail_sales"].values

    model = SARIMAX(
        y_train,
        order=(1, 1, 1),
        seasonal_order=(1, 1, 0, 4),
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    fit = model.fit(disp=False)
    forecast = fit.forecast(steps=len(y_test))
    preds = np.array(forecast)

    metrics = {
        "sarima_rmse": rmse(y_test, preds),
        "sarima_mae": mae(y_test, preds),
        "sarima_mape": mape(y_test, preds),
        "sarima_da": directional_accuracy(y_test, preds),
    }
    return fit, metrics, preds


def _train_holtwinters(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> tuple[Any, dict[str, float]]:
    """Holt-Winters additive seasonal (period=4 quarters)."""
    from statsmodels.tsa.holtwinters import ExponentialSmoothing

    y_train = train_df["retail_sales"].values
    y_test = test_df["retail_sales"].values

    model = ExponentialSmoothing(
        y_train,
        trend="add",
        seasonal="add",
        seasonal_periods=4,
        initialization_method="estimated",
    )
    fit = model.fit(optimized=True)
    preds = fit.forecast(len(y_test))

    metrics = {
        "hw_rmse": rmse(y_test, preds),
        "hw_mae": mae(y_test, preds),
        "hw_mape": mape(y_test, preds),
        "hw_da": directional_accuracy(y_test, preds),
    }
    return fit, metrics, preds


def _train_prophet(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> tuple[Any, dict[str, float]]:
    from prophet import Prophet

    prophet_train = train_df[["date", "retail_sales"]].rename(
        columns={"date": "ds", "retail_sales": "y"}
    )
    prophet_test = test_df[["date", "retail_sales"]].rename(
        columns={"date": "ds", "retail_sales": "y"}
    )

    model = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
    model.fit(prophet_train)
    forecast = model.predict(prophet_test[["ds"]])
    preds = forecast["yhat"].values
    actual = prophet_test["y"].values

    metrics = {
        "prophet_rmse": rmse(actual, preds),
        "prophet_mae": mae(actual, preds),
        "prophet_mape": mape(actual, preds),
        "prophet_da": directional_accuracy(actual, preds),
    }
    return model, metrics


def _train_stacking(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    base_preds_train: dict[str, np.ndarray],
    base_preds_test: dict[str, np.ndarray],
    cfg: dict[str, Any],
) -> tuple[Pipeline, dict[str, float], np.ndarray]:
    """Level-1 stacking: Ridge meta-learner on base model OOF predictions."""
    # Stack base model test predictions as meta-features
    meta_train = np.column_stack(list(base_preds_train.values()))
    meta_test = np.column_stack(list(base_preds_test.values()))

    meta_model = Pipeline([
        ("scaler", StandardScaler()),
        ("ridge", Ridge(alpha=1.0)),
    ])
    meta_model.fit(meta_train, y_train)
    preds = meta_model.predict(meta_test)

    metrics = {
        "stacking_rmse": rmse(y_test, preds),
        "stacking_mae": mae(y_test, preds),
        "stacking_mape": mape(y_test, preds),
        "stacking_da": directional_accuracy(y_test, preds),
    }
    return meta_model, metrics, preds


def _get_oof_predictions(
    pipe_or_model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_splits: int,
    is_sklearn: bool = True,
) -> np.ndarray:
    """Generate out-of-fold predictions for stacking meta-features."""
    tscv = TimeSeriesSplit(n_splits=n_splits)
    oof = np.zeros(len(y_train))
    for tr_idx, val_idx in tscv.split(X_train):
        if is_sklearn:
            import copy
            m = copy.deepcopy(pipe_or_model)
            m.fit(X_train[tr_idx], y_train[tr_idx])
            oof[val_idx] = m.predict(X_train[val_idx])
    return oof


def _print_leaderboard(results: dict[str, dict[str, float]]) -> None:
    """Print a formatted comparison table of all models."""
    header = f"\n{'Model':<16} {'RMSE':>8} {'MAE':>8} {'MAPE%':>8} {'DA%':>8}"
    separator = "-" * 54
    logger.info(header)
    logger.info(separator)
    rows = []
    for name, m in results.items():
        prefix = name.lower().replace(" ", "_").split("_")[0]
        rmse_key = next((k for k in m if k.endswith("_rmse")), None)
        mae_key = next((k for k in m if k.endswith("_mae")), None)
        mape_key = next((k for k in m if k.endswith("_mape")), None)
        da_key = next((k for k in m if k.endswith("_da")), None)
        if rmse_key:
            rows.append((m[rmse_key], name, m.get(rmse_key, 0), m.get(mae_key, 0),
                         m.get(mape_key, 0), m.get(da_key, 0) * 100 if da_key else 0))
    rows.sort(key=lambda x: x[0])
    for _, name, r, a, p, d in rows:
        logger.info(f"  {name:<14} {r:>8.1f} {a:>8.1f} {p:>7.2f}% {d:>7.1f}%")
    logger.info(separator)


def run_training() -> Path:
    """Full training pipeline: fetch data → engineer features → train all models → log → save."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    cfg = load_config()

    mlflow.set_tracking_uri(cfg["mlflow"]["tracking_uri"])
    mlflow.set_experiment(cfg["mlflow"]["experiment_name"])

    # --- Data ---------------------------------------------------------------
    logger.info("Loading data …")
    merged = build_merged_dataset(cfg["data"]["start_date"])

    # --- Features -----------------------------------------------------------
    fe = TimeSeriesFeatureEngineer(
        lag_months=cfg["features"]["lag_periods"],
        rolling_windows=cfg["features"]["rolling_windows"],
        nan_strategy=cfg["features"]["nan_fill_strategy"],
    )
    featured = fe.fit_transform(merged)

    test_months = cfg["model"]["test_size_months"]
    train_df = featured.iloc[:-test_months]
    test_df = featured.iloc[-test_months:]

    feature_cols = _get_feature_cols(featured)
    X_train = train_df[feature_cols].values
    y_train = train_df["retail_sales"].values
    X_test = test_df[feature_cols].values
    y_test = test_df["retail_sales"].values

    n_splits = cfg["model"]["cv_folds"]
    logger.info(
        "Dataset: %d train / %d test quarters | %d features",
        len(X_train), len(X_test), len(feature_cols),
    )

    all_metrics: dict[str, dict[str, float]] = {}

    with mlflow.start_run(run_name="full_training"):
        mlflow.log_params({
            "start_date": cfg["data"]["start_date"],
            "test_quarters": test_months,
            "n_features": len(feature_cols),
            "n_train": len(X_train),
        })

        # --- LightGBM -------------------------------------------------------
        logger.info("Training LightGBM (Optuna-tuned, tighter search space) …")
        lgbm_model, lgbm_metrics, best_params = _train_lgbm(
            X_train, y_train, X_test, y_test, cfg
        )
        mlflow.log_params({f"lgbm_{k}": v for k, v in best_params.items()})
        mlflow.log_metrics(lgbm_metrics)
        all_metrics["LightGBM"] = lgbm_metrics

        # --- Ridge ----------------------------------------------------------
        logger.info("Training Ridge baseline …")
        ridge_model, ridge_metrics = _train_ridge(X_train, y_train, X_test, y_test, cfg)
        mlflow.log_metrics(ridge_metrics)
        all_metrics["Ridge"] = ridge_metrics

        # --- ElasticNet -----------------------------------------------------
        logger.info("Training ElasticNet …")
        en_model, en_metrics = _train_elasticnet(X_train, y_train, X_test, y_test, cfg)
        mlflow.log_metrics(en_metrics)
        all_metrics["ElasticNet"] = en_metrics

        # --- SARIMA ---------------------------------------------------------
        logger.info("Training SARIMA(1,1,1)(1,1,0)[4] …")
        sarima_preds_test = np.zeros(len(y_test))
        try:
            sarima_model, sarima_metrics, sarima_preds_test = _train_sarima(train_df, test_df)
            mlflow.log_metrics(sarima_metrics)
            all_metrics["SARIMA"] = sarima_metrics
        except Exception as exc:
            logger.warning("SARIMA failed (skipping): %s", exc)
            sarima_model = None

        # --- Holt-Winters ---------------------------------------------------
        logger.info("Training Holt-Winters …")
        hw_preds_test = np.zeros(len(y_test))
        try:
            hw_model, hw_metrics, hw_preds_test = _train_holtwinters(train_df, test_df)
            mlflow.log_metrics(hw_metrics)
            all_metrics["Holt-Winters"] = hw_metrics
        except Exception as exc:
            logger.warning("Holt-Winters failed (skipping): %s", exc)
            hw_model = None

        # --- Prophet --------------------------------------------------------
        logger.info("Training Prophet …")
        try:
            prophet_model, prophet_metrics = _train_prophet(train_df, test_df)
            mlflow.log_metrics(prophet_metrics)
            all_metrics["Prophet"] = prophet_metrics
        except Exception as exc:
            logger.warning("Prophet failed (skipping): %s", exc)
            prophet_model = None

        # --- Stacking ensemble ----------------------------------------------
        logger.info("Building Stacking ensemble (Ridge meta-learner) …")
        try:
            lgbm_preds_test = lgbm_model.predict(X_test)
            ridge_preds_test = ridge_model.predict(X_test)
            en_preds_test = en_model.predict(X_test)

            # OOF for train meta-features
            import copy
            oof_lgbm = np.zeros(len(y_train))
            oof_ridge = np.zeros(len(y_train))
            oof_en = np.zeros(len(y_train))
            tscv = TimeSeriesSplit(n_splits=n_splits)
            for tr_i, val_i in tscv.split(X_train):
                m_lgbm = lgb.LGBMRegressor(**best_params)
                m_lgbm.fit(X_train[tr_i], y_train[tr_i])
                oof_lgbm[val_i] = m_lgbm.predict(X_train[val_i])

                m_ridge = copy.deepcopy(ridge_model)
                m_ridge.fit(X_train[tr_i], y_train[tr_i])
                oof_ridge[val_i] = m_ridge.predict(X_train[val_i])

                m_en = copy.deepcopy(en_model)
                m_en.fit(X_train[tr_i], y_train[tr_i])
                oof_en[val_i] = m_en.predict(X_train[val_i])

            base_preds_train = {"lgbm": oof_lgbm, "ridge": oof_ridge, "en": oof_en}
            base_preds_test = {
                "lgbm": lgbm_preds_test,
                "ridge": ridge_preds_test,
                "en": en_preds_test,
            }
            if sarima_model is not None:
                base_preds_test["sarima"] = sarima_preds_test
            if hw_model is not None:
                base_preds_test["hw"] = hw_preds_test

            # Align train dict keys with test dict keys
            base_preds_train_aligned = {k: base_preds_train[k] for k in base_preds_train if k in base_preds_test}
            base_preds_test_aligned = {k: base_preds_test[k] for k in base_preds_train if k in base_preds_test}

            stacking_model, stacking_metrics, stacking_preds = _train_stacking(
                X_train, y_train, X_test, y_test,
                base_preds_train_aligned, base_preds_test_aligned, cfg,
            )
            mlflow.log_metrics(stacking_metrics)
            all_metrics["Stacking"] = stacking_metrics
        except Exception as exc:
            logger.warning("Stacking failed (skipping): %s", exc)
            stacking_model = None

        # --- Artifacts ------------------------------------------------------
        fi_path = str(resolve_path("models") / "feature_importance.png")
        lgb.plot_importance(lgbm_model, max_num_features=20, figsize=(10, 6))
        import matplotlib.pyplot as plt
        plt.tight_layout()
        plt.savefig(fi_path, dpi=150)
        plt.close()
        mlflow.log_artifact(fi_path)

        lgbm_preds_test_final = lgbm_model.predict(X_test)
        forecast_fig = plot_forecast_vs_actual(
            test_df["date"], y_test, lgbm_preds_test_final,
            title="LightGBM — Test Set Forecast vs Actual"
        )
        fc_path = str(resolve_path("models") / "forecast_plot.png")
        forecast_fig.savefig(fc_path, dpi=150)
        plt.close()
        mlflow.log_artifact(fc_path)

        run_id = mlflow.active_run().info.run_id
        logger.info("MLflow run_id: %s", run_id)

        # --- Leaderboard ----------------------------------------------------
        _print_leaderboard(all_metrics)

    # --- Determine best model -----------------------------------------------
    # Use MAPE as ranking metric; pick best non-stacking model for production bundle
    mape_scores = {
        "lgbm": lgbm_metrics["lgbm_mape"],
        "ridge": ridge_metrics["ridge_mape"],
        "elasticnet": en_metrics["elasticnet_mape"],
    }
    if "SARIMA" in all_metrics:
        mape_scores["sarima"] = all_metrics["SARIMA"]["sarima_mape"]
    if "Holt-Winters" in all_metrics:
        mape_scores["hw"] = all_metrics["Holt-Winters"]["hw_mape"]
    best_name = min(mape_scores, key=mape_scores.get)
    logger.info("Best model by MAPE: %s (%.3f%%)", best_name, mape_scores[best_name])

    # Always include lgbm_model for feature importance / API compatibility
    bundle = {
        "model": lgbm_model,
        "ridge_model": ridge_model,
        "en_model": en_model,
        "feature_engineer": fe,
        "feature_cols": feature_cols,
        "config": cfg,
        "run_id": run_id,
        "metrics": {**lgbm_metrics, **ridge_metrics, **en_metrics},
        "all_metrics": all_metrics,
        "best_model_name": best_name,
    }
    model_path = resolve_path("models") / "best_model.joblib"
    joblib.dump(bundle, model_path)
    logger.info("Saved model bundle → %s", model_path)
    return model_path


if __name__ == "__main__":
    run_training()
