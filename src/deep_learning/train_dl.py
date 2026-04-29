"""Train LSTM and GRU baselines on NZ retail sales and log to MLflow.

Usage:
    python -m deep_learning.train_dl
"""

from __future__ import annotations

import logging
import warnings
from typing import Any

import mlflow
import numpy as np
import optuna
import torch
import torch.nn as nn
from sklearn.impute import SimpleImputer
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

from deep_learning.lstm_model import (
    EarlyStopping,
    GRUForecaster,
    LSTMForecaster,
    SequenceDataset,
    make_sequences,
)
from forecasting.config import load_config
from forecasting.data import build_merged_dataset
from forecasting.evaluate import directional_accuracy, mae, mape, rmse
from forecasting.features import TimeSeriesFeatureEngineer

logger = logging.getLogger(__name__)
optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore")

_EXCLUDE_COLS = {"date", "retail_sales"}
_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------

def _get_feature_cols(df) -> list[str]:
    return [c for c in df.columns if c not in _EXCLUDE_COLS]


def prepare_data(cfg: dict[str, Any]):
    """Load data, engineer features, scale, return train/test arrays."""
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

    feat_cols = _get_feature_cols(featured)

    # Fit imputer + scaler on training data only
    imputer  = SimpleImputer(strategy="median")
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    X_train_raw = train_df[feat_cols].values
    y_train_raw = train_df["retail_sales"].values.reshape(-1, 1)
    X_test_raw  = test_df[feat_cols].values
    y_test_raw  = test_df["retail_sales"].values.reshape(-1, 1)

    X_train = scaler_X.fit_transform(imputer.fit_transform(X_train_raw))
    X_test  = scaler_X.transform(imputer.transform(X_test_raw))
    y_train = scaler_y.fit_transform(y_train_raw).ravel()
    y_test  = scaler_y.transform(y_test_raw).ravel()

    return X_train, y_train, X_test, y_test, scaler_y, test_df["retail_sales"].values


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------

def _train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
) -> float:
    model.train()
    total_loss = 0.0
    for xb, yb in loader:
        xb, yb = xb.to(_DEVICE), yb.to(_DEVICE)
        optimizer.zero_grad()
        loss = criterion(model(xb), yb)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item() * len(xb)
    return total_loss / len(loader.dataset)


def _eval_loss(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
) -> float:
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(_DEVICE), yb.to(_DEVICE)
            total_loss += criterion(model(xb), yb).item() * len(xb)
    return total_loss / len(loader.dataset)


def _predict(model: nn.Module, loader: DataLoader) -> np.ndarray:
    model.eval()
    preds = []
    with torch.no_grad():
        for xb, _ in loader:
            preds.append(model(xb.to(_DEVICE)).cpu().numpy())
    return np.concatenate(preds).ravel()


def train_model(
    model_cls: type[nn.Module],
    input_dim: int,
    params: dict[str, Any],
    train_ds: SequenceDataset,
    val_ds: SequenceDataset | None,
    max_epochs: int,
    patience: int,
) -> nn.Module:
    """Train model with early stopping. Returns model with best val weights."""
    model = model_cls(
        input_dim=input_dim,
        hidden_dim=params["hidden_dim"],
        num_layers=params["num_layers"],
        dropout=params["dropout"],
    ).to(_DEVICE)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=params["lr"],
        weight_decay=params.get("weight_decay", 1e-4),
    )
    criterion = nn.MSELoss()
    batch_size = params.get("batch_size", 16)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False)
    val_loader   = DataLoader(val_ds, batch_size=len(val_ds), shuffle=False) if val_ds else None

    stopper = EarlyStopping(patience=patience)

    for epoch in range(max_epochs):
        _train_epoch(model, train_loader, optimizer, criterion)
        if val_loader:
            val_loss = _eval_loss(model, val_loader, criterion)
            if stopper.step(val_loss, model):
                logger.debug("Early stop at epoch %d (val_loss=%.5f)", epoch, val_loss)
                break

    if val_loader:
        stopper.restore_best(model)
    return model


# ---------------------------------------------------------------------------
# Walk-forward CV for Optuna
# ---------------------------------------------------------------------------

def _wf_cv_score(
    model_cls: type[nn.Module],
    params: dict[str, Any],
    X_train: np.ndarray,
    y_train: np.ndarray,
    seq_len: int,
    n_splits: int,
    max_epochs: int,
    patience: int,
) -> float:
    """Walk-forward cross-validation RMSE (scaled space)."""
    tscv = TimeSeriesSplit(n_splits=n_splits)
    input_dim = X_train.shape[1]
    fold_rmses: list[float] = []

    for tr_idx, val_idx in tscv.split(X_train):
        X_tr, X_val = X_train[tr_idx], X_train[val_idx]
        y_tr, y_val = y_train[tr_idx], y_train[val_idx]

        if len(X_tr) <= seq_len or len(X_val) <= seq_len:
            continue

        tr_ds  = SequenceDataset(X_tr, y_tr, seq_len)
        val_ds = SequenceDataset(X_val, y_val, seq_len)
        if len(tr_ds) == 0 or len(val_ds) == 0:
            continue

        model = train_model(
            model_cls, input_dim, params, tr_ds, val_ds, max_epochs, patience
        )
        val_loader = DataLoader(val_ds, batch_size=len(val_ds))
        preds = _predict(model, val_loader)
        _, val_targets = make_sequences(X_val, y_val, seq_len)
        fold_rmses.append(float(np.sqrt(np.mean((val_targets - preds) ** 2))))

    return float(np.mean(fold_rmses)) if fold_rmses else float("inf")


# ---------------------------------------------------------------------------
# Optuna objective builder
# ---------------------------------------------------------------------------

def _build_objective(
    model_cls: type[nn.Module],
    X_train: np.ndarray,
    y_train: np.ndarray,
    cfg: dict[str, Any],
):
    dl_cfg    = cfg["deep_learning"]
    n_splits  = cfg["model"]["cv_folds"]
    max_epochs = dl_cfg["max_epochs"]
    patience  = dl_cfg["patience"]

    def objective(trial: optuna.Trial) -> float:
        params = {
            "hidden_dim":   trial.suggest_categorical("hidden_dim", [32, 64, 128]),
            "num_layers":   trial.suggest_int("num_layers", 1, 2),
            "dropout":      trial.suggest_float("dropout", 0.1, 0.5),
            "lr":           trial.suggest_float("lr", 1e-4, 1e-2, log=True),
            "weight_decay": trial.suggest_float("weight_decay", 1e-5, 1e-2, log=True),
            "batch_size":   trial.suggest_categorical("batch_size", [8, 16, 32]),
        }
        seq_len = trial.suggest_categorical("seq_len", dl_cfg["sequence_lengths"])
        return _wf_cv_score(
            model_cls, params, X_train, y_train,
            seq_len, n_splits, max_epochs, patience,
        )

    return objective


# ---------------------------------------------------------------------------
# Main training run
# ---------------------------------------------------------------------------

def _run_one_model(
    name: str,
    model_cls: type[nn.Module],
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test_scaled: np.ndarray,
    y_test_actual: np.ndarray,
    scaler_y: StandardScaler,
    cfg: dict[str, Any],
) -> dict[str, float]:
    dl_cfg = cfg["deep_learning"]
    n_trials = cfg["model"]["optuna_trials"]

    logger.info("Optuna search for %s (%d trials) ...", name, n_trials)
    study = optuna.create_study(direction="minimize")
    study.optimize(_build_objective(model_cls, X_train, y_train, cfg), n_trials=n_trials)

    best = study.best_params
    best_seq_len = best.pop("seq_len")
    logger.info("%s best CV RMSE=%.4f  seq_len=%d  params=%s",
                name, study.best_value, best_seq_len, best)

    # Final model on full training set
    tr_ds   = SequenceDataset(X_train, y_train, best_seq_len)
    # Use last 20% of training as validation for early stopping
    n_val   = max(best_seq_len + 1, len(tr_ds) // 5)
    tr_only = SequenceDataset(X_train[:-n_val], y_train[:-n_val], best_seq_len)
    val_ds  = SequenceDataset(X_train[-n_val:], y_train[-n_val:], best_seq_len)

    model = train_model(
        model_cls, X_train.shape[1], best,
        tr_only, val_ds,
        dl_cfg["max_epochs"], dl_cfg["patience"],
    )

    # Need test sequences that include the last seq_len-1 training rows as context
    X_full = np.vstack([X_train, X_test])
    y_full = np.concatenate([y_train, y_test_scaled])
    _, _ = make_sequences(X_full, y_full, best_seq_len)

    # Build test-only sequences using tail of training as context
    context_X = X_train[-(best_seq_len - 1):]
    context_y = y_train[-(best_seq_len - 1):]
    X_ext = np.vstack([context_X, X_test])
    y_ext = np.concatenate([context_y, y_test_scaled])
    te_ds = SequenceDataset(X_ext, y_ext, best_seq_len)

    te_loader = DataLoader(te_ds, batch_size=len(te_ds))
    preds_scaled = _predict(model, te_loader)
    preds_actual = scaler_y.inverse_transform(preds_scaled.reshape(-1, 1)).ravel()

    metrics = {
        f"{name}_rmse": rmse(y_test_actual, preds_actual),
        f"{name}_mae":  mae(y_test_actual, preds_actual),
        f"{name}_mape": mape(y_test_actual, preds_actual),
        f"{name}_da":   directional_accuracy(y_test_actual, preds_actual),
        f"{name}_cv_rmse": study.best_value,
        f"{name}_seq_len": float(best_seq_len),
        f"{name}_hidden_dim": float(best["hidden_dim"]),
    }
    return metrics


def run_dl_training() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    cfg = load_config()

    X_train, y_train, X_test, y_test_scaled, scaler_y, y_test_actual = prepare_data(cfg)
    input_dim = X_train.shape[1]
    logger.info("Dataset: %d train / %d test sequences  |  %d features  |  device: %s",
                len(X_train), len(X_test), input_dim, _DEVICE)

    mlflow.set_tracking_uri(cfg["mlflow"]["tracking_uri"])
    mlflow.set_experiment(cfg["deep_learning"]["experiment_name"])

    all_metrics: dict[str, float] = {}

    with mlflow.start_run(run_name="lstm_gru_comparison"):
        for name, cls in [("lstm", LSTMForecaster), ("gru", GRUForecaster)]:
            logger.info("Training %s ...", name.upper())
            m = _run_one_model(
                name, cls,
                X_train, y_train, X_test, y_test_scaled, y_test_actual, scaler_y, cfg,
            )
            all_metrics.update(m)
            mlflow.log_metrics(m)
            logger.info(
                "  [%s]  MAPE=%.2f%%  RMSE=%.1f  DA=%.1f%%",
                name.upper(), m[f"{name}_mape"], m[f"{name}_rmse"], m[f"{name}_da"] * 100,
            )

        mlflow.log_param("device", str(_DEVICE))
        mlflow.log_param("train_quarters", len(X_train))
        mlflow.log_param("test_quarters", len(X_test))
        mlflow.log_param("feature_dim", input_dim)

    print("\n=== Deep Learning Results ===")
    print(f"  LSTM  MAPE={all_metrics['lstm_mape']:.2f}%  RMSE={all_metrics['lstm_rmse']:.1f}  "
          f"DA={all_metrics['lstm_da']*100:.1f}%  (seq_len={int(all_metrics['lstm_seq_len'])})")
    print(f"  GRU   MAPE={all_metrics['gru_mape']:.2f}%  RMSE={all_metrics['gru_rmse']:.1f}  "
          f"DA={all_metrics['gru_da']*100:.1f}%  (seq_len={int(all_metrics['gru_seq_len'])})")
    print("\nExpected: LSTM/GRU will NOT beat Ridge (2.58%) — small-sample linear regime.")
    print("See notebooks/01_lstm_vs_ridge.ipynb for the full scientific narrative.")


if __name__ == "__main__":
    run_dl_training()
