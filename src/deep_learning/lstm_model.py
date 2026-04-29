"""PyTorch LSTM and GRU forecasters for quarterly time series.

Both models share the same interface: feed a (batch, seq_len, input_dim)
tensor and receive a (batch, 1) point forecast.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset

# ---------------------------------------------------------------------------
# Sequence dataset
# ---------------------------------------------------------------------------

class SequenceDataset(Dataset):
    """Sliding-window dataset for supervised sequence forecasting.

    At index i the sample is:
        X : feature matrix rows [i, i+seq_len)  — shape (seq_len, n_features)
        y : target at row i+seq_len              — scalar
    """

    def __init__(self, X: np.ndarray, y: np.ndarray, seq_len: int) -> None:
        seqs, targets = make_sequences(X, y, seq_len)
        self.X = torch.tensor(seqs, dtype=torch.float32)
        self.y = torch.tensor(targets, dtype=torch.float32).unsqueeze(1)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx]


def make_sequences(
    X: np.ndarray, y: np.ndarray, seq_len: int
) -> tuple[np.ndarray, np.ndarray]:
    """Create overlapping windows for sequence modelling.

    Args:
        X: Feature matrix (n_samples, n_features), leakage-free at each row.
        y: Target vector (n_samples,).
        seq_len: Number of time steps per input window.

    Returns:
        sequences : (n_samples - seq_len + 1, seq_len, n_features)
        targets   : (n_samples - seq_len + 1,)
    """
    n = len(X)
    seqs = np.stack([X[i : i + seq_len] for i in range(n - seq_len + 1)])
    tgts = y[seq_len - 1 :]
    return seqs, tgts


# ---------------------------------------------------------------------------
# Early stopping
# ---------------------------------------------------------------------------

class EarlyStopping:
    """Stop training when validation loss stops improving."""

    def __init__(self, patience: int = 20, min_delta: float = 1e-4) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float("inf")
        self.counter = 0
        self.best_state: dict | None = None

    def step(self, val_loss: float, model: nn.Module) -> bool:
        """Return True if training should stop."""
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            self.counter += 1
        return self.counter >= self.patience

    def restore_best(self, model: nn.Module) -> None:
        if self.best_state is not None:
            model.load_state_dict(self.best_state)


# ---------------------------------------------------------------------------
# Model definitions
# ---------------------------------------------------------------------------

class LSTMForecaster(nn.Module):
    """Single- or multi-layer LSTM regression head.

    Args:
        input_dim: Number of input features per time step.
        hidden_dim: LSTM hidden state size.
        num_layers: Number of stacked LSTM layers.
        dropout: Dropout probability between LSTM layers (ignored if num_layers=1).
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 1,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, input_dim)
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :])  # use last time step


class GRUForecaster(nn.Module):
    """Single- or multi-layer GRU regression head.

    Same interface as LSTMForecaster.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 1,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.gru(x)
        return self.head(out[:, -1, :])
