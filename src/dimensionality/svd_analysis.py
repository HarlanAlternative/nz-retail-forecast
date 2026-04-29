"""SVD analysis of the lag-feature matrix.

Questions answered:
1. What is the intrinsic dimensionality of the retail lag-feature space?
2. Which singular vectors correspond to interpretable temporal patterns?
3. How many SVD components are needed to preserve Ridge forecast accuracy?

Usage:
    python -m dimensionality.svd_analysis
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import svd as scipy_svd
from sklearn.preprocessing import StandardScaler

from dimensionality._utils import (
    eval_ridge,
    load_featured_data,
    retail_feature_cols,
)
from forecasting.config import get_project_root, load_config

logger = logging.getLogger(__name__)

_MODELS_DIR = get_project_root() / "models"

_K_VALUES = [1, 2, 3, 5, 10, "full"]

_COMPONENT_LABELS = {
    0: "Level (baseline demand)",
    1: "Trend / momentum",
    2: "Seasonal oscillation",
}


# ---------------------------------------------------------------------------
# Core SVD helpers
# ---------------------------------------------------------------------------

def compute_svd(X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Full economy SVD: X = U S Vt  (no full_matrices padding)."""
    U, S, Vt = scipy_svd(X, full_matrices=False)
    return U, S, Vt


def explained_variance(S: np.ndarray) -> np.ndarray:
    ev = S**2
    return ev / ev.sum()


def reconstruct(U: np.ndarray, S: np.ndarray, Vt: np.ndarray, k: int) -> np.ndarray:
    return U[:, :k] @ np.diag(S[:k]) @ Vt[:k, :]


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def _plot_spectrum(S: np.ndarray, save_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    ax = axes[0]
    ax.semilogy(range(1, len(S) + 1), S, "o-", color="#1f77b4", markersize=5)
    ax.set_xlabel("Component index")
    ax.set_ylabel("Singular value (log scale)")
    ax.set_title("Singular Value Spectrum")
    ax.axvline(x=3, color="red", linestyle="--", alpha=0.6, label="k=3 elbow")
    ax.legend()

    ax2 = axes[1]
    cum_ev = np.cumsum(explained_variance(S)) * 100
    ax2.plot(range(1, len(S) + 1), cum_ev, "s-", color="#ff7f0e", markersize=5)
    ax2.axhline(y=90, color="gray", linestyle="--", label="90% threshold")
    ax2.axhline(y=95, color="red",  linestyle="--", label="95% threshold")
    k90 = int(np.searchsorted(cum_ev, 90)) + 1
    k95 = int(np.searchsorted(cum_ev, 95)) + 1
    ax2.axvline(x=k90, color="gray", linestyle=":", alpha=0.6)
    ax2.axvline(x=k95, color="red",  linestyle=":", alpha=0.6)
    ax2.set_xlabel("Number of components k")
    ax2.set_ylabel("Cumulative explained variance (%)")
    ax2.set_title(f"Cumulative Variance  (90%→k={k90}, 95%→k={k95})")
    ax2.legend()

    fig.suptitle("SVD of NZ Retail Lag-Feature Matrix  X ∈ ℝⁿˣᵖ", fontsize=13)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    logger.info("Saved → %s", save_path)
    plt.close(fig)


def _plot_right_singular_vectors(Vt: np.ndarray, feat_cols: list[str], save_path: Path) -> None:
    n_show = min(3, len(Vt))
    fig, axes = plt.subplots(n_show, 1, figsize=(12, 3 * n_show))
    if n_show == 1:
        axes = [axes]

    for k in range(n_show):
        ax = axes[k]
        v = Vt[k]
        colors = ["#2196F3" if x >= 0 else "#FF5722" for x in v]
        ax.bar(range(len(v)), v, color=colors)
        ax.set_xticks(range(len(v)))
        ax.set_xticklabels(feat_cols, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("Loading")
        label = _COMPONENT_LABELS.get(k, f"Component {k+1}")
        ax.set_title(f"Right singular vector {k+1}: {label}")
        ax.axhline(0, color="black", linewidth=0.5)

    fig.suptitle("Latent Temporal Patterns in NZ Retail Feature Space", fontsize=13)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    logger.info("Saved → %s", save_path)
    plt.close(fig)


def _plot_reconstruction_ablation(
    k_values: list,
    mapes: list[float],
    baseline_mape: float,
    save_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(9, 4))
    labels = [str(k) for k in k_values]
    x_pos = range(len(k_values))
    bars = ax.bar(x_pos, mapes, color="#2196F3", alpha=0.8)
    ax.axhline(baseline_mape, color="#FF5722", linestyle="--",
               linewidth=2, label=f"Full-feature Ridge baseline ({baseline_mape:.2f}%)")
    ax.set_xticks(list(x_pos))
    ax.set_xticklabels(labels)
    ax.set_xlabel("SVD rank k (number of components)")
    ax.set_ylabel("MAPE (%)")
    ax.set_title("Ridge MAPE vs SVD Reconstruction Rank\n"
                 "(does low-rank approximation preserve forecast accuracy?)")
    ax.legend()
    for bar, val in zip(bars, mapes, strict=True):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                f"{val:.2f}%", ha="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    logger.info("Saved → %s", save_path)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_svd_analysis() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    _MODELS_DIR.mkdir(exist_ok=True)

    cfg = load_config()
    train_df, test_df, _ = load_featured_data(cfg)

    feat_cols = retail_feature_cols(train_df)
    logger.info("Retail-intrinsic features: %d  %s", len(feat_cols), feat_cols)

    # ── 1. Scale (center + unit variance) ──────────────────────────────────
    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_df[feat_cols].values)
    X_test  = scaler.transform(test_df[feat_cols].values)
    y_train = train_df["retail_sales"].values
    y_test  = test_df["retail_sales"].values

    p = X_train.shape[1]
    logger.info("X_train shape: %s  (n=%d, p=%d)", X_train.shape, *X_train.shape)

    # ── 2. Full SVD ─────────────────────────────────────────────────────────
    U, S, Vt = compute_svd(X_train)
    ev = explained_variance(S)
    cum_ev = np.cumsum(ev) * 100

    logger.info("Singular values: %s", np.round(S, 2))
    logger.info("Explained variance per component: %s",
                [f"{v*100:.1f}%" for v in ev])
    k90 = int(np.searchsorted(cum_ev, 90)) + 1
    k95 = int(np.searchsorted(cum_ev, 95)) + 1
    logger.info("Components for 90%% variance: %d  |  95%%: %d", k90, k95)

    # ── 3. Figures: spectrum + singular vectors ─────────────────────────────
    _plot_spectrum(S, _MODELS_DIR / "svd_spectrum.png")
    _plot_right_singular_vectors(Vt, feat_cols, _MODELS_DIR / "svd_right_vectors.png")

    # ── 4. Full-feature Ridge baseline ─────────────────────────────────────
    ridge_alpha = cfg["model"]["ridge"]["alpha"]
    full_metrics = eval_ridge(X_train, y_train, X_test, y_test, alpha=ridge_alpha)
    logger.info("Full-feature Ridge (retail cols only) MAPE=%.2f%%", full_metrics["mape"])

    # ── 5. Reconstruction ablation ─────────────────────────────────────────
    ablation_mapes = []
    print(f"\n{'k':>6}  {'MAPE%':>8}  {'RMSE':>8}  {'vs full':>10}")
    print("-" * 40)

    for k in _K_VALUES:
        k_int = p if k == "full" else int(k)

        # Reconstruct X_train in the SVD subspace and project X_test
        X_tr_k = reconstruct(U, S, Vt, k_int)
        # For test: project into the same rank-k subspace
        X_te_k = X_test @ Vt[:k_int].T @ Vt[:k_int]

        m = eval_ridge(X_tr_k, y_train, X_te_k, y_test, alpha=cfg["model"]["ridge"]["alpha"])
        delta = m["mape"] - full_metrics["mape"]
        ablation_mapes.append(m["mape"])
        print(f"{str(k):>6}  {m['mape']:>7.2f}%  {m['rmse']:>8.1f}  {delta:>+9.2f}pp")

    _plot_reconstruction_ablation(
        _K_VALUES, ablation_mapes, full_metrics["mape"],
        _MODELS_DIR / "svd_reconstruction_ablation.png",
    )

    # ── 6. Summary ──────────────────────────────────────────────────────────
    print(f"\nIntrinsic dimensionality: {k90} components explain 90% variance")
    print("Top-3 singular vectors interpreted as:")
    for k, label in _COMPONENT_LABELS.items():
        dom = feat_cols[int(np.argmax(np.abs(Vt[k])))]
        print(f"  SV{k+1}: {label:30s} (dominant feature: {dom})")
    print("\nNote: SVD is the mathematical engine behind PCA, latent-factor")
    print("recommenders (SVD collaborative filtering), and truncated SVD in NLP.")
    print("See src/dimensionality/pca_features.py for the PCA ablation.")


if __name__ == "__main__":
    run_svd_analysis()
