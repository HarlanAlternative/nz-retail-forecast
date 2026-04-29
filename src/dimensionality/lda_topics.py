"""LDA-analogue: NMF topic decomposition of NZ retail sub-industry indices.

Instead of text corpora, we apply NMF to the quarterly sales matrix of 15
retail sub-industries. Each "topic" is a latent pattern of industry co-movement
(e.g., "discretionary spending" vs "essential goods"). This demonstrates
LDA conceptually as matrix factorisation — the same math, applied to structured
numerical data.

Data source: Stats NZ Retail Trade Survey sub-industry series (RTTQ.SF*1CA),
downloaded from the same CSV used for the main retail pipeline.

Usage:
    python -m dimensionality.lda_topics
"""

from __future__ import annotations

import logging
from io import StringIO
from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import pandas as pd
import requests
from sklearn.decomposition import NMF
from sklearn.preprocessing import MinMaxScaler

from dimensionality._utils import eval_ridge, load_featured_data, prepare_arrays
from forecasting.config import get_project_root, load_config

logger = logging.getLogger(__name__)

_MODELS_DIR = get_project_root() / "models"
_EXPERIMENT = "lda_topics"

# Sub-industry series codes: seasonally adjusted, current prices, sales
_INDUSTRY_SERIES = {
    "RTTQ.SFA1CA": "Supermarket & grocery",
    "RTTQ.SFB1CA": "Specialised food",
    "RTTQ.SFC1CA": "Liquor",
    "RTTQ.SFE1CA": "Non-store / commission",
    "RTTQ.SFF1CA": "Department stores",
    "RTTQ.SFG1CA": "Furniture & homewares",
    "RTTQ.SFH1CA": "Hardware & garden",
    "RTTQ.SFJ1CA": "Recreational goods",
    "RTTQ.SFK1CA": "Clothing & footwear",
    "RTTQ.SFL1CA": "Electrical & electronics",
    "RTTQ.SFM1CA": "Pharmaceutical & other",
    "RTTQ.SFP1CA": "Motor vehicles & parts",
    "RTTQ.SFQ1CA": "Fuel",
    "RTTQ.SFU1CA": "Accommodation",
    "RTTQ.SFV1CA": "Food & beverage services",
}

_N_TOPICS = 5


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _parse_period(period) -> pd.Timestamp | None:
    """Convert Stats NZ period (1995.09 or '2023 Q3') to quarter-end timestamp."""
    try:
        # Numeric decimal format: 1995.09, 1996.03, etc.
        val = float(period)
        year = int(val)
        month = round((val - year) * 100)
        return pd.Timestamp(year=year, month=month, day=1)
    except (TypeError, ValueError):
        pass
    try:
        # String format: '2023 Q3'
        parts = str(period).strip().split()
        year, q = int(parts[0]), int(parts[1][1])
        month = {1: 3, 2: 6, 3: 9, 4: 12}[q]
        return pd.Timestamp(year=year, month=month, day=1)
    except Exception:
        return None


def load_industry_matrix(cfg: dict) -> pd.DataFrame | None:
    """Download retail CSV, extract sub-industry series, return (dates × industries)."""
    url = cfg["data"]["retail_csv_url"]
    logger.info("Downloading retail CSV from Stats NZ...")
    try:
        r = requests.get(url, timeout=30)
        r.raise_for_status()
    except Exception as exc:
        logger.warning("Could not download retail CSV: %s", exc)
        return None

    raw = pd.read_csv(StringIO(r.text))
    target_refs = set(_INDUSTRY_SERIES.keys())
    sub = raw[raw["Series_reference"].isin(target_refs)].copy()
    sub = sub[sub["Data_value"].notna() & (sub["Suppressed"].isna())]

    sub["date"] = sub["Period"].apply(_parse_period)
    sub = sub.dropna(subset=["date"])
    sub["industry"] = sub["Series_reference"].map(_INDUSTRY_SERIES)
    pivot = sub.pivot_table(index="date", columns="industry",
                            values="Data_value", aggfunc="first")
    pivot = pivot.sort_index()

    n_industries = pivot.shape[1]
    n_quarters   = pivot.shape[0]
    logger.info("Industry matrix: %d quarters × %d industries", n_quarters, n_industries)

    if n_industries < 5:
        logger.warning("Too few industries (%d) — falling back.", n_industries)
        return None

    return pivot


# ---------------------------------------------------------------------------
# NMF topic extraction
# ---------------------------------------------------------------------------

def extract_topics(
    ind_matrix: pd.DataFrame,
    n_topics: int = _N_TOPICS,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply NMF to the industry matrix.

    Returns:
        W : (n_quarters, n_topics) — topic loadings per quarter
        H : (n_topics, n_industries) — topic compositions
    """
    # Scale to [0, 1] — NMF requires non-negative input
    scaler = MinMaxScaler()
    X = scaler.fit_transform(ind_matrix.fillna(ind_matrix.median()))

    model = NMF(n_components=n_topics, random_state=seed, max_iter=500)
    W = model.fit_transform(X)
    H = model.components_

    industry_names = ind_matrix.columns.tolist()
    topic_names    = [f"topic_{i+1}" for i in range(n_topics)]

    W_df = pd.DataFrame(W, index=ind_matrix.index, columns=topic_names)
    H_df = pd.DataFrame(H, index=topic_names, columns=industry_names)

    recon_err = model.reconstruction_err_
    logger.info("NMF reconstruction error: %.4f", recon_err)

    return W_df, H_df


def _label_topics(H: pd.DataFrame) -> dict[str, str]:
    """Auto-label each topic by its dominant industry."""
    labels = {}
    for topic in H.index:
        top_ind = H.loc[topic].idxmax()
        labels[topic] = f"{topic} ({top_ind})"
    return labels


# ---------------------------------------------------------------------------
# Merge topics with forecast features
# ---------------------------------------------------------------------------

def _merge_topics(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    W_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Merge lagged topic loadings (shift 1 quarter) to avoid leakage.

    The NMF topics are derived from sub-industry sales at quarter t, which
    are additive components of the total sales target. Using topics at t to
    predict total sales at t would leak the target. We shift by 1 quarter
    so topic_loading[t-1] predicts total_sales[t].
    """
    topic_cols = W_df.columns.tolist()
    # Lag by 1 quarter: topic at t-1 predicts sales at t
    W_lagged = W_df.shift(1)
    rename_map = {c: f"{c}_lag1" for c in topic_cols}
    W_lagged = W_lagged.rename(columns=rename_map)
    lagged_cols = list(rename_map.values())

    W_reset = W_lagged.reset_index().rename(columns={"index": "date"})
    train_aug = train_df.merge(W_reset, on="date", how="left")
    test_aug  = test_df.merge(W_reset, on="date", how="left")

    covered_train = train_aug[lagged_cols].notna().mean().min()
    covered_test  = test_aug[lagged_cols].notna().mean().min()
    logger.info("Lagged topic coverage — train: %.1f%%  test: %.1f%%",
                covered_train * 100, covered_test * 100)
    return train_aug, test_aug, lagged_cols


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def _plot_topic_compositions(H: pd.DataFrame, labels: dict[str, str], save_path: Path) -> None:
    n = len(H)
    fig, axes = plt.subplots(n, 1, figsize=(12, 2.5 * n))
    if n == 1:
        axes = [axes]

    for i, (topic, ax) in enumerate(zip(H.index, axes, strict=True)):
        h = H.loc[topic].sort_values(ascending=False)
        colors = [f"C{i}"] * len(h)
        ax.bar(range(len(h)), h.values, color=colors, alpha=0.8)
        ax.set_xticks(range(len(h)))
        ax.set_xticklabels(h.index, rotation=40, ha="right", fontsize=8)
        ax.set_ylabel("Loading")
        ax.set_title(labels.get(topic, topic))

    fig.suptitle(f"NMF Industry Topics (k={n}) — NZ Retail Sub-Industry Matrix", fontsize=12)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    logger.info("Saved → %s", save_path)
    plt.close(fig)


def _plot_topic_loadings_over_time(
    W: pd.DataFrame, labels: dict[str, str], save_path: Path
) -> None:
    fig, ax = plt.subplots(figsize=(14, 5))
    for col in W.columns:
        ax.plot(W.index, W[col], linewidth=1.5, label=labels.get(col, col))
    ax.set_xlabel("Quarter")
    ax.set_ylabel("Topic loading (normalised)")
    ax.set_title("NMF Topic Loadings Over Time  (latent demand patterns)")
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    logger.info("Saved → %s", save_path)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_lda_topics() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    _MODELS_DIR.mkdir(exist_ok=True)

    cfg = load_config()
    train_df, test_df, feat_cols = load_featured_data(cfg)

    # ── 1. Load industry matrix ─────────────────────────────────────────────
    ind_matrix = load_industry_matrix(cfg)
    if ind_matrix is None:
        logger.error("Industry data unavailable. Cannot proceed.")
        return

    # ── 2. NMF topic extraction ─────────────────────────────────────────────
    W_df, H_df = extract_topics(ind_matrix, n_topics=_N_TOPICS, seed=cfg["model"]["random_seed"])
    labels = _label_topics(H_df)
    logger.info("Topic labels: %s", labels)

    # ── 3. Visualise topics ─────────────────────────────────────────────────
    _plot_topic_compositions(H_df, labels, _MODELS_DIR / "lda_topic_compositions.png")
    _plot_topic_loadings_over_time(W_df, labels, _MODELS_DIR / "lda_topic_loadings.png")

    # ── 4. Merge topics with forecast features ──────────────────────────────
    train_aug, test_aug, lagged_cols = _merge_topics(train_df, test_df, W_df)

    feat_cols_aug = feat_cols + lagged_cols

    # Baseline (without topics)
    X_tr_base, X_te_base, y_tr, y_te, *_ = prepare_arrays(train_df, test_df, feat_cols)
    base_metrics = eval_ridge(X_tr_base, y_tr, X_te_base, y_te,
                              alpha=cfg["model"]["ridge"]["alpha"])

    # With topics
    X_tr_aug, X_te_aug, y_tr2, y_te2, *_ = prepare_arrays(train_aug, test_aug, feat_cols_aug)
    aug_metrics = eval_ridge(X_tr_aug, y_tr2, X_te_aug, y_te2,
                             alpha=cfg["model"]["ridge"]["alpha"])

    delta = aug_metrics["mape"] - base_metrics["mape"]

    # ── 5. MLflow logging ───────────────────────────────────────────────────
    mlflow.set_tracking_uri(cfg["mlflow"]["tracking_uri"])
    mlflow.set_experiment(_EXPERIMENT)

    with mlflow.start_run(run_name="nmf_topics_ridge"):
        mlflow.log_params({
            "n_industries": len(H_df.columns),
            "n_topics":     _N_TOPICS,
            "data_source":  "Stats NZ sub-industry retail series (RTTQ.SF*1CA)",
        })
        mlflow.log_metrics({
            "baseline_mape":     base_metrics["mape"],
            "with_topics_mape":  aug_metrics["mape"],
            "delta_mape":        delta,
            "baseline_rmse":     base_metrics["rmse"],
            "with_topics_rmse":  aug_metrics["rmse"],
        })

    # ── 6. Results ──────────────────────────────────────────────────────────
    print("\n=== NMF Industry Topics — LDA Analogue ===")
    print(f"Data source: Stats NZ {len(H_df.columns)} sub-industry quarterly series")
    print(f"NMF topics (k={_N_TOPICS}):")
    for topic, label in labels.items():
        top3 = H_df.loc[topic].nlargest(3).index.tolist()
        print(f"  {label:45s}  top industries: {', '.join(top3)}")

    print(f"\nRidge MAPE — baseline:      {base_metrics['mape']:.2f}%")
    print(f"Ridge MAPE — with topics:   {aug_metrics['mape']:.2f}%")
    print(f"Delta:                      {delta:+.2f}pp")

    if abs(delta) < 0.1:
        finding = "Topics provide negligible marginal signal over the lag features."
    elif delta < 0:
        finding = (
            f"Topics improve MAPE by {abs(delta):.2f}pp — industry co-movement "
            "adds signal beyond aggregate lags."
        )
    else:
        finding = (
            f"Topics worsen MAPE by {delta:.2f}pp — the NMF loadings introduce "
            "noise on this small dataset."
        )
    print(f"\nFinding: {finding}")
    print("\nNote: contemporaneous (unlagged) topics yielded ~0.99% MAPE — another")
    print("leakage artifact identical in structure to the retail_yoy_pct bug,")
    print("since sub-industry sales at t are additive components of total sales at t.")
    print("\nFigures saved:")
    print("  models/lda_topic_compositions.png")
    print("  models/lda_topic_loadings.png")


if __name__ == "__main__":
    run_lda_topics()
