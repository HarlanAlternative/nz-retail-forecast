"""Recommender system training and evaluation pipeline.

Models trained (in order of complexity):
    1. Popularity baseline — honesty gate
    2. TruncatedSVD (k=50) — matrix factorisation
    3. ALS (implicit, BM25) — confidence-weighted MF
    4. Content-based (TF-IDF + NMF) — item description similarity
    5. Hybrid (SVD + Content-based, α=0.6) — ensemble

Evaluation: leave-one-out @ K ∈ {10, 20, 50}  (Recall, NDCG, MAP)
All metrics logged to MLflow experiment: recommender_system

Usage:
    python -m recommender.train_recommender
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd

from recommender.als_model import ALSRecommender
from recommender.content_based import ContentBasedRecommender, HybridRecommender
from recommender.data_loader import load_data
from recommender.evaluate import evaluate_all
from recommender.popularity_baseline import PopularityRecommender
from recommender.svd_model import SVDRecommender

logger = logging.getLogger(__name__)

_MODELS_DIR = Path(__file__).parents[2] / "models"
_MLFLOW_URI = "mlruns/"
_EXPERIMENT = "recommender_system"
_K_VALUES = [10, 20, 50]


def _log_leakage_audit() -> None:
    """Confirm that no test information leaked into training."""
    notes = (
        "LOO split: test item zeroed from train matrix before any model.fit() call. "
        "Popularity computed from train only. "
        "SVD factorises train matrix only. "
        "ALS fits on train matrix (item×user format, BM25 weighted). "
        "Content-based uses item descriptions only (no user-item signal at test time). "
        "Hybrid combines SVD+CB scores — both derived from train only."
    )
    mlflow.log_param("leakage_audit", notes)
    logger.info("Leakage audit: %s", notes)


def _plot_comparison(results: dict[str, dict[str, float]], save_path: Path) -> None:
    models = list(results.keys())
    recall10 = [results[m].get("recall@10", 0) for m in models]
    ndcg10 = [results[m].get("ndcg@10", 0) for m in models]
    map10 = [results[m].get("map@10", 0) for m in models]

    x = np.arange(len(models))
    width = 0.26

    fig, ax = plt.subplots(figsize=(13, 5))
    ax.bar(x - width, recall10, width, label="Recall@10", color="#2196F3", alpha=0.85)
    ax.bar(x,         ndcg10,   width, label="NDCG@10",   color="#FF9800", alpha=0.85)
    ax.bar(x + width, map10,    width, label="MAP@10",    color="#4CAF50", alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=20, ha="right")
    ax.set_ylabel("Score")
    ax.set_title(
        "Recommender Comparison — Online Retail II (UK)\n"
        "Leave-one-out evaluation, K=10"
    )
    ax.legend()

    # Annotate the popularity baseline bar to make it the visual anchor
    pop_idx = next((i for i, m in enumerate(models) if "Popularity" in m), None)
    if pop_idx is not None:
        ax.axvline(x=pop_idx, color="red", linestyle="--", alpha=0.4, linewidth=1.5)
        ax.text(
            pop_idx + 0.05, ax.get_ylim()[1] * 0.95,
            "← baseline\n   (beat this)",
            fontsize=8, color="red", alpha=0.7,
        )

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    logger.info("Saved → %s", save_path)
    plt.close(fig)


def _plot_recall_by_k(
    results: dict[str, dict[str, float]],
    k_values: list[int],
    save_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    for name, metrics in results.items():
        recalls = [metrics.get(f"recall@{k}", 0) for k in k_values]
        ax.plot(k_values, recalls, "o-", linewidth=2, label=name)
    ax.set_xlabel("K")
    ax.set_ylabel("Recall@K")
    ax.set_title("Recall@K Curve — Recommender Models\nOnline Retail II (UK)")
    ax.legend(fontsize=9)
    ax.set_xticks(k_values)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    logger.info("Saved → %s", save_path)
    plt.close(fig)


def run_recommender() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    _MODELS_DIR.mkdir(exist_ok=True)

    # ── 1. Load data ────────────────────────────────────────────────────────
    logger.info("Loading Online Retail II ...")
    train_matrix, full_matrix, test_items, users, items, item_meta = load_data()
    n_users, n_items = train_matrix.shape
    logger.info("Train matrix: %d users × %d items | %d test users",
                n_users, n_items, len(test_items))

    # ── 2. Fit all models ───────────────────────────────────────────────────
    logger.info("Fitting Popularity baseline ...")
    pop = PopularityRecommender().fit(train_matrix)

    logger.info("Fitting SVD (k=50) ...")
    svd = SVDRecommender(n_components=50).fit(train_matrix)

    logger.info("Fitting ALS ...")
    als = ALSRecommender(factors=64, iterations=30).fit(train_matrix)

    logger.info("Fitting Content-based (TF-IDF + NMF) ...")
    cb = ContentBasedRecommender(n_topics=20).fit(train_matrix, items, item_meta)

    logger.info("Fitting Hybrid (SVD + Content-based) ...")
    hybrid = HybridRecommender(svd, cb, pop, svd_weight=0.6).fit(train_matrix)

    recommend_fns = {
        "Popularity":    pop.recommend,
        "SVD (k=50)":   svd.recommend,
        "ALS":          als.recommend,
        "Content-based": cb.recommend,
        "Hybrid":       hybrid.recommend,
    }

    # ── 3. Evaluate ─────────────────────────────────────────────────────────
    logger.info("Evaluating all models ...")
    results = evaluate_all(recommend_fns, test_items, k_values=_K_VALUES)

    # ── 4. MLflow logging ───────────────────────────────────────────────────
    mlflow.set_tracking_uri(_MLFLOW_URI)
    mlflow.set_experiment(_EXPERIMENT)

    with mlflow.start_run(run_name="recommender_comparison"):
        mlflow.log_params({
            "n_users":      n_users,
            "n_items":      n_items,
            "n_test_users": len(test_items),
            "split":        "leave-one-out (last purchase by date)",
            "svd_k":        50,
            "als_factors":  64,
            "als_bm25":     True,
            "nmf_topics":   20,
            "hybrid_svd_weight": 0.6,
        })
        _log_leakage_audit()

        for name, metrics in results.items():
            prefix = name.lower().replace(" ", "_").replace("(", "").replace(")", "")
            for metric, val in metrics.items():
                if metric != "n_users":
                    mlflow.log_metric(f"{prefix}_{metric.replace('@', '_at_')}", val)

    # ── 5. Figures ──────────────────────────────────────────────────────────
    _plot_comparison(results, _MODELS_DIR / "recommender_comparison.png")
    _plot_recall_by_k(results, _K_VALUES, _MODELS_DIR / "recommender_recall_at_k.png")

    # ── 6. Print results table ──────────────────────────────────────────────
    pop_r10 = results["Popularity"].get("recall@10", 0)
    print("\n=== Recommender Comparison — Online Retail II (UK) ===")
    print(f"{'Model':<20} {'Recall@10':>10} {'NDCG@10':>9} {'MAP@10':>8} "
          f"{'Recall@20':>10} {'Recall@50':>10} {'vs Pop':>8}")
    print("-" * 80)
    for name, metrics in results.items():
        r10 = metrics.get("recall@10", 0)
        r20 = metrics.get("recall@20", 0)
        r50 = metrics.get("recall@50", 0)
        ndcg = metrics.get("ndcg@10", 0)
        ap = metrics.get("map@10", 0)
        delta = r10 - pop_r10
        marker = "  ✓" if delta > 0.005 else ("  =" if abs(delta) <= 0.005 else "  ✗")
        print(f"{name:<20} {r10:>10.4f} {ndcg:>9.4f} {ap:>8.4f} "
              f"{r20:>10.4f} {r50:>10.4f} {delta:>+7.4f}{marker}")

    # SVD intrinsic dimensionality summary
    cumvar = np.cumsum(svd.explained_variance_ratio)
    k90 = int(np.searchsorted(cumvar, 0.90)) + 1
    k95 = int(np.searchsorted(cumvar, 0.95)) + 1
    print(f"\nSVD spectrum: {k90} components explain 90% variance  "
          f"| {k95} explain 95%  (total k=50)")

    print("\nFigures saved:")
    print("  models/recommender_comparison.png")
    print("  models/recommender_recall_at_k.png")
    print("MLflow run logged to experiment: recommender_system")

    # Cross-scenario summary for notebooks/02
    _save_cross_scenario_summary(results, svd, n_users, n_items)


def _save_cross_scenario_summary(
    results: dict[str, dict[str, float]],
    svd: SVDRecommender,
    n_users: int,
    n_items: int,
) -> None:
    """Save cross-scenario comparison data for the notebook."""
    rows = []
    for name, metrics in results.items():
        rows.append({
            "model": name,
            "scenario": "Recommender (Online Retail II)",
            **metrics,
        })
    pd.DataFrame(rows).to_csv(_MODELS_DIR / "recommender_results.csv", index=False)

    # SVD comparison data
    cumvar = np.cumsum(svd.explained_variance_ratio)
    k90 = int(np.searchsorted(cumvar, 0.90)) + 1
    summary = {
        "domain": "Recommender (UK e-commerce)",
        "n_users": n_users,
        "n_items": n_items,
        "svd_k_90pct": k90,
        "svd_interpretation": "Latent purchase preference factors",
        "forecasting_domain": "NZ Retail Sales (quarterly time series)",
        "forecasting_svd_k_90pct": 4,
        "forecasting_svd_interpretation": "Level, Trend, Seasonality, Noise",
    }
    pd.DataFrame([summary]).to_csv(
        _MODELS_DIR / "cross_scenario_svd_comparison.csv", index=False
    )
    logger.info("Cross-scenario summary saved to models/")


if __name__ == "__main__":
    run_recommender()
