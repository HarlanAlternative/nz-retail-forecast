"""Ranking evaluation metrics for recommender systems.

Metrics:
    Recall@K  — was the test item in the top-K list?
    NDCG@K    — rank-discounted relevance (log2 discount)
    MAP@K     — precision averaged over positions

All metrics assume a single relevant item per user (leave-one-out protocol).
"""

from __future__ import annotations

import logging
from collections.abc import Callable

import numpy as np

logger = logging.getLogger(__name__)


def recall_at_k(recommended: list[int], test_item: int, k: int) -> float:
    return 1.0 if test_item in recommended[:k] else 0.0


def ndcg_at_k(recommended: list[int], test_item: int, k: int) -> float:
    if test_item in recommended[:k]:
        rank = recommended.index(test_item) + 1
        return 1.0 / np.log2(rank + 1)
    return 0.0


def average_precision_at_k(recommended: list[int], test_item: int, k: int) -> float:
    if test_item in recommended[:k]:
        rank = recommended.index(test_item) + 1
        return 1.0 / rank
    return 0.0


def evaluate_model(
    recommend_fn: Callable[[int, int], list[int]],
    test_items: dict[int, int],
    k: int = 10,
) -> dict[str, float]:
    """Evaluate a recommender across all test users.

    Args:
        recommend_fn: Callable(user_idx, k) → list of item indices (top-k)
        test_items  : {user_row_idx → held-out item_col_idx}
        k           : cut-off rank

    Returns:
        Dict with recall@k, ndcg@k, map@k, n_users
    """
    recalls, ndcgs, maps = [], [], []
    errors = 0
    for u_idx, test_item in test_items.items():
        try:
            recs = recommend_fn(u_idx, k)
        except Exception:
            errors += 1
            recs = []
        recalls.append(recall_at_k(recs, test_item, k))
        ndcgs.append(ndcg_at_k(recs, test_item, k))
        maps.append(average_precision_at_k(recs, test_item, k))

    if errors:
        logger.warning("Recommendation errors for %d / %d users", errors, len(test_items))

    return {
        f"recall@{k}": float(np.mean(recalls)),
        f"ndcg@{k}": float(np.mean(ndcgs)),
        f"map@{k}": float(np.mean(maps)),
        "n_users": len(test_items),
    }


def evaluate_all(
    models: dict[str, Callable[[int, int], list[int]]],
    test_items: dict[int, int],
    k_values: list[int] = (10, 20, 50),
) -> dict[str, dict[str, float]]:
    """Evaluate multiple models at multiple K values.

    Returns:
        {model_name: {metric: value, ...}}
    """
    results: dict[str, dict[str, float]] = {}
    for name, fn in models.items():
        logger.info("Evaluating %s ...", name)
        row: dict[str, float] = {}
        for k in k_values:
            m = evaluate_model(fn, test_items, k=k)
            row.update(m)
        results[name] = row
        logger.info(
            "  %s  Recall@10=%.3f  NDCG@10=%.3f  MAP@10=%.3f",
            name,
            row.get("recall@10", 0),
            row.get("ndcg@10", 0),
            row.get("map@10", 0),
        )
    return results
