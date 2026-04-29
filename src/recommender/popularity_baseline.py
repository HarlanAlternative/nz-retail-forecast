"""Popularity (most-popular-items) baseline recommender.

This is the honesty gate: if no sophisticated model beats it by a meaningful
margin, the sophisticated model has a bug or there is no collaborative signal.

Items are ranked by total log(1+quantity) weight across all training users.
For each test user, we return the top-K popular items that the user has not
already interacted with in the training set.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp


class PopularityRecommender:
    def __init__(self) -> None:
        self._popular_order: np.ndarray | None = None
        self._train: sp.csr_matrix | None = None

    def fit(self, train_matrix: sp.csr_matrix) -> PopularityRecommender:
        self._train = train_matrix
        item_scores = np.asarray(train_matrix.sum(axis=0)).flatten()
        self._popular_order = np.argsort(item_scores)[::-1]
        return self

    def recommend(self, user_idx: int, k: int = 10) -> list[int]:
        assert self._popular_order is not None, "Call fit() first"
        seen = set(self._train.getrow(user_idx).indices.tolist())
        recs: list[int] = []
        for item in self._popular_order:
            if item not in seen:
                recs.append(int(item))
            if len(recs) == k:
                break
        return recs

    @property
    def top_items(self) -> np.ndarray:
        return self._popular_order  # type: ignore[return-value]
