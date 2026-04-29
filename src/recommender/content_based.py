"""Content-based recommender using TF-IDF + NMF (LDA analogue).

Architecture:
    Item descriptions → TF-IDF (5000 vocab) → NMF (20 latent topics)
    → normalised topic vectors per item

For each user: aggregate seen-item topic vectors → user content profile.
Recommendations = items with highest cosine similarity to user profile,
excluding already-seen items.

Connection to the dimensionality module: NMF here decomposes a term-document
matrix (items × TF-IDF terms), while lda_topics.py decomposes a
time×industries matrix. Same NMF maths, different inputs.

Cold-start handling: users with no training interactions get an empty list.
Callers can fall back to the popularity baseline for cold-start users.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer

logger = logging.getLogger(__name__)

_TFIDF_MAX_FEATURES = 5_000
_NMF_TOPICS = 20


class ContentBasedRecommender:
    """TF-IDF + NMF content-based recommender.

    Args:
        n_topics    : Number of NMF latent topics (default 20)
        max_features: TF-IDF vocabulary size (default 5000)
        random_state: Seed for NMF initialisation
    """

    def __init__(
        self,
        n_topics: int = _NMF_TOPICS,
        max_features: int = _TFIDF_MAX_FEATURES,
        random_state: int = 42,
    ) -> None:
        self.n_topics = n_topics
        self._tfidf = TfidfVectorizer(
            max_features=max_features,
            stop_words="english",
            ngram_range=(1, 2),
            min_df=2,
        )
        self._nmf = NMF(n_components=n_topics, random_state=random_state, max_iter=500)
        self._item_topics: np.ndarray | None = None      # (n_items, n_topics), normalised
        self._train: sp.csr_matrix | None = None

    def fit(
        self,
        train_matrix: sp.csr_matrix,
        items: list,
        item_meta: pd.DataFrame,
    ) -> ContentBasedRecommender:
        self._train = train_matrix

        desc_map: dict = dict(
            zip(item_meta["StockCode"], item_meta["Description"], strict=False)
        )
        descriptions = [str(desc_map.get(it, "")) for it in items]

        tfidf_matrix = self._tfidf.fit_transform(descriptions)
        raw_topics = self._nmf.fit_transform(tfidf_matrix)  # (n_items, n_topics)

        # L2-normalise for cosine similarity
        norms = np.linalg.norm(raw_topics, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self._item_topics = raw_topics / norms

        recon_err = self._nmf.reconstruction_err_
        logger.info(
            "Content-based NMF: %d items × %d topics  recon_err=%.4f",
            len(items), self.n_topics, recon_err,
        )
        return self

    def recommend(self, user_idx: int, k: int = 10) -> list[int]:
        assert self._item_topics is not None, "Call fit() first"
        row = self._train.getrow(user_idx)
        seen = row.indices

        if len(seen) == 0:
            return []

        # User profile: interaction-weighted mean of seen item topic vectors
        weights = row.data
        user_profile = (self._item_topics[seen] * weights[:, None]).sum(axis=0)
        norm = np.linalg.norm(user_profile)
        if norm == 0:
            return []
        user_profile /= norm

        scores = self._item_topics @ user_profile
        scores[seen] = -np.inf
        return np.argsort(scores)[::-1][:k].tolist()

    @property
    def topic_word_matrix(self) -> np.ndarray:
        """H matrix: (n_topics, vocab_size)"""
        return self._nmf.components_

    @property
    def feature_names(self) -> list[str]:
        return self._tfidf.get_feature_names_out().tolist()


class HybridRecommender:
    """Weighted hybrid: α·SVD_score + (1-α)·ContentBased_score.

    Falls back to popularity for cold-start users (no training interactions).

    Args:
        svd_model     : Fitted SVDRecommender
        cb_model      : Fitted ContentBasedRecommender
        popularity    : Fitted PopularityRecommender (cold-start fallback)
        svd_weight    : Weight for SVD scores (content weight = 1 - svd_weight)
    """

    def __init__(self, svd_model, cb_model, popularity, svd_weight: float = 0.6) -> None:
        self._svd = svd_model
        self._cb = cb_model
        self._pop = popularity
        self._svd_w = svd_weight
        self._cb_w = 1.0 - svd_weight
        self._train: sp.csr_matrix | None = None
        self._n_items: int = 0

    def fit(self, train_matrix: sp.csr_matrix) -> HybridRecommender:
        self._train = train_matrix
        self._n_items = train_matrix.shape[1]
        return self

    def recommend(self, user_idx: int, k: int = 10) -> list[int]:
        row = self._train.getrow(user_idx)
        seen = set(row.indices.tolist())

        if len(seen) == 0:
            return self._pop.recommend(user_idx, k)

        # Compute normalised scores from each model independently
        svd_scores = self._score_svd(user_idx)
        cb_scores = self._score_cb(user_idx)

        combined = self._svd_w * svd_scores + self._cb_w * cb_scores
        for s in seen:
            combined[s] = -np.inf

        return np.argsort(combined)[::-1][:k].tolist()

    def _score_svd(self, user_idx: int) -> np.ndarray:
        uf = self._svd._user_factors
        itf = self._svd._item_factors
        raw = uf[user_idx] @ itf.T
        rng = raw.max() - raw.min()
        return (raw - raw.min()) / (rng if rng > 0 else 1.0)

    def _score_cb(self, user_idx: int) -> np.ndarray:
        row = self._train.getrow(user_idx)
        seen_idx = row.indices
        if len(seen_idx) == 0:
            return np.zeros(self._n_items)
        weights = row.data
        item_topics = self._cb._item_topics
        user_profile = (item_topics[seen_idx] * weights[:, None]).sum(axis=0)
        norm = np.linalg.norm(user_profile)
        if norm == 0:
            return np.zeros(self._n_items)
        user_profile /= norm
        raw = item_topics @ user_profile
        rng = raw.max() - raw.min()
        return (raw - raw.min()) / (rng if rng > 0 else 1.0)
