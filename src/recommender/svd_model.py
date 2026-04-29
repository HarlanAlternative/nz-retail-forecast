"""TruncatedSVD matrix-factorisation recommender.

Factorises the log(1+qty) user-item matrix as X ≈ U Σ Vᵀ using sklearn's
randomised SVD.  User latent vectors = U·Σ^0.5; item latent vectors = Vᵀ·Σ^0.5.
Recommendations for user u = argmax(user_vec_u @ item_vecs.T), excluding items
seen in training.

This is the same mathematical engine used by:
  - Netflix collaborative filtering (SVD)
  - Word2Vec skip-gram via PMI matrix SVD
  - The dimensionality analysis in src/dimensionality/svd_analysis.py

Difference from that module: here we factor the full user-item matrix (not just
the feature matrix), and use the result for ranking rather than regression.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from sklearn.decomposition import TruncatedSVD


class SVDRecommender:
    """Collaborative filtering via truncated SVD (randomised LAPACK).

    Args:
        n_components: Latent dimension k. Typical range: 20-200.
        random_state: For reproducibility of the randomised SVD.
    """

    def __init__(self, n_components: int = 50, random_state: int = 42) -> None:
        self.n_components = n_components
        self._svd = TruncatedSVD(n_components=n_components, random_state=random_state)
        self._user_factors: np.ndarray | None = None
        self._item_factors: np.ndarray | None = None
        self._train: sp.csr_matrix | None = None

    def fit(self, train_matrix: sp.csr_matrix) -> SVDRecommender:
        self._train = train_matrix
        U_sigma = self._svd.fit_transform(train_matrix)      # (n_users, k)
        sigma_half = np.sqrt(self._svd.singular_values_)
        self._user_factors = U_sigma / sigma_half            # absorb Σ^0.5
        self._item_factors = (self._svd.components_.T        # (n_items, k)
                              * sigma_half)
        return self

    def recommend(self, user_idx: int, k: int = 10) -> list[int]:
        assert self._user_factors is not None, "Call fit() first"
        scores = self._user_factors[user_idx] @ self._item_factors.T
        seen = self._train.getrow(user_idx).indices
        scores[seen] = -np.inf
        return np.argsort(scores)[::-1][:k].tolist()

    @property
    def explained_variance_ratio(self) -> np.ndarray:
        return self._svd.explained_variance_ratio_
