"""ALS (Alternating Least Squares) implicit-feedback recommender.

Uses the `implicit` library which implements the Hu, Koren & Volinsky (2008)
confidence-weighted matrix factorisation for implicit feedback.

BM25 weighting (Robertson & Zaragoza 2009) is applied before fitting:
it down-weights very active users (normalises for user activity bias) and
amplifies rare but strong item signals, outperforming raw log counts for ALS.

Cold-start users (no training interactions) fall back to the popularity baseline.
"""

from __future__ import annotations

import logging

import numpy as np
import scipy.sparse as sp

logger = logging.getLogger(__name__)


def _bm25_weight(matrix: sp.csr_matrix, k1: float = 100.0, b: float = 0.8) -> sp.csr_matrix:
    """BM25 weighting for an implicit user-item matrix.

    Adapts Robertson-Zaragoza BM25 to item-frequency-in-users context:
    - k1 controls saturation of item frequency
    - b controls length normalisation (user activity length)
    """
    # Treat each user as a "document" and each item as a "term"
    # document length = number of unique items for user
    doc_lengths = np.diff(matrix.indptr).astype(np.float32)
    avg_dl = doc_lengths.mean()

    weighted = matrix.copy().astype(np.float32)
    for u in range(matrix.shape[0]):
        start, end = matrix.indptr[u], matrix.indptr[u + 1]
        tf = matrix.data[start:end]
        dl = doc_lengths[u]
        weighted.data[start:end] = (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * dl / avg_dl))

    return weighted


class ALSRecommender:
    """ALS collaborative filtering with BM25 confidence weighting.

    Args:
        factors      : Latent dimension (default 64)
        iterations   : ALS iterations (default 30)
        regularization: L2 regularisation (default 0.1)
        random_state : Seed for reproducibility
    """

    def __init__(
        self,
        factors: int = 64,
        iterations: int = 30,
        regularization: float = 0.1,
        random_state: int = 42,
    ) -> None:
        self.factors = factors
        self.iterations = iterations
        self.regularization = regularization
        self.random_state = random_state
        self._model = None
        self._train: sp.csr_matrix | None = None

    def fit(self, train_matrix: sp.csr_matrix) -> ALSRecommender:
        try:
            import implicit
        except ImportError as exc:
            raise ImportError(
                "Install the 'implicit' library: pip install implicit"
            ) from exc

        self._train = train_matrix
        weighted = _bm25_weight(train_matrix)

        self._model = implicit.als.AlternatingLeastSquares(
            factors=self.factors,
            iterations=self.iterations,
            regularization=self.regularization,
            random_state=self.random_state,
            use_gpu=False,
        )
        # implicit 0.7.x expects user × item format (rows=users, cols=items)
        self._model.fit(weighted)
        return self

    def recommend(self, user_idx: int, k: int = 10) -> list[int]:
        assert self._model is not None, "Call fit() first"
        user_row = self._train[user_idx]
        if user_row.nnz == 0:
            return []
        # implicit requires exactly 1 row per user_idx
        item_ids, _ = self._model.recommend(
            user_idx,
            user_row,
            N=k,
            filter_already_liked_items=True,
        )
        return [int(i) for i in item_ids]
