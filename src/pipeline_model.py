from __future__ import annotations
"""
Dimensionality reduction module applying PCA.

- Reduce 30 features to 15 principal components
- Calculate explained variance ratios and cumulative variance
- Handle feature scaling integration (operate on standardized features)
- Persist fitted PCA to disk
- Return reduced train/test datasets
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from .utils import setup_logger
from . import config

logger = setup_logger(__name__)

try:
    from sklearn.decomposition import PCA
except Exception as e:  # pragma: no cover
    logger.error("scikit-learn is required for PCA. Please install requirements. Error: %s", e)
    raise

try:
    import joblib  # type: ignore
    _JOBLIB_AVAILABLE = True
except Exception:
    import pickle
    _JOBLIB_AVAILABLE = False


def _save_pca(pca: PCA, filepath: str) -> None:
    if _JOBLIB_AVAILABLE:
        joblib.dump(pca, filepath)
    else:
        with open(filepath, "wb") as f:
            pickle.dump(pca, f)


def _load_pca(filepath: str) -> PCA:
    if _JOBLIB_AVAILABLE:
        return joblib.load(filepath)
    else:
        import pickle
        with open(filepath, "rb") as f:
            return pickle.load(f)


@dataclass
class PCAResult:
    X_train_reduced: np.ndarray
    X_test_reduced: np.ndarray
    explained_variance_ratio: np.ndarray
    cumulative_variance: float
    n_components: int


class PCAReducer:
    def __init__(self, n_components: int = config.PCA_COMPONENTS):
        self.n_components = n_components
        self.pca: Optional[PCA] = None

    def fit_transform(self, X_train: np.ndarray) -> np.ndarray:
        """Fit PCA on training data and transform it."""
        logger.info(f"Fitting PCA with n_components={self.n_components}")
        pca = PCA(n_components=self.n_components, random_state=config.RANDOM_STATE)
        X_train_reduced = pca.fit_transform(X_train)
        self.pca = pca
        cum_var = float(np.cumsum(pca.explained_variance_ratio_)[-1])
        logger.info(
            f"PCA fit complete. Cumulative explained variance ({self.n_components} comps): {cum_var:.4f}"
        )
        # Persist PCA
        try:
            import os
            os.makedirs(os.path.dirname(config.PCA_FILE), exist_ok=True)
            _save_pca(pca, config.PCA_FILE)
            logger.info(f"Saved PCA model to {config.PCA_FILE}")
        except Exception as e:  # pragma: no cover
            logger.warning(f"Failed to save PCA: {e}")
        return X_train_reduced

    def transform(self, X_test: np.ndarray) -> np.ndarray:
        """Transform test data using fitted PCA."""
        if self.pca is None:
            raise RuntimeError("PCA is not fitted. Call fit_transform on training data first.")
        return self.pca.transform(X_test)

    def run(self, X_train: np.ndarray, X_test: np.ndarray) -> PCAResult:
        """Convenience method to fit on train and transform test, returning PCAResult."""
        X_train_reduced = self.fit_transform(X_train)
        X_test_reduced = self.transform(X_test)
        evr = self.pca.explained_variance_ratio_ if self.pca is not None else np.array([])
        cum_var = float(np.cumsum(evr)[-1]) if evr.size > 0 else 0.0
        return PCAResult(
            X_train_reduced=X_train_reduced,
            X_test_reduced=X_test_reduced,
            explained_variance_ratio=evr,
            cumulative_variance=cum_var,
            n_components=self.n_components,
        )
