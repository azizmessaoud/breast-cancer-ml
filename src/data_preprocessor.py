from __future__ import annotations
"""
Preprocessing module for the Wisconsin Breast Cancer dataset.

Provides the DataPreprocessor class with methods:
- remove_id_column()
- encode_diagnosis()
- split_features_target()
- train_test_split_data()
- standardize_features()

Handles missing values and persists the fitted StandardScaler for inference.
"""

from dataclasses import dataclass
from typing import Tuple, Optional

import numpy as np
import pandas as pd

from .utils import setup_logger
from . import config

logger = setup_logger(__name__)

try:
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
except Exception as e:  # pragma: no cover
    logger.error(
        "scikit-learn is required for preprocessing. Please install requirements. Error: %s",
        e,
    )
    raise

try:
    import joblib  # type: ignore
    _JOBLIB_AVAILABLE = True
except Exception:
    import pickle
    _JOBLIB_AVAILABLE = False


def _save_scaler(scaler: StandardScaler, filepath: str) -> None:
    if _JOBLIB_AVAILABLE:
        joblib.dump(scaler, filepath)
    else:
        with open(filepath, "wb") as f:
            pickle.dump(scaler, f)


def _load_scaler(filepath: str) -> StandardScaler:
    if _JOBLIB_AVAILABLE:
        return joblib.load(filepath)
    else:
        import pickle
        with open(filepath, "rb") as f:
            return pickle.load(f)


@dataclass
class SplitData:
    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray


class DataPreprocessor:
    def __init__(self, df: pd.DataFrame):
        """Initialize with a DataFrame.
        Args:
            df: Raw or cleaned DataFrame
        """
        self.df = df.copy()
        self.scaler: Optional[StandardScaler] = None

    def remove_id_column(self) -> pd.DataFrame:
        """Remove ID/irrelevant columns if present.
        Returns:
            DataFrame without optional ID columns
        """
        drop_cols = [c for c in [config.ID_COLUMN, "Unnamed: 32", "id"] if c in self.df.columns]
        if drop_cols:
            logger.info(f"Removing ID columns: {drop_cols}")
            self.df = self.df.drop(columns=drop_cols)
        return self.df

    def encode_diagnosis(self) -> pd.DataFrame:
        """Encode diagnosis column: M->1, B->0.
        Returns:
            DataFrame with encoded diagnosis
        """
        if config.DIAGNOSIS_COLUMN not in self.df.columns:
            raise ValueError(f"Diagnosis column '{config.DIAGNOSIS_COLUMN}' not found")
        col = self.df[config.DIAGNOSIS_COLUMN].astype(str).str.strip().str.upper()
        mapping = {"M": 1, "B": 0}
        if not set(col.unique()).issubset(set(mapping.keys())):
            raise ValueError("Diagnosis column contains unexpected values. Expected only 'M' or 'B'.")
        self.df[config.DIAGNOSIS_COLUMN] = col.map(mapping).astype(int)
        logger.info("Encoded diagnosis: M->1, B->0")
        return self.df

    def split_features_target(self) -> Tuple[pd.DataFrame, pd.Series]:
        """Split DataFrame into features (X) and target (y)."""
        if config.DIAGNOSIS_COLUMN not in self.df.columns:
            raise ValueError(f"Diagnosis column '{config.DIAGNOSIS_COLUMN}' not found")
        X = self.df.drop(columns=[config.DIAGNOSIS_COLUMN])
        y = self.df[config.DIAGNOSIS_COLUMN]
        # Ensure all features numeric
        non_numeric = X.select_dtypes(exclude=[np.number]).columns.tolist()
        if non_numeric:
            raise ValueError(f"Non-numeric feature columns present: {non_numeric}")
        return X, y

    def train_test_split_data(self, X: pd.DataFrame, y: pd.Series) -> SplitData:
        """Stratified 70-30 train/test split."""
        X_train, X_test, y_train, y_test = train_test_split(
            X.values, y.values, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE, stratify=y.values
        )
        logger.info(
            f"Performed stratified train/test split: train={X_train.shape}, test={X_test.shape}"
        )
        return SplitData(X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)

    def standardize_features(self, split: SplitData, save_path: Optional[str] = None) -> SplitData:
        """Standardize features using StandardScaler fitted on training data.
        Args:
            split: SplitData with raw features
            save_path: Optional path to persist the fitted scaler (defaults to config.SCALER_FILE)
        Returns:
            SplitData with standardized features
        """
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(split.X_train)
        X_test_scaled = scaler.transform(split.X_test)
        self.scaler = scaler
        path = save_path or config.SCALER_FILE
        try:
            # Ensure directory exists
            import os
            os.makedirs(os.path.dirname(path), exist_ok=True)
            _save_scaler(scaler, path)
            logger.info(f"Saved StandardScaler to {path}")
        except Exception as e:  # pragma: no cover
            logger.warning(f"Failed to save scaler: {e}")
        return SplitData(X_train=X_train_scaled, X_test=X_test_scaled, y_train=split.y_train, y_test=split.y_test)

    @staticmethod
    def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
        """Robust missing value handling (median for numerics; ffill/bfill for objects)."""
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        for col in num_cols:
            if df[col].isna().any():
                median = df[col].median()
                df[col] = df[col].fillna(median)
        obj_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
        for col in obj_cols:
            if df[col].isna().any():
                df[col] = df[col].ffill().bfill()
        return df

