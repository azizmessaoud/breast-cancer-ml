from __future__ import annotations
"""
Data loading and EDA module for the Wisconsin Breast Cancer dataset.

Responsibilities:
- Load CSV dataset safely with validation
- Clean and validate data (missing values, types)
- Exploratory data analysis (class distribution, correlations)
- Return processed pandas DataFrame
- Provide logging and error handling

Usage:
    python -m src.data_loader --data breast-cancer-detection/data/data.csv --out breast-cancer-detection/outputs/eda
"""

import argparse
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .utils import setup_logger, safe_mkdir
from . import config

logger = setup_logger(__name__)

REQUIRED_COLUMNS: List[str] = [config.DIAGNOSIS_COLUMN]
OPTIONAL_ID_COLUMNS: List[str] = [config.ID_COLUMN, "Unnamed: 32", "id"]


def load_dataset(csv_path: str) -> pd.DataFrame:
    """Load the dataset from a CSV file.

    Args:
        csv_path: Path to the CSV file.
    Returns:
        DataFrame with raw data.
    Raises:
        FileNotFoundError: if the path does not exist.
        ValueError: if the CSV is empty or cannot be read.
    """
    logger.info(f"Loading dataset from: {csv_path}")
    if not os.path.exists(csv_path):
        logger.error(f"Dataset not found at {csv_path}")
        raise FileNotFoundError(f"Dataset not found at {csv_path}")
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        logger.exception("Failed to read CSV")
        raise ValueError(f"Failed to read CSV: {e}")

    if df.empty:
        logger.error("Loaded dataset is empty")
        raise ValueError("Loaded dataset is empty")
    logger.info(f"Dataset shape: {df.shape}")
    return df


def _drop_optional_id_columns(df: pd.DataFrame) -> pd.DataFrame:
    drop_cols = [c for c in OPTIONAL_ID_COLUMNS if c in df.columns]
    if drop_cols:
        logger.info(f"Dropping optional ID columns: {drop_cols}")
        df = df.drop(columns=drop_cols)
    return df


def _validate_columns(df: pd.DataFrame) -> None:
    for col in REQUIRED_COLUMNS:
        if col not in df.columns:
            logger.error(f"Required column missing: {col}")
            raise ValueError(f"Required column missing: {col}")
    # Ensure at least 30 feature columns (excluding diagnosis)
    feature_cols = [c for c in df.columns if c != config.DIAGNOSIS_COLUMN]
    if len(feature_cols) < 30:
        logger.warning(
            f"Expected at least 30 feature columns, found {len(feature_cols)}. Proceeding, but check dataset."
        )


def _clean_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    # Drop rows without diagnosis
    before = len(df)
    df = df.dropna(subset=[config.DIAGNOSIS_COLUMN])
    after = len(df)
    if after < before:
        logger.info(f"Dropped {before - after} rows with missing diagnosis")

    # For numeric columns, fill NA with median
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    for col in num_cols:
        if df[col].isna().any():
            median = df[col].median()
            df[col] = df[col].fillna(median)
            logger.info(f"Filled NA in numeric column '{col}' with median {median:.4f}")
    # For non-numeric, forward fill then back fill as fallback
    obj_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    for col in obj_cols:
        if df[col].isna().any():
            df[col] = df[col].ffill().bfill()
            logger.info(f"Filled NA in object column '{col}' via ffill/bfill")
    return df


def _standardize_diagnosis_values(df: pd.DataFrame) -> pd.DataFrame:
    # Normalize diagnosis to uppercase M/B if present in variant forms
    if config.DIAGNOSIS_COLUMN in df.columns:
        df[config.DIAGNOSIS_COLUMN] = df[config.DIAGNOSIS_COLUMN].astype(str).str.strip().str.upper()
    return df


def clean_and_validate(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and validate the dataset.

    - Drop optional ID columns
    - Standardize diagnosis values
    - Validate required columns
    - Handle missing values
    """
    df = _drop_optional_id_columns(df)
    df = _standardize_diagnosis_values(df)
    _validate_columns(df)
    df = _clean_missing_values(df)
    logger.info("Data cleaning and validation complete")
    return df


def generate_eda(df: pd.DataFrame, out_dir: str) -> Dict[str, str]:
    """Generate basic EDA plots and save to out_dir.

    Plots:
    - Class distribution bar plot for diagnosis
    - Correlation heatmap among numeric features (top 15 by variance)

    Args:
        df: Cleaned DataFrame
        out_dir: Output directory to save figures
    Returns:
        Dict with file paths of saved figures
    """
    outputs: Dict[str, str] = {}
    try:
        import seaborn as sns  # type: ignore
        import matplotlib.pyplot as plt  # type: ignore
    except Exception as e:
        logger.warning(f"EDA plotting dependencies not available: {e}. Skipping plots.")
        return outputs

    safe_mkdir(out_dir)

    # Class distribution
    cls_fig = os.path.join(out_dir, "class_distribution.png")
    plt.figure(figsize=(6, 4))
    sns.countplot(x=df[config.DIAGNOSIS_COLUMN])
    plt.title("Diagnosis Class Distribution")
    plt.xlabel("Diagnosis (M/B)")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(cls_fig)
    plt.close()
    outputs["class_distribution"] = cls_fig
    logger.info(f"Saved class distribution plot: {cls_fig}")

    # Correlation heatmap of numeric features
    num_df = df.select_dtypes(include=[np.number])
    if not num_df.empty:
        # Select top 15 variables by variance for readability
        variances = num_df.var().sort_values(ascending=False)
        top_cols = variances.head(15).index.tolist()
        corr = num_df[top_cols].corr()
        heat_fig = os.path.join(out_dir, "correlation_heatmap.png")
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr, cmap="coolwarm", annot=False)
        plt.title("Correlation Heatmap (Top 15 by Variance)")
        plt.tight_layout()
        plt.savefig(heat_fig)
        plt.close()
        outputs["correlation_heatmap"] = heat_fig
        logger.info(f"Saved correlation heatmap: {heat_fig}")
    else:
        logger.warning("No numeric features found for correlation heatmap")

    return outputs


def get_processed_dataframe(csv_path: str, out_dir: Optional[str] = None) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """Load, clean, validate, and optionally perform EDA on the dataset.

    Args:
        csv_path: Path to dataset CSV
        out_dir: If provided, EDA figures will be saved there
    Returns:
        Tuple of (processed DataFrame, EDA outputs dict)
    """
    df = load_dataset(csv_path)
    df = clean_and_validate(df)
    outputs: Dict[str, str] = {}
    if out_dir:
        outputs = generate_eda(df, out_dir)
    return df, outputs


def main() -> None:
    parser = argparse.ArgumentParser(description="Data loader and EDA for Wisconsin Breast Cancer dataset")
    parser.add_argument("--data", type=str, default=config.DATA_PATH, help="Path to data.csv")
    parser.add_argument("--out", type=str, default=os.path.join(config.PROJECT_ROOT, "outputs", "eda"), help="Output directory for EDA")
    args = parser.parse_args()

    try:
        df, outputs = get_processed_dataframe(args.data, args.out)
        logger.info(f"Processed DataFrame shape: {df.shape}")
        logger.info(f"EDA outputs: {outputs}")
    except Exception as e:
        logger.exception(f"Data loading/EDA failed: {e}")
        raise


if __name__ == "__main__":
    main()
