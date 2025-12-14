from __future__ import annotations
"""
Main orchestrator for the modular breast cancer detection pipeline.

Usage:
    python main.py --step 1   # Load & clean data, save basic info
    python main.py --step 2   # Preprocess: encode, split, standardize, save scaler
    python main.py --step 3   # PCA dimensionality reduction, save PCA
    python main.py --step 4   # Train models, evaluate, save models & metrics

Logs INFO-level messages, handles errors gracefully.
"""

import argparse
import os
import sys
from typing import Dict

# Allow running from repo root
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from src.utils import setup_logger, save_json
from src import config
from src.data_loader import get_processed_dataframe
from src.data_preprocessor import DataPreprocessor
from src.pipeline_model import PCAReducer
from src.model_trainer import ModelTrainer
from src.model_evaluator import load_metrics, best_model_by_metric

logger = setup_logger("orchestrator")


def step1_load() -> Dict[str, str]:
    logger.info("Step 1: Loading and validating dataset")
    df, outputs = get_processed_dataframe(
        os.path.join(PROJECT_ROOT, "data", "data.csv"),
        os.path.join(PROJECT_ROOT, "outputs", "eda"),
    )
    # Save basic dataset info for later reference
    info = {
        "shape": list(df.shape),
        "columns": df.columns.tolist(),
        "diagnosis_counts": df[config.DIAGNOSIS_COLUMN].value_counts().to_dict(),
    }
    save_json(info, os.path.join(PROJECT_ROOT, "outputs", "dataset_info.json"))
    logger.info(f"Saved dataset info and EDA outputs: {outputs}")
    return info


def step2_preprocess() -> Dict[str, str]:
    logger.info("Step 2: Preprocessing and standardization")
    df, _ = get_processed_dataframe(os.path.join(PROJECT_ROOT, "data", "data.csv"), None)
    prep = DataPreprocessor(df)
    prep.remove_id_column()
    prep.encode_diagnosis()
    X, y = prep.split_features_target()
    split = prep.train_test_split_data(X, y)
    split_std = prep.standardize_features(split, save_path=os.path.join(PROJECT_ROOT, "models", "scaler.pkl"))
    # Save feature names for API /features endpoint
    features_json = {
        "features": X.columns.tolist(),
        "target": config.DIAGNOSIS_COLUMN,
    }
    save_json(features_json, os.path.join(PROJECT_ROOT, "outputs", "features.json"))
    logger.info("Saved features.json and scaler.pkl")
    return {"X_train": str(split_std.X_train.shape), "X_test": str(split_std.X_test.shape)}


def step3_pca() -> Dict[str, str]:
    logger.info("Step 3: PCA dimensionality reduction")
    df, _ = get_processed_dataframe(os.path.join(PROJECT_ROOT, "data", "data.csv"), None)
    prep = DataPreprocessor(df)
    prep.remove_id_column(); prep.encode_diagnosis()
    X, y = prep.split_features_target()
    split = prep.train_test_split_data(X, y)
    split_std = prep.standardize_features(split, save_path=os.path.join(PROJECT_ROOT, "models", "scaler.pkl"))
    pca = PCAReducer(n_components=config.PCA_COMPONENTS)
    res = pca.run(split_std.X_train, split_std.X_test)
    # Save PCA info
    pca_info = {
        "n_components": res.n_components,
        "cumulative_variance": float(res.cumulative_variance),
        "explained_variance_ratio": [float(v) for v in res.explained_variance_ratio],
    }
    save_json(pca_info, os.path.join(PROJECT_ROOT, "outputs", "pca_info.json"))
    logger.info("Saved PCA info to outputs/pca_info.json and PCA model to models/pca.pkl")
    return {"train_reduced": str(res.X_train_reduced.shape), "test_reduced": str(res.X_test_reduced.shape)}


def step4_train() -> Dict[str, str]:
    logger.info("Step 4: Training models and evaluating performance")
    df, _ = get_processed_dataframe(os.path.join(PROJECT_ROOT, "data", "data.csv"), None)
    prep = DataPreprocessor(df)
    prep.remove_id_column(); prep.encode_diagnosis()
    X, y = prep.split_features_target()
    split = prep.train_test_split_data(X, y)
    split_std = prep.standardize_features(split, save_path=os.path.join(PROJECT_ROOT, "models", "scaler.pkl"))
    pca = PCAReducer(n_components=config.PCA_COMPONENTS)
    res = pca.run(split_std.X_train, split_std.X_test)

    trainer = ModelTrainer(cv_folds=config.CV_FOLDS)
    results = trainer.train_all(res.X_train_reduced, split_std.y_train, res.X_test_reduced, split_std.y_test)

    # Choose best model by accuracy and copy path
    best_name = best_model_by_metric("accuracy") or "svm"
    metrics = load_metrics()
    best_path = metrics.get(best_name, {}).get("model_path")
    if best_path and os.path.exists(os.path.join(PROJECT_ROOT, best_path.replace(f"{config.PROJECT_ROOT}/", ""))):
        # Create/overwrite a symlink-like file reference
        save_json({"best_model": best_name, "path": best_path}, os.path.join(PROJECT_ROOT, "models", "best_model.json"))
        logger.info(f"Best model: {best_name} at {best_path}")
    else:
        logger.warning("Best model path not found; skipping best_model.json")
    return {name: str(r.metrics) for name, r in [(res.name, res) for res in results]}


def main() -> None:
    parser = argparse.ArgumentParser(description="Breast cancer detection pipeline orchestrator")
    parser.add_argument("--step", type=int, choices=[1, 2, 3, 4], required=True, help="Pipeline step to run")
    args = parser.parse_args()
    try:
        if args.step == 1:
            info = step1_load()
            logger.info(f"Step 1 complete: {info}")
        elif args.step == 2:
            out = step2_preprocess()
            logger.info(f"Step 2 complete: {out}")
        elif args.step == 3:
            out = step3_pca()
            logger.info(f"Step 3 complete: {out}")
        elif args.step == 4:
            out = step4_train()
            logger.info(f"Step 4 complete: {out}")
    except Exception as e:
        logger.exception(f"Pipeline failed at step {args.step}: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
