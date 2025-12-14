from __future__ import annotations
"""Test suite for modular pipeline.

Verifies:
- Imports work
- Data loading and cleaning
- Preprocessing pipeline (encode, split, standardize)
- PCA reduction
- Model training end-to-end (small run)

Outputs clear PASS/FAIL per section.
"""

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

PASS = "\u2714"
FAIL = "\u2716"


def print_result(section: str, ok: bool, detail: str = "") -> None:
    status = PASS if ok else FAIL
    msg = f"[{status}] {section}"
    if detail:
        msg += f" - {detail}"
    print(msg)


def test_imports() -> bool:
    try:
        import src.data_loader  # noqa: F401
        import src.data_preprocessor  # noqa: F401
        import src.pipeline_model  # noqa: F401
        import src.model_trainer  # noqa: F401
        import src.model_evaluator  # noqa: F401
        import app  # noqa: F401
        return True
    except Exception as e:
        print(f"Import error: {e}")
        return False


def test_data_loading() -> bool:
    try:
        from src.data_loader import get_processed_dataframe
        csv_path = os.path.join(PROJECT_ROOT, "data", "data.csv")
        df, _ = get_processed_dataframe(csv_path, None)
        ok = not df.empty and "diagnosis" in df.columns
        detail = f"shape={df.shape}, columns={len(df.columns)}"
        print(f"Diagnosis counts: {df['diagnosis'].value_counts().to_dict()}")
        return ok
    except Exception as e:
        print(f"Data loading error: {e}")
        return False


def test_preprocessing() -> bool:
    try:
        from src.data_loader import get_processed_dataframe
        from src.data_preprocessor import DataPreprocessor
        df, _ = get_processed_dataframe(os.path.join(PROJECT_ROOT, "data", "data.csv"), None)
        p = DataPreprocessor(df)
        p.remove_id_column(); p.encode_diagnosis()
        X, y = p.split_features_target()
        split = p.train_test_split_data(X, y)
        split_std = p.standardize_features(split, save_path=os.path.join(PROJECT_ROOT, "models", "scaler.pkl"))
        ok = split_std.X_train.shape[0] > 0 and split_std.X_test.shape[0] > 0
        detail = f"train={split_std.X_train.shape}, test={split_std.X_test.shape}"
        print(detail)
        return ok
    except Exception as e:
        print(f"Preprocessing error: {e}")
        return False


def test_pca() -> bool:
    try:
        from src.data_loader import get_processed_dataframe
        from src.data_preprocessor import DataPreprocessor
        from src.pipeline_model import PCAReducer
        df, _ = get_processed_dataframe(os.path.join(PROJECT_ROOT, "data", "data.csv"), None)
        p = DataPreprocessor(df)
        p.remove_id_column(); p.encode_diagnosis()
        X, y = p.split_features_target()
        split = p.train_test_split_data(X, y)
        split_std = p.standardize_features(split, save_path=os.path.join(PROJECT_ROOT, "models", "scaler.pkl"))
        pca = PCAReducer(n_components=15)
        res = pca.run(split_std.X_train, split_std.X_test)
        ok = res.X_train_reduced.shape[1] == 15 and res.cumulative_variance > 0.95
        detail = f"reduced_train={res.X_train_reduced.shape}, cum_var={res.cumulative_variance:.4f}"
        print(detail)
        return ok
    except Exception as e:
        print(f"PCA error: {e}")
        return False


def test_training() -> bool:
    try:
        from src.data_loader import get_processed_dataframe
        from src.data_preprocessor import DataPreprocessor
        from src.pipeline_model import PCAReducer
        from src.model_trainer import ModelTrainer
        from src.model_evaluator import load_metrics
        df, _ = get_processed_dataframe(os.path.join(PROJECT_ROOT, "data", "data.csv"), None)
        p = DataPreprocessor(df)
        p.remove_id_column(); p.encode_diagnosis()
        X, y = p.split_features_target()
        split = p.train_test_split_data(X, y)
        split_std = p.standardize_features(split, save_path=os.path.join(PROJECT_ROOT, "models", "scaler.pkl"))
        pca = PCAReducer(n_components=15)
        res = pca.run(split_std.X_train, split_std.X_test)
        trainer = ModelTrainer(cv_folds=5)
        results = trainer.train_all(res.X_train_reduced, split_std.y_train, res.X_test_reduced, split_std.y_test)
        metrics = load_metrics()
        ok = isinstance(metrics, dict) and len(metrics) >= 4
        print("Models trained:", [r.name for r in results])
        print("Metrics keys:", list(metrics.keys()))
        return ok
    except Exception as e:
        print(f"Training error: {e}")
        return False


if __name__ == "__main__":
    sections = [
        ("Imports", test_imports),
        ("Data Loading", test_data_loading),
        ("Preprocessing", test_preprocessing),
        ("PCA", test_pca),
        ("Model Training", test_training),
    ]
    overall = True
    for name, func in sections:
        ok = func()
        print_result(name, ok)
        overall = overall and ok
    print("\nOverall:", PASS if overall else FAIL)
