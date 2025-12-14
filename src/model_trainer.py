from __future__ import annotations
"""
Model training module: trains RandomForest, SVM, Neural Network, GradientBoosting with GridSearchCV and 5-fold CV.
Persists best models and returns evaluation metrics.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .utils import setup_logger, save_json
from . import config

logger = setup_logger(__name__)

try:
    from sklearn.model_selection import GridSearchCV
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.svm import SVC
    from sklearn.neural_network import MLPClassifier
    from sklearn.metrics import (
        accuracy_score,
        precision_score,
        recall_score,
        f1_score,
        confusion_matrix,
        roc_auc_score,
        RocCurveDisplay,
    )
except Exception as e:  # pragma: no cover
    logger.error("scikit-learn is required. Error: %s", e)
    raise

try:
    import joblib  # type: ignore
    _JOBLIB_AVAILABLE = True
except Exception:
    import pickle
    _JOBLIB_AVAILABLE = False


def _save_model(model: Any, filepath: str) -> None:
    import os
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    if _JOBLIB_AVAILABLE:
        joblib.dump(model, filepath)
    else:
        with open(filepath, "wb") as f:
            import pickle
            pickle.dump(model, f)


def _maybe_save_plot(display: Any, filepath: str) -> None:
    try:
        import matplotlib.pyplot as plt  # type: ignore
        display.plot()
        import os
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        plt.savefig(filepath)
        plt.close()
        logger.info(f"Saved plot: {filepath}")
    except Exception as e:
        logger.warning(f"Plotting skipped (dependency unavailable or error): {e}")


@dataclass
class ModelResult:
    name: str
    params: Dict[str, Any]
    metrics: Dict[str, float]
    confusion_matrix: np.ndarray
    roc_auc: float
    model_path: str
    roc_curve_path: Optional[str]


class ModelTrainer:
    def __init__(self, cv_folds: int = config.CV_FOLDS):
        self.cv_folds = cv_folds

    def _evaluate(self, model: Any, X_test: np.ndarray, y_test: np.ndarray, name: str) -> Tuple[Dict[str, float], np.ndarray, float, Optional[str]]:
        y_pred = model.predict(X_test)
        acc = float(accuracy_score(y_test, y_pred))
        prec = float(precision_score(y_test, y_pred))
        rec = float(recall_score(y_test, y_pred))
        f1 = float(f1_score(y_test, y_pred))
        cm = confusion_matrix(y_test, y_pred)
        roc_path = None
        try:
            if hasattr(model, "predict_proba"):
                y_scores = model.predict_proba(X_test)[:, 1]
            elif hasattr(model, "decision_function"):
                y_scores = model.decision_function(X_test)
            else:
                y_scores = None
            roc_auc = float(roc_auc_score(y_test, y_scores)) if y_scores is not None else float("nan")
            if y_scores is not None:
                disp = RocCurveDisplay.from_predictions(y_test, y_scores)
                roc_path = f"{config.PROJECT_ROOT}/models/{name}_roc_curve.png"
                _maybe_save_plot(disp, roc_path)
        except Exception as e:
            logger.warning(f"Failed to compute ROC for {name}: {e}")
            roc_auc = float("nan")
        metrics = {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}
        return metrics, cm, roc_auc, roc_path

    def train_random_forest(self, X_train: np.ndarray, y_train: np.ndarray) -> Any:
        rf = RandomForestClassifier(random_state=config.RANDOM_STATE)
        grid = {
            "n_estimators": [100, 200],
            "max_depth": [None, 10],
            "min_samples_split": [2, 4],
        }
        gs = GridSearchCV(rf, grid, cv=self.cv_folds, n_jobs=-1)
        gs.fit(X_train, y_train)
        logger.info(f"RandomForest best params: {gs.best_params_}")
        return gs.best_estimator_

    def train_svm(self, X_train: np.ndarray, y_train: np.ndarray) -> Any:
        svm = SVC(probability=True, random_state=config.RANDOM_STATE)
        grid = {
            "C": [0.5, 1.0, 2.0],
            "kernel": ["rbf"],
            "gamma": ["scale", "auto"],
        }
        gs = GridSearchCV(svm, grid, cv=self.cv_folds, n_jobs=-1)
        gs.fit(X_train, y_train)
        logger.info(f"SVM best params: {gs.best_params_}")
        return gs.best_estimator_

    def train_neural_network(self, X_train: np.ndarray, y_train: np.ndarray) -> Any:
        # Use scikit-learn MLPClassifier as primary to ensure compatibility
        mlp = MLPClassifier(random_state=config.RANDOM_STATE, max_iter=500)
        grid = {
            "hidden_layer_sizes": [(32,), (64,), (32, 16)],
            "activation": ["relu"],
            "alpha": [0.0001, 0.001],
            "learning_rate": ["adaptive"],
        }
        gs = GridSearchCV(mlp, grid, cv=self.cv_folds, n_jobs=-1)
        gs.fit(X_train, y_train)
        logger.info(f"MLP best params: {gs.best_params_}")
        return gs.best_estimator_

    def train_gradient_boosting(self, X_train: np.ndarray, y_train: np.ndarray) -> Any:
        gb = GradientBoostingClassifier(random_state=config.RANDOM_STATE)
        grid = {
            "n_estimators": [100, 200],
            "learning_rate": [0.05, 0.1],
            "max_depth": [3, 4],
        }
        gs = GridSearchCV(gb, grid, cv=self.cv_folds, n_jobs=-1)
        gs.fit(X_train, y_train)
        logger.info(f"GradientBoosting best params: {gs.best_params_}")
        return gs.best_estimator_

    def train_all(self, X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray) -> List[ModelResult]:
        results: List[ModelResult] = []
        registry = [
            ("random_forest", self.train_random_forest),
            ("svm", self.train_svm),
            ("neural_network", self.train_neural_network),
            ("gradient_boosting", self.train_gradient_boosting),
        ]
        for name, trainer in registry:
            logger.info(f"Training model: {name}")
            model = trainer(X_train, y_train)
            # Evaluate
            metrics, cm, roc_auc, roc_path = self._evaluate(model, X_test, y_test, name)
            # Save model
            model_path = f"{config.PROJECT_ROOT}/models/{name}.joblib"
            _save_model(model, model_path)
            logger.info(f"Saved {name} model to {model_path}")
            results.append(
                ModelResult(
                    name=name,
                    params=getattr(model, "get_params", lambda: {})(),
                    metrics=metrics,
                    confusion_matrix=cm,
                    roc_auc=roc_auc,
                    model_path=model_path,
                    roc_curve_path=roc_path,
                )
            )
        # Save aggregated metrics to JSON
        metrics_json = {
            r.name: {
                "params": r.params,
                "metrics": r.metrics,
                "roc_auc": r.roc_auc,
                "model_path": r.model_path,
                "roc_curve_path": r.roc_curve_path,
            }
            for r in results
        }
        save_json(metrics_json, config.METRICS_FILE)
        logger.info(f"Saved metrics to {config.METRICS_FILE}")
        return results
