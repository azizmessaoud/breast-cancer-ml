from __future__ import annotations
from typing import List

# Reproducibility
RANDOM_STATE: int = 42

# Data & pipeline settings
TEST_SIZE: float = 0.30  # 70/30 split
CV_FOLDS: int = 5
PCA_COMPONENTS: int = 15
TARGET_VARIANCE: float = 0.95

# Feature/target columns (to be validated by data_loader)
ID_COLUMN: str = "id"
DIAGNOSIS_COLUMN: str = "diagnosis"

# Model registry
MODEL_NAMES: List[str] = [
    "random_forest",
    "svm",
    "neural_network",
    "gradient_boosting",
]

# Paths
PROJECT_ROOT: str = "breast-cancer-detection"
DATA_PATH: str = f"{PROJECT_ROOT}/data/data.csv"
MODELS_DIR: str = f"{PROJECT_ROOT}/models"
METRICS_FILE: str = f"{PROJECT_ROOT}/models/metrics.json"
SCALER_FILE: str = f"{PROJECT_ROOT}/models/scaler.pkl"
PCA_FILE: str = f"{PROJECT_ROOT}/models/pca.pkl"

# API
API_PORT: int = 5000
