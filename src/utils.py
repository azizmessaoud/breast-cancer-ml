from __future__ import annotations
import json
import logging
import os
from dataclasses import dataclass
from typing import Any, Dict

DEFAULT_LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"


def setup_logger(name: str = "breast_cancer", level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(DEFAULT_LOG_FORMAT)
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(level)
        logger.propagate = False
    return logger


def safe_mkdir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def save_json(data: Dict[str, Any], filepath: str) -> None:
    safe_mkdir(os.path.dirname(filepath))
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def load_json(filepath: str) -> Dict[str, Any]:
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


@dataclass
class ArtifactPaths:
    models_dir: str
    metrics_file: str
    scaler_file: str
    pca_file: str
