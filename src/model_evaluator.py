from __future__ import annotations
"""
Model evaluator: utilities to aggregate metrics and produce comparison plots.
"""

from typing import Dict, List, Optional

import numpy as np

from .utils import setup_logger, load_json
from . import config

logger = setup_logger(__name__)


def load_metrics() -> Dict[str, dict]:
    """Load metrics JSON saved by ModelTrainer."""
    try:
        return load_json(config.METRICS_FILE)
    except Exception as e:
        logger.warning(f"Failed to load metrics: {e}")
        return {}


def best_model_by_metric(metric: str = "accuracy") -> Optional[str]:
    """Return the model name with the best given metric."""
    metrics = load_metrics()
    best_name = None
    best_value = -np.inf
    for name, m in metrics.items():
        val = m.get("metrics", {}).get(metric)
        if val is not None and val > best_value:
            best_value = val
            best_name = name
    logger.info(f"Best model by {metric}: {best_name} ({best_value})")
    return best_name

