from __future__ import annotations
"""
Flask web API and frontend for breast cancer detection.

Endpoints:
- GET /metrics  -> JSON of model performance stats
- GET /features -> JSON of feature names and target
- POST /predict -> Single prediction given feature values
- GET /         -> Web form for manual input

Run:
    python app.py  # serves on port 5000
"""

import json
import os
from typing import Any, Dict, List, Optional, Tuple

from flask import Flask, jsonify, request, render_template, redirect, url_for

# Flexible imports when running as script
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
SRC_ROOT = os.path.join(PROJECT_ROOT, "src")
import sys
sys.path.insert(0, PROJECT_ROOT)

from src.utils import setup_logger, load_json
from src import config

logger = setup_logger("api")
app = Flask(__name__, static_url_path="/static", static_folder=os.path.join(PROJECT_ROOT, "static"), template_folder=os.path.join(PROJECT_ROOT, "templates"))

try:
    import joblib  # type: ignore
    _JOBLIB_AVAILABLE = True
except Exception:
    _JOBLIB_AVAILABLE = False


def _load_model(path: str) -> Any:
    if _JOBLIB_AVAILABLE:
        return joblib.load(path)
    else:
        import pickle
        with open(path, "rb") as f:
            return pickle.load(f)


def _load_artifacts() -> Tuple[Any, Any, Any, List[str], str]:
    """Load scaler, PCA, best model, feature names, and model name."""
    scaler_path = os.path.join(PROJECT_ROOT, "models", "scaler.pkl")
    pca_path = os.path.join(PROJECT_ROOT, "models", "pca.pkl")
    best_model_info = load_json(os.path.join(PROJECT_ROOT, "models", "best_model.json")) if os.path.exists(os.path.join(PROJECT_ROOT, "models", "best_model.json")) else {}
    model_name = best_model_info.get("best_model", "svm")
    model_rel_path = best_model_info.get("path", f"{config.PROJECT_ROOT}/models/{model_name}.joblib")
    model_path = os.path.join(PROJECT_ROOT, os.path.relpath(model_rel_path, config.PROJECT_ROOT))

    features = []
    features_json_path = os.path.join(PROJECT_ROOT, "outputs", "features.json")
    if os.path.exists(features_json_path):
        fj = load_json(features_json_path)
        features = fj.get("features", [])
    else:
        logger.warning("features.json not found; using empty feature list")

    scaler = _load_model(scaler_path)
    pca = _load_model(pca_path)
    model = _load_model(model_path)
    logger.info(f"Loaded artifacts: scaler={scaler_path}, pca={pca_path}, model={model_path} ({model_name})")
    return scaler, pca, model, features, model_name


SCALER, PCA_MODEL, BEST_MODEL, FEATURE_NAMES, BEST_MODEL_NAME = _load_artifacts()


def _compute_confidence(model: Any, X: Any) -> Tuple[float, Optional[float]]:
    """Return predicted label and confidence score (probability or calibrated decision)."""
    import numpy as np
    y_pred = model.predict(X)
    label = int(y_pred[0])
    conf = None
    try:
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X)[0, 1]
            conf = float(proba)
        elif hasattr(model, "decision_function"):
            # Map decision function to [0,1] via logistic-like transform for display
            df = float(model.decision_function(X)[0])
            conf = float(1 / (1 + np.exp(-df)))
    except Exception as e:
        logger.warning(f"Confidence computation failed: {e}")
    return label, conf


@app.get("/metrics")
def get_metrics():
    path = os.path.join(PROJECT_ROOT, "models", "metrics.json")
    if not os.path.exists(path):
        return jsonify({"error": "metrics.json not found; run training."}), 404
    try:
        data = load_json(path)
        return jsonify(data)
    except Exception as e:
        logger.exception("Failed to load metrics.json")
        return jsonify({"error": str(e)}), 500


@app.get("/features")
def get_features():
    path = os.path.join(PROJECT_ROOT, "outputs", "features.json")
    if not os.path.exists(path):
        return jsonify({"features": FEATURE_NAMES, "target": config.DIAGNOSIS_COLUMN})
    try:
        data = load_json(path)
        return jsonify(data)
    except Exception as e:
        logger.exception("Failed to load features.json")
        return jsonify({"error": str(e)}), 500


@app.post("/predict")
def post_predict():
    """Accepts JSON body: {"values": [...]} or {"features": {feature_name: value, ...}}.
    Returns predicted class (1 malignant, 0 benign) and confidence.
    """
    try:
        payload = request.get_json(silent=True) or {}
        values: Optional[List[float]] = None
        if "values" in payload and isinstance(payload["values"], list):
            values = [float(v) for v in payload["values"]]
            if FEATURE_NAMES and len(values) != len(FEATURE_NAMES):
                return jsonify({"error": f"Expected {len(FEATURE_NAMES)} values, got {len(values)}"}), 400
        elif "features" in payload and isinstance(payload["features"], dict):
            # Order according to FEATURE_NAMES
            if not FEATURE_NAMES:
                return jsonify({"error": "Feature names unavailable"}), 400
            try:
                values = [float(payload["features"][name]) for name in FEATURE_NAMES]
            except Exception as e:
                return jsonify({"error": f"Missing or invalid feature value: {e}"}), 400
        else:
            # Attempt to read from form input (comma-separated)
            form_values = request.form.get("values")
            if form_values:
                parts = [p.strip() for p in form_values.split(",")]
                try:
                    values = [float(p) for p in parts]
                except Exception:
                    return jsonify({"error": "Invalid numeric values in form"}), 400
                if FEATURE_NAMES and len(values) != len(FEATURE_NAMES):
                    return jsonify({"error": f"Expected {len(FEATURE_NAMES)} values, got {len(values)}"}), 400
            else:
                return jsonify({"error": "Provide 'values' list or 'features' dict"}), 400

        import numpy as np
        X = np.array(values, dtype=float).reshape(1, -1)
        Xs = SCALER.transform(X)
        Xp = PCA_MODEL.transform(Xs)
        label, conf = _compute_confidence(BEST_MODEL, Xp)
        result = {
            "model": BEST_MODEL_NAME,
            "label": int(label),
            "confidence": conf,
            "diagnosis": "Malignant" if label == 1 else "Benign",
        }
        # If form submission, render a results page
        if request.form:
            return render_template("results.html", result=result, feature_names=FEATURE_NAMES, values=values)
        return jsonify(result)
    except Exception as e:
        logger.exception("Prediction failed")
        return jsonify({"error": str(e)}), 500


@app.get("/")
def index():
    # Render a simple, mobile-friendly form where users paste comma-separated values
    return render_template("index.html", feature_names=FEATURE_NAMES)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=config.API_PORT, debug=False)
