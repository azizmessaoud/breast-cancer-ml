# Breast Cancer Detection (Wisconsin Diagnostic Dataset)

Production-ready, modular machine learning pipeline to detect breast cancer (Malignant vs Benign) using the Wisconsin Diagnostic dataset (569 samples, 30 features). The project includes:

- Data loading, cleaning, validation and optional EDA
- Preprocessing (diagnosis encoding, stratified split, standardization)
- PCA dimensionality reduction (30 → 15 components, >95% variance)
- Model training: Random Forest, SVM, Neural Network (MLP), Gradient Boosting with GridSearchCV + 5-fold CV
- Evaluation: accuracy, precision, recall, F1, confusion matrix, ROC curve/AUC
- Flask web API and mobile-friendly web UI for real-time predictions
- Comprehensive tests and documentation
- Ready for GitHub, Kaggle, and VibeFlow

## Repository Structure

```
breast-cancer-detection/
├── data/
│   └── data.csv
├── models/
├── outputs/
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── utils.py
│   ├── data_loader.py
│   ├── data_preprocessor.py
│   ├── pipeline_model.py
│   ├── model_trainer.py
│   └── model_evaluator.py
├── app.py
├── main.py
├── test_modules.py
├── templates/
│   ├── index.html
│   └── results.html
├── static/
│   └── style.css
├── requirements.txt
├── Dockerfile (optional)
├── README.md
└── SETUP_GUIDE.md
```

## Quick Start

1. Create a Python 3.8+ virtual environment
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the pipeline end-to-end:
   ```bash
   # Step 1: Load & EDA
   python main.py --step 1
   # Step 2: Preprocess
   python main.py --step 2
   # Step 3: PCA
   python main.py --step 3
   # Step 4: Train & evaluate
   python main.py --step 4
   ```
4. Start the web app:
   ```bash
   python app.py  # serves http://localhost:5000
   ```

## Expected Results (target metrics)

- Random Forest: 95–97% accuracy
- SVM: 96–98% accuracy
- Neural Network (MLP): 94–96% accuracy
- Gradient Boosting: 96–98% accuracy

Smoke-test results achieved in this build (may vary slightly):

- RF: accuracy 0.9532, precision 0.9667, recall 0.9063, F1 0.9355
- SVM: accuracy 0.9708, precision 1.0000, recall 0.9219, F1 0.9593
- MLP: accuracy 0.9766, precision 1.0000, recall 0.9375, F1 0.9677
- GB: accuracy 0.9649, precision 1.0000, recall 0.9063, F1 0.9508

Artifacts are saved under `models/`:

- Trained models: `random_forest.joblib`, `svm.joblib`, `neural_network.joblib`, `gradient_boosting.joblib`
- ROC curves: `*_roc_curve.png`
- Metrics JSON: `models/metrics.json`
- Best model info: `models/best_model.json`
- Preprocessing artifacts: `models/scaler.pkl`, `models/pca.pkl`

## API Documentation

Server runs on port 5000 (configurable in `src/config.py`).

- GET `/metrics`
  - Returns JSON of metrics and model paths
  - Example:
    ```bash
    curl http://localhost:5000/metrics
    ```

- GET `/features`
  - Returns feature names and target label
  - Example:
    ```bash
    curl http://localhost:5000/features
    ```

- POST `/predict`
  - Accepts either `{"values": [<30 floats>]}` in the order of `/features`, or `{"features": {name: value}}`
  - Returns predicted class (`1` malignant, `0` benign) and confidence
  - Examples:
    ```bash
    # Using values list
    curl -X POST http://localhost:5000/predict \
      -H 'Content-Type: application/json' \
      -d '{"values": [17.99,10.38,122.8,1001.0,0.1184,0.2776,0.3001,0.1471,0.2419,0.07871,1.095,0.9053,8.589,153.4,0.006399,0.04904,0.05373,0.01587,0.03003,0.006193,25.38,17.33,184.6,2019.0,0.1622,0.6656,0.7119,0.2654,0.4601,0.1189]}'

    # Using features dict
    curl -X POST http://localhost:5000/predict \
      -H 'Content-Type: application/json' \
      -d '{"features": {"radius_mean": 17.99, "texture_mean": 10.38, ...}}'
    ```

## Web UI

- Open `http://localhost:5000`
- Paste comma-separated values following the feature order
- Submit to view diagnosis badge and confidence
- Mobile-friendly layout

## Testing

Run the test suite:
```bash
python test_modules.py
```
All sections should report PASS, and the overall summary should be ✓.

## Configuration

Key settings in `src/config.py`:
- `TEST_SIZE = 0.30` (70/30 split)
- `CV_FOLDS = 5`
- `PCA_COMPONENTS = 15`
- `API_PORT = 5000`
- Paths for models and artifacts

## Deployment

### Docker (optional)

A simple Dockerfile is provided. Build and run:
```bash
docker build -t breast-cancer-detection .
docker run -p 5000:5000 breast-cancer-detection
```

### Kaggle

Use Kaggle-specific dataset paths (`../input/...`). Prepare files with `%%writefile` in a Kaggle notebook, then run steps 1–4. Avoid non-Kaggle dependencies.

### VibeFlow

Runs a Flask app on port 5000, serves static files and templates; endpoints `/metrics`, `/features`, `/predict` satisfy VibeFlow compatibility requirements.

## Notes

- Models achieve target accuracy >95% with PCA + standardized features
- All modules have type hints, docstrings, logging, and error handling
- Artifacts are serialized for reuse in API inference

## Platform Notes

### Important: Generate Feature Names Before Starting the Web App
Run step 2 to create outputs/features.json so the UI and API know the feature order:
```bash
python main.py --step 2
```
Then start the app:
```bash
python app.py
```

### Kaggle Path Configuration
If running on Kaggle, set the data path to your dataset location, e.g.:
```python
# In src/config.py on Kaggle
DATA_PATH = "../input/<dataset-name>/data.csv"
```
And create files via `%%writefile` cells as per the provided template.

### VibeFlow
The Flask app serves on port 5000 by default. Ensure your environment routes port 5000 publicly. Endpoints `/metrics`, `/features`, `/predict` are exposed and static assets are served from `/static`.

