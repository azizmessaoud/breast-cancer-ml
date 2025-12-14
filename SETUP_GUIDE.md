# Setup Guide

This guide explains how to install, run, and deploy the modular pipeline locally and on common platforms.

## Prerequisites
- Python 3.8 or later
- pip
- Optional: Docker, Kaggle account, VibeFlow environment

## Installation
1. Clone or download the repository.
2. Create a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Windows: .venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Running the Pipeline
Run each step sequentially:
```bash
python main.py --step 1  # Load & EDA
python main.py --step 2  # Preprocess (saves scaler, features.json)
python main.py --step 3  # PCA (saves pca)
python main.py --step 4  # Train & evaluate (saves models & metrics)
```
Artifacts will be stored under `models/` and additional info under `outputs/`.

## Starting the Web App
```bash
python app.py
```
Open `http://localhost:5000` and follow the instructions. API endpoints:
- GET `/metrics`
- GET `/features`
- POST `/predict`

## Docker Deployment (Optional)
```bash
docker build -t breast-cancer-detection .
docker run -p 5000:5000 breast-cancer-detection
```

## Kaggle Notes
- Use data from Kaggle datasets via `../input/<dataset-name>/data.csv`
- Create files using `%%writefile` cells to replicate this structure
- Install only packages supported by Kaggle environment (avoid system-level builds)
- Run the pipeline steps in order; display results and metrics

## VibeFlow Compatibility
- Flask app runs on port 5000
- Static assets served from `/static`
- Endpoints `/metrics`, `/features`, `/predict` are available

## Troubleshooting
- If plotting libraries are missing, EDA and ROC plots will be skipped with a log warning; install `matplotlib` and `seaborn` to enable figures
- Ensure `models/scaler.pkl`, `models/pca.pkl`, and at least one trained model exist before calling `/predict`
- For API `POST /predict`, provide either a complete `values` list in the correct feature order or a `features` dict keyed by feature names

