#!/bin/bash
set -e

# setup_pipeline.sh

echo "=========================================="
echo "          SETTING UP ENVIRONMENT          "
echo "=========================================="

if [ ! -d "venv" ]; then
    echo "[INFO] Creating virtual environment 'venv'..."
    python3 -m venv venv
else
    echo "[INFO] Using existing 'venv'..."
fi

source venv/bin/activate

echo "[INFO] Upgrading pip..."
pip install --upgrade pip

echo "[INFO] Installing requirements from requirements.txt..."
pip install -r requirements.txt

echo "=========================================="
echo "           RUNNING ML PIPELINE            "
echo "=========================================="

echo "[INFO] Step 1: Data Loading & EDA"
python3 main.py --step 1

echo "[INFO] Step 2: Preprocessing"
python3 main.py --step 2

echo "[INFO] Step 3: PCA Dimensionality Reduction"
python3 main.py --step 3

echo "[INFO] Step 4: Model Training & Evaluation"
python3 main.py --step 4

echo "=========================================="
echo "       PIPELINE COMPLETED SUCCESSFULLY    "
echo "=========================================="
