#!/bin/bash
set -e

# run_server.sh

if [ ! -d "venv" ]; then
    echo "[ERROR] Virtual environment not found! Run setup_pipeline.sh first."
    exit 1
fi

source venv/bin/activate

echo "=========================================="
echo "         STARTING FLASK SERVER            "
echo "=========================================="
echo "Server will be available at http://localhost:5000"

python3 app.py
