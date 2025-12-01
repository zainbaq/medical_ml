#!/bin/bash

echo "Starting Alzheimer's Prediction API..."

cd "$(dirname "$0")"

# Create virtual environment if it doesn't exist
if [ ! -d "backend/venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv backend/venv
fi

# Activate virtual environment
source backend/venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install -r backend/requirements.txt

# Start API server
echo "Starting FastAPI server on http://localhost:8001"
uvicorn backend.main:app --host 0.0.0.0 --port 8001 --reload
