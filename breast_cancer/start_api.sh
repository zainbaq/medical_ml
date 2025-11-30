#!/bin/bash
# Script to start the FastAPI server

echo "=========================================="
echo "Starting Breast Cancer Prediction API"
echo "=========================================="
echo ""

# Navigate to breast_cancer directory
cd "$(dirname "$0")"

# Check if virtual environment exists in backend
if [ ! -d "backend/venv" ]; then
    echo "Creating virtual environment..."
    cd backend
    python3 -m venv venv
    cd ..
fi

# Activate virtual environment
echo "Activating virtual environment..."
source backend/venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install -q -r backend/requirements.txt

# Start server from breast_cancer directory
echo ""
echo "Starting FastAPI server..."
echo "API will be available at: http://localhost:8000"
echo "Documentation at: http://localhost:8000/docs"
echo ""
echo "Press CTRL+C to stop the server"
echo ""
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
