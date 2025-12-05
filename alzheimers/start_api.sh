#!/bin/bash

echo "=========================================="
echo "Starting Alzheimer's Prediction API"
echo "=========================================="
echo ""

cd "$(dirname "$0")"

# Start API server
echo "Starting FastAPI server..."
echo "API will be available at: http://localhost:8001"
echo "Documentation at: http://localhost:8001/docs"
echo ""
echo "Press CTRL+C to stop the server"
echo ""
uvicorn backend.main:app --host 0.0.0.0 --port 8001 --reload
