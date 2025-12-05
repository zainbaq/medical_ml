#!/bin/bash
# Script to start the FastAPI server

echo "=========================================="
echo "Starting Cardiovascular Disease API"
echo "=========================================="
echo ""

# Navigate to cardiovascular_disease directory
cd "$(dirname "$0")"

# Start server from cardiovascular_disease directory
echo "Starting FastAPI server..."
echo "API will be available at: http://localhost:8003"
echo "Documentation at: http://localhost:8003/docs"
echo ""
echo "Press CTRL+C to stop the server"
echo ""
uvicorn backend.main:app --host 0.0.0.0 --port 8003 --reload
