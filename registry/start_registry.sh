#!/bin/bash
# Start the Medical ML Service Registry

echo "=========================================="
echo "Starting Medical ML Service Registry"
echo "=========================================="
echo ""

cd "$(dirname "$0")"

echo "Starting service registry..."
echo "API will be available at: http://localhost:9000"
echo "Documentation at: http://localhost:9000/docs"
echo ""
echo "Press CTRL+C to stop the server"
echo ""
uvicorn backend.main:app --host 0.0.0.0 --port 9000 --reload
