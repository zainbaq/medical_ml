#!/bin/bash
# Start the Medical ML Service Registry

echo "Starting Medical ML Service Registry on port 9000..."
cd "$(dirname "$0")/backend"
uvicorn main:app --host 0.0.0.0 --port 9000 --reload
