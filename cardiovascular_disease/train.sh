#!/bin/bash
# Script to train the cardiovascular disease prediction model

echo "=========================================="
echo "Cardiovascular Disease Model Training"
echo "=========================================="
echo ""

# Navigate to training directory
cd "$(dirname "$0")/training"

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Run training
echo ""
echo "Starting model training..."
python train_model.py

echo ""
echo "=========================================="
echo "Training complete!"
echo "=========================================="
