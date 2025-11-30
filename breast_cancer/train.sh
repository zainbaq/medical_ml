#!/bin/bash
# Script to train the breast cancer prediction model

echo "=========================================="
echo "Breast Cancer Model Training"
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
pip install -q -r requirements.txt

# Run training script
echo ""
echo "Starting model training..."
echo ""
python train_model.py

echo ""
echo "Training complete!"
echo "You can now start the API with: ./start_api.sh"
