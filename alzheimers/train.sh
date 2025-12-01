#!/bin/bash

echo "Setting up training environment for Alzheimer's model..."

cd "$(dirname "$0")"

# Create virtual environment if it doesn't exist
if [ ! -d "training/venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv training/venv
fi

# Activate virtual environment
source training/venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install -r training/requirements.txt

# Run training
echo "Starting model training..."
python training/train_model.py

echo "Training complete!"
