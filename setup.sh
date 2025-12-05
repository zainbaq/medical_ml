#!/bin/bash
# One-time setup script for Medical ML project
# Run this after creating and activating your virtual environment

echo "=========================================="
echo "Medical ML Project Setup"
echo "=========================================="
echo ""

# Check if we're in a virtual environment
if [ -z "$VIRTUAL_ENV" ]; then
    echo "ERROR: No virtual environment detected!"
    echo "Please create and activate a virtual environment first:"
    echo ""
    echo "  python3 -m venv venv"
    echo "  source venv/bin/activate  # On Linux/Mac"
    echo "  venv\\Scripts\\activate   # On Windows"
    echo ""
    exit 1
fi

echo "Virtual environment detected: $VIRTUAL_ENV"
echo ""

# Install medical-ml-sdk
echo "Installing medical-ml-sdk..."
pip install -e ./shared

if [ $? -ne 0 ]; then
    echo "ERROR: Failed to install medical-ml-sdk"
    exit 1
fi

# Install dependencies for each service
echo ""
echo "Installing dependencies for each service..."
echo ""

echo "1. Installing Breast Cancer API dependencies..."
pip install -r breast_cancer/backend/requirements.txt

echo ""
echo "2. Installing Alzheimer's API dependencies..."
pip install -r alzheimers/backend/requirements.txt

echo ""
echo "3. Installing Cardiovascular Disease API dependencies..."
pip install -r cardiovascular_disease/backend/requirements.txt

echo ""
echo "4. Installing Registry dependencies..."
pip install -r registry/backend/requirements.txt

echo ""
echo "=========================================="
echo "Setup completed successfully!"
echo "=========================================="
echo ""
echo "You can now start the services:"
echo "  - All services:              ./start_all_services.sh"
echo "  - Breast Cancer API:         ./breast_cancer/start_api.sh"
echo "  - Alzheimer's API:           ./alzheimers/start_api.sh"
echo "  - Cardiovascular API:        ./cardiovascular_disease/start_api.sh"
echo "  - Service Registry:          ./registry/start_registry.sh"
echo ""
