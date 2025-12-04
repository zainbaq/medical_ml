#!/bin/bash
#
# Setup unified virtual environment for Medical ML project
#
# This script creates a single venv for all services
#

set -e

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$PROJECT_ROOT/venv"

echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BLUE}     Medical ML - Environment Setup                         ${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

# Check if venv exists
if [ -d "$VENV_DIR" ]; then
    echo -e "\n${YELLOW}Virtual environment already exists at: $VENV_DIR${NC}"
    read -p "Do you want to recreate it? (y/N) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo -e "${YELLOW}Removing existing virtual environment...${NC}"
        rm -rf "$VENV_DIR"
    else
        echo -e "${GREEN}Using existing virtual environment${NC}"
        source "$VENV_DIR/bin/activate"
        echo -e "${GREEN}✓ Virtual environment activated${NC}"
        exit 0
    fi
fi

# Create virtual environment
echo -e "\n${YELLOW}Creating virtual environment...${NC}"
python3 -m venv "$VENV_DIR"
echo -e "${GREEN}✓ Virtual environment created${NC}"

# Activate virtual environment
echo -e "\n${YELLOW}Activating virtual environment...${NC}"
source "$VENV_DIR/bin/activate"
echo -e "${GREEN}✓ Virtual environment activated${NC}"

# Upgrade pip
echo -e "\n${YELLOW}Upgrading pip...${NC}"
pip install --upgrade pip
echo -e "${GREEN}✓ pip upgraded${NC}"

# Install requirements
echo -e "\n${YELLOW}Installing project requirements...${NC}"
pip install -r "$PROJECT_ROOT/requirements.txt"
echo -e "${GREEN}✓ Requirements installed${NC}"

# Install medical_ml_sdk in development mode
echo -e "\n${YELLOW}Installing medical_ml_sdk in development mode...${NC}"
pip install -e "$PROJECT_ROOT/shared/"
echo -e "${GREEN}✓ medical_ml_sdk installed${NC}"

echo -e "\n${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}     Setup Complete!                                        ${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

echo -e "\n${YELLOW}To activate the virtual environment in the future, run:${NC}"
echo -e "  ${GREEN}source venv/bin/activate${NC}"

echo -e "\n${YELLOW}To start all services:${NC}"
echo -e "  ${GREEN}./start_all_services.sh${NC}\n"
