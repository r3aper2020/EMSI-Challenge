#!/usr/bin/env bash

# Exit immediately if a command exits with a non-zero status
set -e

# Define color codes for clean CLI outputs
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0;0m' # No Color

echo -e "${BLUE}=== ATR Model Production Workbench: Setup Center ===${NC}"

# Check for .env file
if [ ! -f ".env" ]; then
    echo -e "${YELLOW}[-] .env file not found. Cloning .env.example to .env...${NC}"
    cp .env.example .env
    echo -e "${GREEN}[+] Created .env file.${NC}"
else
    echo -e "${GREEN}[+] Existing .env file detected.${NC}"
fi

# Check for Python 3
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}[x] Python 3 is not installed. Please install Python 3 and try again.${NC}"
    exit 1
fi
echo -e "${GREEN}[+] Python 3 is installed: $(python3 --version)${NC}"

# Setup Python Virtual Environment in backend
echo -e "${BLUE}[~] Setting up Python virtual environment in backend...${NC}"
cd backend
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo -e "${GREEN}[+] Created python virtual environment (venv).${NC}"
else
    echo -e "${GREEN}[+] Virtual environment already exists.${NC}"
fi

# Activate and Install Requirements
echo -e "${BLUE}[~] Installing Python backend dependencies...${NC}"
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
deactivate
cd ..
echo -e "${GREEN}[+] Backend python packages installed successfully.${NC}"

# Check for Node and npm
if ! command -v node &> /dev/null; then
    echo -e "${RED}[x] Node.js is not installed. Please install Node.js and try again.${NC}"
    exit 1
fi
if ! command -v npm &> /dev/null; then
    echo -e "${RED}[x] npm is not installed. Please install npm and try again.${NC}"
    exit 1
fi
echo -e "${GREEN}[+] Node.js $(node --version) and npm $(npm --version) detected.${NC}"

# Install Frontend dependencies
echo -e "${BLUE}[~] Installing frontend Node packages...${NC}"
cd frontend
npm install
cd ..
echo -e "${GREEN}[+] Frontend npm packages installed successfully.${NC}"

echo -e "\n${GREEN}=== Setup Completed Successfully! ===${NC}"
echo -e "To start the application, run:"
echo -e "  ${BLUE}./start.sh${NC}\n"
