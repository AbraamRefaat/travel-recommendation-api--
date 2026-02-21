#!/bin/bash

echo "========================================"
echo "Tourist Recommendation API Server"
echo "========================================"
echo ""

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python 3 is not installed"
    echo "Please install Python 3.8 or higher"
    exit 1
fi

echo "Python version:"
python3 --version
echo ""

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
    echo ""
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt
echo ""

# Check if Excel file exists
if [ ! -f "Cairo_Giza_POI_Database_v3.xlsx" ]; then
    echo "WARNING: Cairo_Giza_POI_Database_v3.xlsx not found!"
    echo "Please make sure the Excel file is in this folder."
    echo ""
    read -p "Press enter to continue anyway..."
fi

# Start the server
echo "========================================"
echo "Starting API Server..."
echo "========================================"
echo "Server will be available at:"
echo "  - Local:            http://127.0.0.1:8000"
echo "  - Android Emulator: http://10.0.2.2:8000"
echo "  - iOS Simulator:    http://127.0.0.1:8000"
echo "  - API Docs:         http://127.0.0.1:8000/docs"
echo "========================================"
echo ""

python3 -m uvicorn app:app --reload --host 0.0.0.0 --port 8000
