#!/bin/bash
# Linux startup script for EC2/Ubuntu

echo "====================================="
echo "Starting Travel Recommendation API"
echo "====================================="

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Check if requirements are installed
if ! pip freeze | grep -q "fastapi"; then
    echo "Installing dependencies..."
    pip install -r requirements.txt
fi

# Check if POI file exists
if [ ! -f "Cairo_Giza_POI_Database_v3.xlsx" ]; then
    echo "ERROR: Cairo_Giza_POI_Database_v3.xlsx not found!"
    echo "Please upload the POI database file to this directory."
    exit 1
fi

# Start the API server
echo ""
echo "Starting FastAPI server..."
echo "Server will be available at: http://0.0.0.0:8000"
echo "Press Ctrl+C to stop"
echo ""

python api.py
