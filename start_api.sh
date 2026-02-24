#!/bin/bash

echo "================================================"
echo "   Nile Quest - Starting API Server"
echo "================================================"
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
echo ""

# Install/update dependencies
echo "Installing dependencies..."
pip install -r requirements.txt
echo ""

# Start the server
echo "Starting API server on http://localhost:8000"
echo ""
echo "Press Ctrl+C to stop the server"
echo "================================================"
python api_server.py
