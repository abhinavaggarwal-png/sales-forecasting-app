#!/bin/bash

# Sales Forecasting App - Setup and Run Script

echo "================================================"
echo "Sales Forecasting Dashboard - Setup Script"
echo "================================================"
echo ""

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.8 or higher."
    exit 1
fi

echo "✅ Python 3 found: $(python3 --version)"
echo ""

# Check if pip is installed
if ! command -v pip &> /dev/null && ! command -v pip3 &> /dev/null; then
    echo "❌ pip is not installed. Please install pip."
    exit 1
fi

echo "✅ pip found"
echo ""

# Install requirements
echo "📦 Installing dependencies..."
pip3 install -r requirements.txt

if [ $? -eq 0 ]; then
    echo "✅ Dependencies installed successfully"
else
    echo "❌ Failed to install dependencies"
    exit 1
fi

echo ""
echo "================================================"
echo "Setup complete! 🎉"
echo "================================================"
echo ""
echo "To run the app, execute:"
echo "  streamlit run app.py"
echo ""
echo "Or run this script with the 'run' argument:"
echo "  ./setup.sh run"
echo ""

# If 'run' argument is provided, start the app
if [ "$1" = "run" ]; then
    echo "🚀 Starting Sales Forecasting Dashboard..."
    streamlit run app.py
fi
