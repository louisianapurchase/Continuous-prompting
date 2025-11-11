#!/bin/bash
# Quick setup script for Continuous Prompting Framework

echo "=========================================="
echo "Continuous Prompting Framework Setup"
echo "=========================================="

# Check if Python is installed
if ! command -v python &> /dev/null; then
    echo "Error: Python is not installed. Please install Python 3.8 or higher."
    exit 1
fi

echo "Python version:"
python --version

# Create virtual environment
echo ""
echo "Creating virtual environment..."
python -m venv venv

# Activate virtual environment
echo "Activating virtual environment..."
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
    source venv/Scripts/activate
else
    source venv/bin/activate
fi

# Install dependencies
echo ""
echo "Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Check if Ollama is installed
echo ""
echo "Checking for Ollama..."
if ! command -v ollama &> /dev/null; then
    echo "Warning: Ollama is not installed or not in PATH."
    echo "Please install Ollama from: https://ollama.ai/"
    echo "After installation, run: ollama pull mistral"
else
    echo "Ollama found!"
    echo ""
    echo "Available models:"
    ollama list
    echo ""
    echo "If you don't see 'mistral' or another model, run:"
    echo "  ollama pull mistral"
fi

# Create necessary directories
echo ""
echo "Creating directories..."
mkdir -p logs/conversations
mkdir -p logs/metrics
mkdir -p outputs

echo ""
echo "=========================================="
echo "Setup complete!"
echo "=========================================="
echo ""
echo "To get started:"
echo ""
echo "Option 1: Web Interface (Recommended)"
echo "  python run_web.py"
echo ""
echo "Option 2: Command Line"
echo "  1. Activate the virtual environment:"
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
    echo "     source venv/Scripts/activate"
else
    echo "     source venv/bin/activate"
fi
echo "  2. Make sure Ollama is running"
echo "  3. Run the framework:"
echo "     python run.py"
echo ""
echo "Documentation:"
echo "  - docs/WEB_INTERFACE_GUIDE.md - Web UI guide"
echo "  - docs/QUICKSTART.md - Quick start guide"
echo "  - README.md - Full documentation"
echo ""
echo "To customize experiments, edit config.yaml"
echo "=========================================="

