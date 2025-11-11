#!/bin/bash
# Quick launcher for the Streamlit web interface

echo "Launching Continuous Prompting Framework Web Interface..."
echo ""

# Check if virtual environment is activated
if [[ -z "$VIRTUAL_ENV" ]]; then
    echo "Virtual environment not activated!"
    echo "Activating virtual environment..."

    if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
        source venv/Scripts/activate
    else
        source venv/bin/activate
    fi
fi

# Check if streamlit is installed
if ! python -c "import streamlit" 2>/dev/null; then
    echo "Streamlit not installed!"
    echo "Installing dependencies..."
    pip install -r requirements.txt
fi

# Launch Streamlit
echo "Starting web interface..."
echo "Your browser will open automatically"
echo "Press Ctrl+C to stop"
echo ""

streamlit run src/app/web.py

