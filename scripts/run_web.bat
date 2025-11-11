@echo off
REM Quick launcher for the Streamlit web interface (Windows)

echo Launching Continuous Prompting Framework Web Interface...
echo.

REM Check if virtual environment exists
if not exist "venv\Scripts\activate.bat" (
    echo Virtual environment not found!
    echo Please run setup.sh first or create venv manually
    pause
    exit /b 1
)

REM Activate virtual environment
call venv\Scripts\activate.bat

REM Check if streamlit is installed
python -c "import streamlit" 2>nul
if errorlevel 1 (
    echo Streamlit not installed!
    echo Installing dependencies...
    pip install -r requirements.txt
)

REM Launch Streamlit
echo Starting web interface...
echo Your browser will open automatically
echo Press Ctrl+C to stop
echo.

streamlit run src\app\web.py

