@echo off
REM Cybersecurity Application Startup Script for Windows
REM This script starts the application with WiFi accessibility

echo ========================================================
echo   🛡️  Cybersecurity Threat Predictor
echo ========================================================
echo.
echo 🚀 Starting application...
echo.

REM Change to the script directory
cd /d "%~dp0"

REM Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Error: Python is not installed or not in PATH
    echo.
    echo 📥 Please install Python from: https://python.org
    echo.
    pause
    exit /b 1
)

REM Check if required packages are installed
echo 📦 Checking dependencies...
python -c "import streamlit" >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Streamlit not found. Installing required packages...
    echo.
    pip install streamlit pandas numpy plotly scikit-learn xgboost lightgbm
    echo.
)

REM Start the application
echo 🌐 Starting server...
echo.
python start_app.py

REM Keep window open if there's an error
if %errorlevel% neq 0 (
    echo.
    echo ❌ Application stopped with an error
    pause
)
