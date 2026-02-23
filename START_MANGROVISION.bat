@echo off
REM ========================================
REM MangroVision Startup Script
REM ========================================

echo.
echo ================================================
echo      Starting MangroVision System
echo ================================================
echo.

cd /d "%~dp0"

REM Start Tile Server in background
echo [1/2] Starting Tile Server on port 8080...
start "MangroVision Tile Server" cmd /c "venv\Scripts\python.exe start_tile_server.py"
timeout /t 2 /nobreak >nul

REM Start Streamlit App
echo [2/2] Starting MangroVision App on port 8502...
echo.
echo ================================================
echo   MangroVision is starting...
echo   Your browser will open automatically
echo ================================================
echo.
echo   Local URL: http://localhost:8502
echo   Press Ctrl+C to stop the app
echo ================================================
echo.

venv\Scripts\python.exe -m streamlit run app.py

echo.
echo MangroVision stopped.
pause
