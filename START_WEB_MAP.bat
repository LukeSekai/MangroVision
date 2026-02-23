@echo off
REM ========================================
REM MangroVision Web Map Startup Script
REM ========================================

echo.
echo ================================================
echo   Starting MangroVision Web Map System
echo ================================================
echo.

cd /d "%~dp0"

REM Start Tile Server in background
echo [1/2] Starting Tile Server on port 8080...
start "Tile Server" cmd /k "venv\Scripts\python.exe start_tile_server.py"
timeout /t 3 /nobreak >nul

REM Start Backend API in background
echo [2/2] Starting Backend API on port 8000...
start "Backend API" cmd /k "venv\Scripts\python.exe map_backend.py"
timeout /t 3 /nobreak >nul

echo.
echo ================================================
echo   MangroVision Web Map is READY!
echo ================================================
echo.
echo   Backend API: http://localhost:8000
echo   Tile Server: http://localhost:8080
echo.
echo   Now opening the web interface...
echo ================================================
echo.

REM Open the web interface
timeout /t 2 /nobreak >nul
start map_frontend.html

echo.
echo Web interface opened in your browser!
echo.
echo To stop the system:
echo   1. Close the "Tile Server" window
echo   2. Close the "Backend API" window
echo   OR run STOP_WEB_MAP.bat
echo.
pause
