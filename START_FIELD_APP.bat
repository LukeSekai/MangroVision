@echo off
REM ========================================
REM MangroVision Field App Startup Script
REM ========================================

setlocal
cd /d "%~dp0"

echo.
echo ================================================
echo      Starting MangroVision Field App
echo ================================================
echo.

if not exist "venv\Scripts\python.exe" (
    echo Missing Python environment: venv\Scripts\python.exe
    echo Create or restore the project virtual environment first.
    pause
    exit /b 1
)

echo Enter this PC's LAN IPv4 address for phone access.
echo Use the Wi-Fi adapter address from ipconfig, not a VMware or virtual adapter.
echo Example: 192.168.18.66
set /p LAN_IP=LAN IPv4 address: 

if not defined LAN_IP (
    echo No LAN IPv4 address was entered.
    pause
    exit /b 1
)

set "MANGROVISION_TILE_SERVER_BASE_URL=http://%LAN_IP%:8080"

echo [1/2] Starting Tile Server on port 8080...
start "MangroVision Tile Server" cmd /c "venv\Scripts\python.exe start_tile_server.py"
timeout /t 2 /nobreak >nul

echo [2/2] Starting Field App on port 8503...
echo.
echo ================================================
echo   Field App URL for phones on the same Wi-Fi:
echo   http://%LAN_IP%:8503
echo.
echo   Tile Server URL:
echo   %MANGROVISION_TILE_SERVER_BASE_URL%
echo.
echo   Note: iPhone geolocation may still require HTTPS.
echo   Press Ctrl+C to stop the field app.
echo ================================================
echo.

venv\Scripts\python.exe -m streamlit run field_app.py --server.port 8503 --server.address 0.0.0.0

echo.
echo MangroVision Field App stopped.
pause
