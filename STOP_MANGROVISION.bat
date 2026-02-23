@echo off
REM ========================================
REM MangroVision Stop Script
REM ========================================

echo.
echo ================================================
echo      Stopping MangroVision System
echo ================================================
echo.

echo Stopping all Python processes for MangroVision...

REM Kill tile server and streamlit
taskkill /FI "WindowTitle eq MangroVision Tile Server*" /F >nul 2>&1
taskkill /FI "IMAGENAME eq streamlit.exe" /F >nul 2>&1

REM Also kill Python processes on ports 8080 and 8502
for /f "tokens=5" %%a in ('netstat -aon ^| findstr :8080 ^| findstr LISTENING') do taskkill /F /PID %%a >nul 2>&1
for /f "tokens=5" %%a in ('netstat -aon ^| findstr :8502 ^| findstr LISTENING') do taskkill /F /PID %%a >nul 2>&1

echo.
echo ================================================
echo   MangroVision stopped successfully
echo ================================================
echo.
pause
