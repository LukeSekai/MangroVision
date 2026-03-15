@echo off
REM ========================================
REM Stop MangroVision Web Map System
REM ========================================

echo.
echo Stopping MangroVision Web Map System...
echo.

REM Kill Python processes (will stop both backend and tile server)
taskkill /F /FI "WINDOWTITLE eq Tile Server*" 2>nul
taskkill /F /FI "WINDOWTITLE eq Backend API*" 2>nul

echo.
echo MangroVision Web Map stopped.
echo.
pause
