@echo off
setlocal
REM ========================================
REM MangroVision Environment Setup Script
REM ========================================

echo.
echo ================================================
echo      Setting Up MangroVision Environment
echo ================================================
echo.

cd /d "%~dp0"

if exist "venv\Scripts\python.exe" (
    echo Existing virtual environment found.
) else (
    echo [1/3] Creating virtual environment...
    py -3 -m venv venv
    if errorlevel 1 (
        echo [WARN] 'py -3' was not available. Trying 'python -m venv venv'...
        python -m venv venv
        if errorlevel 1 (
            echo.
            echo [ERROR] Could not create the virtual environment.
            echo Install Python 3.12+ and make sure 'py' or 'python' is available in PATH.
            echo.
            pause
            exit /b 1
        )
    )
)

echo [2/3] Upgrading pip...
venv\Scripts\python.exe -m pip install --upgrade pip
if errorlevel 1 (
    echo.
    echo [ERROR] Failed while upgrading pip.
    echo.
    pause
    exit /b 1
)

echo [3/3] Installing project requirements...
venv\Scripts\python.exe -m pip install -r requirements.txt
if errorlevel 1 (
    echo.
    echo [ERROR] Failed while installing requirements.
    echo.
    pause
    exit /b 1
)

echo.
echo ================================================
echo   Setup complete. You can now run START_MANGROVISION.bat
echo ================================================
echo.
pause
