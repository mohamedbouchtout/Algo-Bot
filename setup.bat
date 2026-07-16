@echo off
setlocal enabledelayedexpansion

cd /d "%~dp0"

set VENV_DIR=.venv
set CONFIG_FILE=config\config.json
set ENV_FILE=.env

echo ===========================================
echo   Algo-Bot Setup
echo ===========================================
echo.

:: --- Check Python ---
set PYTHON=
for %%C in (python py) do (
    where %%C >nul 2>&1
    if !errorlevel! equ 0 (
        for /f "tokens=*" %%V in ('%%C -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2^>nul') do set PY_VER=%%V
        for /f "tokens=*" %%V in ('%%C -c "import sys; print(sys.version_info.major)" 2^>nul') do set PY_MAJOR=%%V
        for /f "tokens=*" %%V in ('%%C -c "import sys; print(sys.version_info.minor)" 2^>nul') do set PY_MINOR=%%V
        if !PY_MAJOR! geq 3 if !PY_MINOR! geq 11 (
            set PYTHON=%%C
            goto :found_python
        )
    )
)

echo ERROR: Python 3.11 or higher is required but not found.
echo Please install Python from https://www.python.org/downloads/
exit /b 1

:found_python
for /f "tokens=*" %%V in ('!PYTHON! --version') do echo Found Python: %%V

:: --- Create virtual environment ---
if not exist "%VENV_DIR%" (
    echo.
    echo Creating virtual environment...
    !PYTHON! -m venv %VENV_DIR%
    echo Virtual environment created.
) else (
    echo Virtual environment already exists.
)

:: --- Activate venv ---
call %VENV_DIR%\Scripts\activate.bat

:: --- Install dependencies ---
echo.
echo Installing dependencies (this may take a few minutes)...
pip install --upgrade pip setuptools wheel -q
pip install -r requirements.txt -q
echo Dependencies installed.

:: --- Configure email alerts ---
if not exist "%ENV_FILE%" (
    echo.
    echo ===========================================
    echo   Email Alert Configuration
    echo ===========================================
    echo.
    echo The bot can send email alerts for trades, errors, and daily summaries.
    echo This requires a Gmail account with an App Password.
    echo (See: https://support.google.com/accounts/answer/185833^)
    echo.

    set /p configure_email="Do you want to configure email alerts? (y/n): "

    if /i "!configure_email!"=="y" (
        echo.
        set /p gmail_user="Gmail address (sender): "
        set /p gmail_password="Gmail App Password: "
        set /p recipient_email="Recipient email (where alerts are sent): "

        :: Write .env file
        (
            echo GMAIL_USER=!gmail_user!
            echo GMAIL_PASSWORD=!gmail_password!
        ) > "%ENV_FILE%"

        echo.
        echo .env file created.

        :: Update config.json with recipient email
        set "RECIPIENT_EMAIL=!recipient_email!"
        !PYTHON! -c "import json, os; f=open('%CONFIG_FILE%'); c=json.load(f); f.close(); c['alerts']['email']=os.environ['RECIPIENT_EMAIL']; c['alerts']['enabled']=True; f=open('%CONFIG_FILE%','w'); json.dump(c,f,indent=4); f.close()"

        echo config.json updated with alert settings.
    ) else (
        echo.
        echo Skipping email configuration.
        echo You can configure it later by creating a .env file (see .env.example^).
    )
) else (
    echo.
    echo .env file already exists, skipping email configuration.
)

echo.
echo ===========================================
echo   Setup Complete!
echo ===========================================
echo.
echo To start the bot:
echo   run.bat
echo.
echo Make sure IB Gateway or TWS is running with API enabled before starting.
echo.

endlocal
