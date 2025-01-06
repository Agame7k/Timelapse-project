@echo off
title Timelapse Bot

:: Check for virtual environment
if not exist "venv" (
    echo Creating Python virtual environment...
    python -m venv venv
    call venv\Scripts\activate
    pip install -r requirements.txt
) else (
    call venv\Scripts\activate
)

:: Check for environment file
if not exist "cred.env" (
    echo Error: cred.env not found! Please create it using the template.
    pause
    exit /b 1
)

:: Create required directories
mkdir timelapse_photos_primary\camera0 2>nul
mkdir timelapse_photos_backup\camera0 2>nul

:: Start the bot
echo Starting Timelapse Bot...
python Timelapse.py
pause