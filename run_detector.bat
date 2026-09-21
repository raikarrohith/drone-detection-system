@echo off
title Drone Detection System
cd /d "%~dp0"
echo ========================================================
echo Starting Drone Detection System...
echo ========================================================
if "%~1"=="" (
    venv\Scripts\python.exe webcam.py
) else (
    echo Loading input video: "%~1"
    venv\Scripts\python.exe webcam.py --video "%~1"
)
pause
