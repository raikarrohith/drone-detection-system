@echo off
title Drone Detection System
cd /d "%~dp0"
echo ========================================================
echo Starting Drone Detection System...
echo ========================================================
venv\Scripts\python.exe webcam.py
pause
