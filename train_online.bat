@echo off
title Online Incremental Training - Drone Defense System
cd /d "%~dp0"
echo ========================================================
echo  Drone Defense System - Online Incremental Training
echo ========================================================
echo.
echo Starting automated fine-tuning on custom captured samples...
echo.
venv\Scripts\python.exe train_online.py %*
echo.
echo Training complete. Press [U] in webcam.py to reload weights!
pause
