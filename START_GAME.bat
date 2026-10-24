@echo off
title GestureGame - Cyber Surfer Runner
echo ========================================================
echo   STARTING CYBER SURFER GESTURE GAME...
echo ========================================================
echo.
cd /d "%~dp0"
if exist ".\venv\Scripts\python.exe" (
    .\venv\Scripts\python.exe run_game.py
) else (
    python run_game.py
)
pause
