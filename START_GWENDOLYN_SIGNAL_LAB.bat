@echo off
setlocal EnableExtensions
cd /d "%~dp0"
set "GWENDOLYN_SESSION_MODE=signal_lab"
if not exist ".venv\Scripts\python.exe" (
  echo Gwendolyn is not set up yet.
  echo Run SETUP_GWENDOLYN.bat first.
  pause
  exit /b 1
)
start "Gwendolyn Signal Lab" ".venv\Scripts\python.exe" gwendolyn_direct.py
