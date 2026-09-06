@echo off
setlocal EnableExtensions
cd /d "%~dp0"
set "GWENDOLYN_SESSION_MODE=vector"
if not exist ".venv\Scripts\python.exe" (
  echo Gwendolyn is not set up yet. Run SETUP_GWENDOLYN.bat first.
  pause
  exit /b 1
)
".venv\Scripts\python.exe" gwendolyn_direct.py
pause
