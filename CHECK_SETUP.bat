@echo off
setlocal EnableExtensions
cd /d "%~dp0"
if exist ".venv\Scripts\python.exe" (
  ".venv\Scripts\python.exe" check_setup.py
) else (
  py -3 check_setup.py 2>nul || python check_setup.py
)
echo.
pause
