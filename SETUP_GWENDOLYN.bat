@echo off
setlocal EnableExtensions
cd /d "%~dp0"

echo.
echo === Gwendolyn Direct setup ===
echo.
set "PYTHON_CMD="
py -3.12 -c "import sys" >nul 2>&1 && set "PYTHON_CMD=py -3.12"
if not defined PYTHON_CMD py -3.11 -c "import sys" >nul 2>&1 && set "PYTHON_CMD=py -3.11"
if not defined PYTHON_CMD python -c "import sys; raise SystemExit(sys.version_info[:2] not in ((3,11),(3,12)))" >nul 2>&1 && set "PYTHON_CMD=python"
if not defined PYTHON_CMD (
  echo Python 3.11 or 3.12 was not found.
  echo Install 64-bit Python, enable Add Python to PATH, then run this again.
  pause
  exit /b 1
)

if not exist ".venv\Scripts\python.exe" (
  echo Creating Gwendolyn's local Python environment...
  %PYTHON_CMD% -m venv .venv
  if errorlevel 1 goto :fail
)

echo Installing speech recognition and audio dependencies...
".venv\Scripts\python.exe" -m pip install --upgrade pip
if errorlevel 1 goto :fail
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 goto :fail

for %%F in (config director_profile director_voices private_lexicon) do (
  if not exist "%%F.json" copy /y "%%F.example.json" "%%F.json" >nul
)
if not exist voices mkdir voices

echo.
echo Setup complete. Run CHECK_SETUP.bat, then START_GWENDOLYN_VECTOR.bat.
echo The Whisper speech model downloads on first use and is cached locally.
pause
exit /b 0

:fail
echo.
echo Setup failed. Review the message above.
pause
exit /b 1
