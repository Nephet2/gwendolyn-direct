@echo off
setlocal EnableExtensions
cd /d "%~dp0"
echo === Optional Kokoro voice setup ===
py -3.12 -m venv .venv-kokoro 2>nul
if not exist ".venv-kokoro\Scripts\python.exe" py -3.11 -m venv .venv-kokoro 2>nul
if not exist ".venv-kokoro\Scripts\python.exe" python -m venv .venv-kokoro
if not exist ".venv-kokoro\Scripts\python.exe" goto :fail
".venv-kokoro\Scripts\python.exe" -m pip install --upgrade pip
if errorlevel 1 goto :fail
".venv-kokoro\Scripts\python.exe" -m pip install "kokoro>=0.9.4" soundfile
if errorlevel 1 goto :fail
echo Kokoro is installed. Use START_KOKORO.bat before Gwendolyn.
pause
exit /b 0
:fail
echo Kokoro setup failed. Review the messages above.
pause
exit /b 1
