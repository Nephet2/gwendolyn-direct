@echo off
setlocal EnableExtensions
cd /d "%~dp0"
if not exist ".venv-kokoro\Scripts\python.exe" (
  echo Run SETUP_KOKORO.bat first.
  pause
  exit /b 1
)
".venv-kokoro\Scripts\python.exe" bridges\kokoro_bridge.py
