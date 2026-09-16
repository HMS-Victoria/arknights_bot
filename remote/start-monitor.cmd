@echo off
cd /d "%~dp0"
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0monitor.ps1" %*
if errorlevel 1 (
  echo.
  echo Monitor exited with an error. See the message above.
  pause
)
