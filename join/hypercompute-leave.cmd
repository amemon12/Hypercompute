@echo off
setlocal
"%SystemRoot%\System32\WindowsPowerShell\v1.0\powershell.exe" -NoProfile -ExecutionPolicy Bypass -File "%~dp0hypercompute-leave.ps1" %*
set rc=%ERRORLEVEL%
if not "%HC_NO_PAUSE%"=="1" pause
exit /b %rc%
