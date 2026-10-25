@echo off
rem Double-click to open the Hypercompute app (host a pool or join one) in your browser.
cd /d "%~dp0.."
uv run --no-sync python app\hcapp.py %*
