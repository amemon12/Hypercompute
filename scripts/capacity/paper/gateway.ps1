$ErrorActionPreference = 'Stop'
$env:HYPERCOMPUTE_HOME = 'C:\Users\operator\.hypercompute\iroh-session'
$env:HYPERCOMPUTE_LLAMA_BIN_DIR = 'C:\Users\operator\.hypercompute\engines\llama-b11770-windows-x64-cuda13.4'
$env:HYPERCOMPUTE_CONFIG_DIR = 'C:\Users\operator\.hypercompute\worktrees\live-host\configs'
$env:HYPERCOMPUTE_FRONTEND_DIST = 'C:\Users\operator\OneDrive\Documents\ChatGPT\CalHacks 13.0\hypercompute\frontend\dist'
Set-Location -LiteralPath 'C:\Users\operator\.hypercompute\worktrees\live-host'
& 'C:\Users\operator\.local\bin\uv.exe' run --no-sync hypercompute host --transport iroh --port 8001 --no-browser
exit $LASTEXITCODE
