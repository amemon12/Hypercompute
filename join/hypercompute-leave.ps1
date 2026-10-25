# Stop sharing this computer. hypercompute-leave.cmd runs this. See README.md.
. (Join-Path $PSScriptRoot 'lib\hc-common.ps1')
$hc = Initialize-HcBootstrap 'leave' $PSScriptRoot $args
if (-not $hc.Ok) { exit $hc.Code }
Set-Location -LiteralPath $hc.Repo
& $hc.Uv run --no-sync python join\lib\oneclick.py leave --repo $hc.Repo @args
exit $LASTEXITCODE
