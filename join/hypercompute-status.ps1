# Show whether this computer is sharing or hosting. hypercompute-status.cmd runs this. See README.md.
. (Join-Path $PSScriptRoot 'lib\hc-common.ps1')
$hc = Initialize-HcBootstrap 'status' $PSScriptRoot $args
if (-not $hc.Ok) { exit $hc.Code }
Set-Location -LiteralPath $hc.Repo
& $hc.Uv run --no-sync python join\lib\oneclick.py status --repo $hc.Repo @args
exit $LASTEXITCODE
