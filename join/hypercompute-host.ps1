# Start the Hypercompute host and its dashboard on this computer. hypercompute-host.cmd runs this. See README.md.
. (Join-Path $PSScriptRoot 'lib\hc-common.ps1')
$hc = Initialize-HcBootstrap 'host' $PSScriptRoot $args
if (-not $hc.Ok) { exit $hc.Code }
Set-Location -LiteralPath $hc.Repo
& $hc.Uv run --no-sync python join\lib\oneclick.py host --repo $hc.Repo @args
exit $LASTEXITCODE
