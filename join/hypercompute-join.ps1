# Share this computer with a Hypercompute host. hypercompute-join.cmd runs this. See README.md.
. (Join-Path $PSScriptRoot 'lib\hc-common.ps1')
$hc = Initialize-HcBootstrap 'join' $PSScriptRoot $args
if (-not $hc.Ok) { exit $hc.Code }
Set-Location -LiteralPath $hc.Repo
& $hc.Uv run --no-sync python join\lib\oneclick.py join --repo $hc.Repo @args
exit $LASTEXITCODE
