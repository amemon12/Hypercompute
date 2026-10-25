# Make a one-use invite for one friend (host only). hypercompute-invite.cmd runs this. See README.md.
. (Join-Path $PSScriptRoot 'lib\hc-common.ps1')
$hc = Initialize-HcBootstrap 'invite' $PSScriptRoot $args
if (-not $hc.Ok) { exit $hc.Code }
Set-Location -LiteralPath $hc.Repo
& $hc.Uv run --no-sync python join\lib\oneclick.py invite --repo $hc.Repo @args
exit $LASTEXITCODE
