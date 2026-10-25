# Hypercompute one-click bootstrap, dot-sourced by the hypercompute-*.ps1 wrappers in join\.
# Windows PowerShell 5.1: no &&, no ternary, no ??, and no 2>&1 on native commands.
# Steps: find the checkout, find uv, create the Python environment once (it asks first). The wrapper then hands
# over to join\lib\oneclick.py. With --dry-run it prints each command instead of running it.
# It never downloads uv and never pipes an installer. It never echoes an invite, and it stops before uv when
# an invite is passed on the command line.

$HcUvDocs = 'https://docs.astral.sh/uv/getting-started/installation/'

function Get-HcMasked([string]$Text) {
    # Hide anything shaped like an invite ticket or code before it is shown.
    $masked = $Text -creplace '[A-Za-z0-9_-]{30,}', '<r>'
    return ($masked -creplace '[A-HJ-NP-Z2-9]{5}-[A-HJ-NP-Z2-9]{5}', '<code>')
}

function Test-HcInviteArg([object[]]$Rest) {
    # True when an argument looks like part of an invite: the flags, a bare ticket or a bare code.
    foreach ($item in $Rest) {
        $s = "$item"
        if ($s -match '^--(ticket|code)(=|$)') { return $true }
        if ($s -cmatch '^[A-Za-z0-9_-]{30,}$') { return $true }
        if ($s -cmatch '^[A-HJ-NP-Z2-9]{5}-[A-HJ-NP-Z2-9]{5}$') { return $true }
    }
    return $false
}

function Find-HcRepo([string]$Here) {
    if ($env:HC_REPO) { $candidate = $env:HC_REPO } else { $candidate = Join-Path $Here '..' }
    $dir = $null
    try {
        $dir = (Resolve-Path -LiteralPath $candidate -ErrorAction Stop).ProviderPath
    } catch {
        $dir = $null  # a missing folder gets the same message as a wrong one
    }
    if ($dir) {
        $pyproject = Join-Path $dir 'pyproject.toml'
        $cli = Join-Path $dir 'backend\hypercompute\cli.py'
        if ((Test-Path -LiteralPath $pyproject) -and (Test-Path -LiteralPath $cli)) {
            if (Select-String -LiteralPath $pyproject -Pattern '^name = "hypercompute"\s*$' -CaseSensitive -Quiet) {
                return $dir
            }
        }
    }
    Write-Host 'This folder is not inside a Hypercompute checkout'
    return $null
}

function Find-HcUv {
    $found = Get-Command uv -CommandType Application -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($found) { return $found.Source }
    $fallback = Join-Path $env:USERPROFILE '.local\bin\uv.exe'
    if (Test-Path -LiteralPath $fallback) { return $fallback }
    Write-Host "uv is not installed. Install it once from $HcUvDocs and then open this again."
    return $null
}

function Initialize-HcVenv([string]$Repo, [string]$Uv, [bool]$DryRun) {
    # Returns 0 when ready, 1 to stop, 3 when a dry run stops here.
    if (Test-Path -LiteralPath (Join-Path $Repo '.venv')) { return 0 }
    Write-Host "The first run downloads Python packages into $Repo\.venv. This takes a few minutes."
    Write-Host 'If this computer has no Python 3.12, uv also downloads Python 3.12 into its own folder outside'
    Write-Host 'Hypercompute. That copy stays after you delete the Hypercompute folder.'
    if ($DryRun) {
        Write-Host '[dry-run] would ask: Continue? [y/N]'
        Write-Host "[dry-run] would run: cd `"$Repo`"; & `"$Uv`" sync --frozen"
        return 3
    }
    $answer = Read-Host 'Continue? [y/N]'
    if ($answer -notmatch '^\s*(y|yes)\s*$') {
        Write-Host 'Stopped. Nothing was installed.'
        return 1
    }
    Push-Location -LiteralPath $Repo
    try {
        & $Uv sync --frozen | Out-Host
        $rc = $LASTEXITCODE
    } finally {
        Pop-Location
    }
    return $rc
}

function Initialize-HcBootstrap([string]$Action, [string]$Here, [object[]]$Rest) {
    # Returns an object: Ok (hand over to oneclick.py), Code (exit code when not Ok), Repo, Uv.
    if ($null -eq $Rest) { $Rest = @() }
    $result = [pscustomobject]@{ Ok = $false; Code = 1; Repo = $null; Uv = $null }
    if (Test-HcInviteArg $Rest) {
        Write-Host 'Run this with nothing after it, or only --dry-run. It asks you to paste the invite.'
        $result.Code = 2
        return $result
    }
    $dry = $false
    foreach ($item in $Rest) { if ("$item" -eq '--dry-run') { $dry = $true } }
    $repo = Find-HcRepo $Here
    if (-not $repo) { return $result }
    $uv = Find-HcUv
    if (-not $uv) { return $result }
    $venv = Initialize-HcVenv $repo $uv $dry
    if ($venv -eq 3) {
        Write-Host '[dry-run] stopping here: the next steps need the Python environment.'
        $result.Code = 0
        return $result
    }
    if ($venv -ne 0) {
        $result.Code = $venv
        return $result
    }
    if ($dry) {
        $line = "[dry-run] next: cd `"$repo`"; & `"$uv`" run --no-sync python join\lib\oneclick.py $Action --repo `"$repo`" " + ($Rest -join ' ')
        Write-Host (Get-HcMasked $line)
        Write-Host '[dry-run] that helper runs in dry-run mode too: it prints its commands and launches nothing.'
    }
    $result.Ok = $true
    $result.Code = 0
    $result.Repo = $repo
    $result.Uv = $uv
    return $result
}
