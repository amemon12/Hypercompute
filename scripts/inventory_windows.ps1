#requires -Version 5.1

<#
.SYNOPSIS
  Hypercompute raw hardware inventory for Windows (Windows PowerShell 5.1 or PowerShell 7).

.DESCRIPTION
  Read-only. Writes exactly one JSON document to the pipeline (stdout under "powershell -File"), or to
  -OutFile (pure ASCII, so the file is also BOM-free UTF-8; non-ASCII is \u-escaped):
    {"schema": 1, "platform": "windows", "script": "inventory_windows.ps1", "collected_at": "<UTC ISO-8601>",
     "probes": {"<name>": {"cmd": [...], "exit_code": <int|null>, "stdout": "...", "stderr": "...",
                           "error": "..." (tool missing or timed out), "note": "..." (caveat)}}}
  Cmdlet probes emit JSON text in stdout and use exit_code 0 (ok) / 1 (cmdlet error); native probes
  report the process exit code. A missing tool gets exit_code null and an error string.
  No serial numbers, UUIDs, MAC addresses or Wi-Fi network names are selected.

.PARAMETER OutFile
  Write the JSON document to this file (ASCII, no BOM) instead of the pipeline. Use it to save the
  inventory from Windows PowerShell 5.1, which re-encodes any text that passes through it: "> file" and
  Out-File write UTF-16LE with a BOM, and piping into a native program (... | python) prepends a UTF-8 BOM
  to that program's stdin for each of [Console]::InputEncoding and $OutputEncoding that carries one
  (console code page 65001 does). Strict JSON readers reject both. The script's own stdout has no BOM.
  The directory must already exist; it is checked before any probe runs.

.EXAMPLE
  PS C:\> powershell -NoProfile -ExecutionPolicy Bypass -File scripts\inventory_windows.ps1 -OutFile inventory.json

.EXAMPLE
  PS C:\> $inventory = & .\scripts\inventory_windows.ps1 | ConvertFrom-Json
  In-process the JSON text goes to the pipeline; add -OutFile to save it as a file. In-process also means
  the session's execution policy applies and must let this unsigned script run (Restricted, the Windows
  client default, does not): start the session with powershell -ExecutionPolicy Bypass, or first run
  Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass. Example 1 does not depend on the policy.

.EXAMPLE
  PS C:\> cmd /c "powershell -NoProfile -ExecutionPolicy Bypass -File scripts\inventory_windows.ps1 > inventory.json"
  cmd.exe redirection and a program reading the process's stdout bytes keep the output byte-exact.
#>
[CmdletBinding()]
param(
    [ValidateRange(1, 600)]
    [int]$TimeoutSeconds = 45,

    [ValidateNotNullOrEmpty()]
    [string]$OutFile
)

Set-StrictMode -Version 2.0
$ErrorActionPreference = 'Stop'

# Resolve -OutFile against the PowerShell location (not the process directory) and fail before probing.
$OutFilePath = $null
if ($PSBoundParameters.ContainsKey('OutFile')) {
    $outProvider = $null
    $outDrive = $null
    $OutFilePath = $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($OutFile, [ref]$outProvider, [ref]$outDrive)
    if ($outProvider.Name -ne 'FileSystem') {
        throw "-OutFile must be a file system path (got provider '$($outProvider.Name)' for '$OutFile')"
    }
    $outDir = [IO.Path]::GetDirectoryName($OutFilePath)
    if ($outDir -and -not [IO.Directory]::Exists($outDir)) {
        throw "-OutFile directory does not exist: $outDir"
    }
}

$script:Probes = [ordered]@{}

function Add-Probe {
    param([string]$Name, [string[]]$Cmd, $ExitCode, [string]$Stdout, [string]$Stderr, [string]$ErrorText, [string]$Note)
    $entry = [ordered]@{
        cmd       = [object[]]$Cmd
        exit_code = $ExitCode
        stdout    = $Stdout
        stderr    = $Stderr
    }
    if ($ErrorText) { $entry['error'] = $ErrorText }
    if ($Note) { $entry['note'] = $Note }
    $script:Probes[$Name] = $entry
}

function ConvertTo-NativeArgument {
    param([string]$Value)
    if ($Value -match '[\s"]') { return '"' + ($Value -replace '"', '\"') + '"' }
    return $Value
}

# Runs an executable with redirected streams and a timeout (no PowerShell stderr-to-ErrorRecord wrapping).
function Invoke-NativeProbe {
    param([string]$Name, [string]$Exe, [string[]]$Arguments, [string]$Note)
    $cmd = @($Exe) + $Arguments
    $app = Get-Command -Name $Exe -CommandType Application -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($null -eq $app) {
        Add-Probe -Name $Name -Cmd $cmd -ExitCode $null -Stdout '' -Stderr '' -ErrorText "not found: $Exe" -Note $Note
        return
    }
    $psi = New-Object System.Diagnostics.ProcessStartInfo
    $psi.FileName = $app.Path
    $psi.Arguments = (($Arguments | ForEach-Object { ConvertTo-NativeArgument $_ }) -join ' ')
    $psi.UseShellExecute = $false
    $psi.RedirectStandardOutput = $true
    $psi.RedirectStandardError = $true
    $psi.CreateNoWindow = $true
    try {
        $proc = [System.Diagnostics.Process]::Start($psi)
    } catch {
        Add-Probe -Name $Name -Cmd $cmd -ExitCode $null -Stdout '' -Stderr '' -ErrorText ("failed to start: " + $_.Exception.Message) -Note $Note
        return
    }
    $outTask = $proc.StandardOutput.ReadToEndAsync()
    $errTask = $proc.StandardError.ReadToEndAsync()
    if (-not $proc.WaitForExit($TimeoutSeconds * 1000)) {
        $killNote = ''
        try { $proc.Kill() } catch { $killNote = ' (kill failed: ' + $_.Exception.Message + ')' }
        Add-Probe -Name $Name -Cmd $cmd -ExitCode $null -Stdout '' -Stderr '' -ErrorText ("timed out after ${TimeoutSeconds}s" + $killNote) -Note $Note
        return
    }
    $proc.WaitForExit()  # drains the async readers
    Add-Probe -Name $Name -Cmd $cmd -ExitCode $proc.ExitCode -Stdout $outTask.Result -Stderr $errTask.Result -Note $Note
}

# Runs a PowerShell pipeline (given as text so the recorded cmd is exactly what ran).
function Invoke-CmdletProbe {
    param([string]$Name, [string]$Pipeline, [string]$Requires, [string]$Note)
    $cmd = @('powershell', '-NoProfile', '-Command', $Pipeline)
    if ($Requires -and -not (Get-Command -Name $Requires -ErrorAction SilentlyContinue)) {
        Add-Probe -Name $Name -Cmd $cmd -ExitCode $null -Stdout '' -Stderr '' -ErrorText "not found: $Requires" -Note $Note
        return
    }
    try {
        $out = & ([scriptblock]::Create($Pipeline))
        Add-Probe -Name $Name -Cmd $cmd -ExitCode 0 -Stdout ([string]$out) -Stderr '' -Note $Note
    } catch {
        Add-Probe -Name $Name -Cmd $cmd -ExitCode 1 -Stdout '' -Stderr $_.Exception.Message -Note $Note
    }
}

# ------------------------------------------------------------------ probes

Invoke-CmdletProbe -Name 'computer_system' -Requires 'Get-CimInstance' -Pipeline (
    'Get-CimInstance -ClassName Win32_ComputerSystem | ' +
    'Select-Object Manufacturer, Model, TotalPhysicalMemory | ConvertTo-Json -Compress')

Invoke-CmdletProbe -Name 'operating_system' -Requires 'Get-CimInstance' -Note 'FreePhysicalMemory is in KiB.' -Pipeline (
    'Get-CimInstance -ClassName Win32_OperatingSystem | ' +
    'Select-Object Caption, Version, BuildNumber, OSArchitecture, FreePhysicalMemory | ConvertTo-Json -Compress')

Invoke-CmdletProbe -Name 'video_controllers' -Requires 'Get-CimInstance' `
    -Note 'AdapterRAM is a uint32 and is not authoritative above 4 GiB; use nvidia-smi memory.total or the engine device list for VRAM.' `
    -Pipeline (
    'ConvertTo-Json -Compress -InputObject @(Get-CimInstance -ClassName Win32_VideoController | ' +
    'Select-Object Name, AdapterCompatibility, DriverVersion, AdapterRAM)')

Invoke-CmdletProbe -Name 'processors' -Requires 'Get-CimInstance' -Note 'Architecture: 9 = x64, 12 = ARM64.' -Pipeline (
    'ConvertTo-Json -Compress -InputObject @(Get-CimInstance -ClassName Win32_Processor | ' +
    'Select-Object Name, Manufacturer, NumberOfCores, NumberOfLogicalProcessors, MaxClockSpeed, Architecture)')

Invoke-CmdletProbe -Name 'net_adapters' -Requires 'Get-NetAdapter' -Pipeline (
    'ConvertTo-Json -Compress -InputObject @(Get-NetAdapter | ' +
    'Select-Object Name, Status, LinkSpeed, InterfaceDescription)')

Invoke-CmdletProbe -Name 'net_connection_profiles' -Requires 'Get-NetConnectionProfile' -Pipeline (
    'ConvertTo-Json -Compress -InputObject @(Get-NetConnectionProfile | ' +
    'Select-Object InterfaceAlias, @{Name = ''NetworkCategory''; Expression = { [string]$_.NetworkCategory }})')

Invoke-CmdletProbe -Name 'net_ipv4' -Requires 'Get-NetIPAddress' -Pipeline (
    'ConvertTo-Json -Compress -InputObject @(Get-NetIPAddress -AddressFamily IPv4 | ' +
    'Select-Object InterfaceAlias, IPAddress, PrefixLength, @{Name = ''AddressState''; Expression = { [string]$_.AddressState }})')

Invoke-NativeProbe -Name 'tcp_excluded_port_ranges' -Exe 'netsh' -Arguments @('int', 'ipv4', 'show', 'excludedportrange', 'protocol=tcp') `
    -Note 'Hyper-V/WSL/Docker reserve TCP ranges here; a worker port inside one cannot be bound.'

$profileDrive = Split-Path -Qualifier $env:USERPROFILE
if ($profileDrive -notmatch '^[A-Za-z]:$') {
    Add-Probe -Name 'profile_drive_free' -Cmd @('Split-Path', '-Qualifier', '$env:USERPROFILE') -ExitCode $null `
        -Stdout '' -Stderr '' -ErrorText "unexpected profile drive qualifier '$profileDrive'"
} else {
    Invoke-CmdletProbe -Name 'profile_drive_free' -Requires 'Get-CimInstance' -Note 'FreeSpace and Size are bytes.' -Pipeline (
        "Get-CimInstance -ClassName Win32_LogicalDisk -Filter `"DeviceID='$profileDrive'`" | " +
        'Select-Object DeviceID, FreeSpace, Size | ConvertTo-Json -Compress')
}

Invoke-NativeProbe -Name 'nvidia_smi' -Exe 'nvidia-smi' `
    -Arguments @('--query-gpu=name,memory.total,memory.free,driver_version,compute_cap', '--format=csv')

# ------------------------------------------------------------------ output

$doc = [ordered]@{
    schema       = 1
    platform     = 'windows'
    script       = 'inventory_windows.ps1'
    collected_at = [DateTime]::UtcNow.ToString("yyyy-MM-dd'T'HH:mm:ss'Z'", [Globalization.CultureInfo]::InvariantCulture)
    probes       = $script:Probes
}
$json = ConvertTo-Json -InputObject $doc -Depth 8
# Escape every non-ASCII UTF-16 unit so the bytes are identical whatever the console code page is.
$json = [regex]::Replace($json, '[^\x00-\x7F]', { param($m) '\u{0:x4}' -f [int][char]$m.Value })
if ($OutFilePath) {
    # Raw bytes, no BOM: the text is ASCII, so this is also valid UTF-8 for any JSON reader.
    [IO.File]::WriteAllText($OutFilePath, $json + "`n", (New-Object System.Text.UTF8Encoding $false))
} else {
    # The pipeline, not [Console]::Out: in-process callers ($x = & .\inventory_windows.ps1) get the text
    # instead of it bypassing their redirection, and under "powershell -File" the console host writes it
    # to stdout unchanged (ASCII, no BOM, no line wrapping, then CRLF).
    Write-Output $json
}
