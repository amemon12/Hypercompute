#requires -Version 5.1
<#
.SYNOPSIS
  Build llama.cpp from source on Windows x64 at the commit pinned in configs\engines.lock.json:
  llama-server, llama-cli and ggml-rpc-server with RPC enabled, CUDA when nvcc is found, else CPU.

.PARAMETER SourceDir
  Clone/checkout location. Default: $HOME\.hypercompute\src\llama.cpp

.PARAMETER CudaArchitectures
  CMAKE_CUDA_ARCHITECTURES override, checked against `nvcc --list-gpu-arch`:
  120 = RTX 50-series (sm_120, CUDA >= 12.8), 121 = GB10 (sm_121, CUDA >= 12.9).
  llama.cpp rewrites a plain 12X to 12Xa. Default: llama.cpp's own choice ("native").

.PARAMETER Cpu
  Build CPU-only even though an NVIDIA GPU is present.

.PARAMETER Jobs
  Parallel build jobs (default: logical processor count).

.PARAMETER Generator
  CMake Visual Studio generator. Default: 'Visual Studio 17 2022'.

.PARAMETER CheckOnly
  Only run the prerequisite checks.

.EXAMPLE
  powershell -NoProfile -ExecutionPolicy Bypass -File scripts\build_llama_windows.ps1
.EXAMPLE
  powershell -NoProfile -ExecutionPolicy Bypass -File scripts\build_llama_windows.ps1 -CudaArchitectures 120
#>
[CmdletBinding()]
param(
    [string]$SourceDir = (Join-Path $HOME '.hypercompute\src\llama.cpp'),
    [string]$CudaArchitectures = '',
    [switch]$Cpu,
    [ValidateRange(1, 256)]
    [int]$Jobs = [Environment]::ProcessorCount,
    [ValidatePattern('^Visual Studio \d+ \d{4}$')]
    [string]$Generator = 'Visual Studio 17 2022',
    [switch]$CheckOnly
)

Set-StrictMode -Version 2.0
# Native tools (git, cmake, msbuild) write progress to stderr; Windows PowerShell 5.1 would turn that into
# terminating errors under 'Stop'. Every native call below checks $LASTEXITCODE instead.
$ErrorActionPreference = 'Continue'

$FallbackCommit = '39da130fcd1b6ca47d3c0522a40eccff41ea3dfb'
$FallbackBuildNumber = '11770'
$FallbackRepo = 'https://github.com/ggml-org/llama.cpp'
$Targets = @('llama-server', 'llama-cli', 'ggml-rpc-server')
$MinCmake = [version]'3.18'

function Write-Log([string]$Message) { Write-Host "[hypercompute] $Message" }
function Stop-Build([string]$Message, [int]$Code = 1) {
    Write-Host "[hypercompute] ERROR: $Message" -ForegroundColor Red
    exit $Code
}
function Get-JsonValue($Node, [string[]]$Path) {
    foreach ($part in $Path) {
        if ($null -eq $Node) { return $null }
        $prop = $Node.PSObject.Properties[$part]
        if ($null -eq $prop) { return $null }
        $Node = $prop.Value
    }
    return $Node
}
function Find-Exe([string]$Name) {
    $cmd = Get-Command -Name $Name -CommandType Application -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($null -eq $cmd) { return $null }
    return $cmd.Path
}
# Test-Path -LiteralPath, not Get-Command on a joined path: Get-Command reads the [ ] in a directory such as
# "C:\tools [x64]" as a wildcard. Extensions are tried in PATHEXT order.
function Find-ExeIn([string]$Dir, [string]$Name) {
    $exts = if ($env:PATHEXT) { $env:PATHEXT -split ';' } else { @('.exe', '.bat', '.cmd') }
    foreach ($ext in $exts) {
        if (-not $ext) { continue }
        $path = Join-Path $Dir ($Name + $ext.ToLowerInvariant())
        if (Test-Path -LiteralPath $path -PathType Leaf) { return $path }
    }
    return $null
}
# `uv tool install cmake` needs no admin rights but puts cmake.exe in uv's tool bin dir, which is often not
# on PATH until `uv tool update-shell` and a new terminal. Returns @{Path; Source} (which dir it was found in) or $null.
function Find-UvToolExe([string]$Name) {
    $dirs = [ordered]@{}
    $uv = Find-Exe 'uv'
    if ($uv) {
        $uvOut = @(& $uv tool dir --bin 2>$null)
        if ($LASTEXITCODE -eq 0 -and $uvOut.Count -gt 0 -and "$($uvOut[0])".Trim()) { $dirs["uv's tool bin dir"] = "$($uvOut[0])".Trim() }
    }
    $dirs['$HOME\.local\bin'] = Join-Path $HOME '.local\bin'
    foreach ($source in $dirs.Keys) {
        $path = Find-ExeIn $dirs[$source] $Name
        if ($path) { return @{ Path = $path; Source = $source } }
    }
    return $null
}

if ($CudaArchitectures -and $CudaArchitectures -notmatch '^\d{2,3}[af]?(-real|-virtual)?(;\d{2,3}[af]?(-real|-virtual)?)*$') {
    Stop-Build "-CudaArchitectures '$CudaArchitectures' is not a CMAKE_CUDA_ARCHITECTURES list such as 120 or 121 or 89;120"
}
if ($Cpu -and $CudaArchitectures) { Stop-Build '-Cpu and -CudaArchitectures contradict each other' }

# ------------------------------------------------------------------ pinned engine identity

$configDir = if ($env:HYPERCOMPUTE_CONFIG_DIR) { $env:HYPERCOMPUTE_CONFIG_DIR } else { Join-Path (Split-Path -Parent $PSScriptRoot) 'configs' }
$lockFile = Join-Path $configDir 'engines.lock.json'
if (Test-Path -LiteralPath $lockFile) {
    try {
        $lock = Get-Content -LiteralPath $lockFile -Raw | ConvertFrom-Json
    } catch {
        Stop-Build "cannot parse ${lockFile}: $($_.Exception.Message)"
    }
    $commit = Get-JsonValue $lock @('llama_cpp', 'source_build', 'commit')
    if (-not $commit) { $commit = Get-JsonValue $lock @('source_build', 'commit') }
    if (-not $commit) { Stop-Build "$lockFile has no llama_cpp.source_build.commit" }
    $buildNumber = [string](Get-JsonValue $lock @('llama_cpp', 'build_number'))
    $repoUrl = [string](Get-JsonValue $lock @('llama_cpp', 'repo'))
    Write-Log "pinned commit $commit from $lockFile"
} else {
    $commit = $FallbackCommit
    $buildNumber = $FallbackBuildNumber
    $repoUrl = $FallbackRepo
    Write-Log "$lockFile not found; falling back to commit $commit (b$buildNumber)"
}
if (-not $repoUrl) { $repoUrl = $FallbackRepo }
if ($commit -notmatch '^[0-9a-f]{40}$') { Stop-Build "pinned commit '$commit' is not a 40-hex git sha" }
if ($repoUrl -notmatch '^https://github\.com/[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$') { Stop-Build "repo '$repoUrl' is not a github https URL" }
if ($buildNumber -and $buildNumber -notmatch '^\d+$') { Stop-Build "build_number '$buildNumber' is not an integer" }

# ------------------------------------------------------------------ prerequisites

$problems = New-Object System.Collections.Generic.List[string]
$vsMajor = [int]([regex]::Match($Generator, '^Visual Studio (\d+)').Groups[1].Value)

if (-not (Find-Exe 'git')) { $problems.Add('git missing: winget install --id Git.Git -e   (then open a new terminal)') }

$vsPath = $null
$vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio\Installer\vswhere.exe'
if (Test-Path -LiteralPath $vswhere) {
    $vsPath = & $vswhere -version "[$vsMajor.0,$($vsMajor + 1).0)" -products '*' `
        -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath -latest |
        Select-Object -First 1
}
if (-not $vsPath) {
    if ($vsMajor -eq 17) {
        $problems.Add('Visual Studio 2022 C++ build tools missing: winget install --id Microsoft.VisualStudio.2022.BuildTools -e --override "--quiet --wait --add Microsoft.VisualStudio.Workload.VCTools --includeRecommended"')
    } else {
        $problems.Add("Visual Studio $vsMajor with the 'Desktop development with C++' workload is missing (generator '$Generator')")
    }
}

$cmakeExe = Find-Exe 'cmake'
if (-not $cmakeExe) {
    $uvTool = Find-UvToolExe 'cmake'
    if ($uvTool) { $cmakeExe = $uvTool.Path; Write-Log "cmake is not on PATH; using the one in $($uvTool.Source): $cmakeExe" }
}
if (-not $cmakeExe -and $vsPath) {
    $bundled = Join-Path $vsPath 'Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe'
    if (Test-Path -LiteralPath $bundled) { $cmakeExe = $bundled; Write-Log "using the CMake bundled with Visual Studio: $cmakeExe" }
}
if (-not $cmakeExe) {
    $problems.Add('cmake missing: winget install --id Kitware.CMake -e   (then open a new terminal), or without admin rights: uv tool install cmake')
} else {
    $cmakeLine = (& $cmakeExe --version | Select-Object -First 1)
    if ($cmakeLine -match '(\d+)\.(\d+)') {
        if ([version]"$($Matches[1]).$($Matches[2])" -lt $MinCmake) { $problems.Add("cmake is older than ${MinCmake}: winget upgrade --id Kitware.CMake -e") }
    } else {
        $problems.Add("cannot read the cmake version from '$cmakeLine'")
    }
}

$nvcc = Find-Exe 'nvcc'
if (-not $nvcc -and $env:CUDA_PATH -and (Test-Path -LiteralPath (Join-Path $env:CUDA_PATH 'bin\nvcc.exe'))) {
    $nvcc = Join-Path $env:CUDA_PATH 'bin\nvcc.exe'
}
$hasNvidiaGpu = $false
$smi = Find-Exe 'nvidia-smi'
if ($smi) {
    & $smi -L | Out-Null
    $hasNvidiaGpu = ($LASTEXITCODE -eq 0)
}

$accel = 'cpu'
$cudaRoot = $null
$cudaToolset = $null
if ($Cpu) {
    $accel = 'cpu'
} elseif ($nvcc) {
    $accel = 'cuda'
    $cudaRoot = Split-Path -Parent (Split-Path -Parent $nvcc)
} elseif ($hasNvidiaGpu) {
    $problems.Add('NVIDIA GPU found but no CUDA toolkit (nvcc): winget install --id Nvidia.CUDA -e   (then open a new terminal). Pass -Cpu to build CPU-only instead.')
} elseif ($CudaArchitectures) {
    $problems.Add('-CudaArchitectures given but nvcc was not found (looked on PATH and in $env:CUDA_PATH\bin)')
}

if ($accel -eq 'cuda') {
    $cudaLine = (& $nvcc --version | Select-String -Pattern 'release \d+\.\d+' | Select-Object -First 1 -ExpandProperty Line)
    Write-Log "CUDA toolkit at $cudaRoot ($cudaLine)"
    # -T cuda=<root> points CMake at this toolkit's own MSBuild integration, so a CUDA install that
    # predates Visual Studio still works. CMake requires <root>\extras\visual_studio_integration\
    # MSBuildExtensions for that form (CMAKE_VS_PLATFORM_TOOLSET_CUDA_CUSTOM_DIR), so without that folder
    # -T is left out and the globally installed integration is used instead.
    $toolkitIntegration = Join-Path $cudaRoot 'extras\visual_studio_integration\MSBuildExtensions'
    if (Test-Path -LiteralPath $toolkitIntegration) {
        $cudaToolset = "cuda=$cudaRoot"
    } elseif ($vsPath) {
        $globalDir = Join-Path $vsPath "MSBuild\Microsoft\VC\v${vsMajor}0\BuildCustomizations"
        if (-not (Get-ChildItem -LiteralPath $globalDir -Filter 'CUDA *.props' -ErrorAction SilentlyContinue)) {
            $problems.Add("CUDA's Visual Studio integration is missing: rerun the CUDA installer and select 'Visual Studio Integration'")
        }
    }
    if ($CudaArchitectures) {
        $listed = @(& $nvcc --list-gpu-arch | ForEach-Object { "$_".Trim() })
        if ($LASTEXITCODE -ne 0) {
            Write-Log "WARNING: nvcc --list-gpu-arch failed; cannot verify -CudaArchitectures $CudaArchitectures"
        } else {
            foreach ($arch in $CudaArchitectures.Split(';')) {
                $base = [regex]::Match($arch, '^\d+').Value
                if ($listed -notcontains "compute_$base") {
                    $problems.Add("nvcc does not list compute_$base (-CudaArchitectures $arch). sm_120 needs CUDA >= 12.8, sm_121 needs CUDA >= 12.9.")
                }
            }
        }
    }
}

if ($problems.Count -gt 0) {
    Write-Host '[hypercompute] cannot build yet. Fix these first:'
    foreach ($p in $problems) { Write-Host "  - $p" }
    exit 2
}
Write-Log "prerequisites ok (accelerator: $accel, generator: $Generator, jobs: $Jobs)"
if ($CheckOnly) { exit 0 }

# ------------------------------------------------------------------ checkout

if ((Test-Path -LiteralPath $SourceDir) -and -not (Test-Path -LiteralPath (Join-Path $SourceDir '.git'))) {
    if (Get-ChildItem -LiteralPath $SourceDir -Force | Select-Object -First 1) {
        Stop-Build "$SourceDir exists and is not a git checkout; pass another -SourceDir"
    }
}
$parent = Split-Path -Parent $SourceDir
if (-not (Test-Path -LiteralPath $parent)) { New-Item -ItemType Directory -Path $parent | Out-Null }
$SourceDir = Join-Path (Resolve-Path -LiteralPath $parent).Path (Split-Path -Leaf $SourceDir)

# llama.cpp has tracked paths of ~170 characters (tools/ui), so any -SourceDir longer than ~89 characters
# crosses MAX_PATH. Without core.longpaths git prints "Filename too long" for those files and still exits 0.
$gitLong = @('-c', 'core.longpaths=true')
if (-not (Test-Path -LiteralPath (Join-Path $SourceDir '.git'))) {
    Write-Log "cloning $repoUrl into $SourceDir (blobless clone, full commit history)"
    & git @gitLong clone --filter=blob:none --no-checkout $repoUrl $SourceDir
    if ($LASTEXITCODE -ne 0) { Stop-Build 'git clone failed' }
} else {
    $dirty = & git @gitLong -C $SourceDir status --porcelain --untracked-files=no
    if ($LASTEXITCODE -ne 0) { Stop-Build "git status failed in $SourceDir" }
    if ($dirty) { Stop-Build "$SourceDir has local modifications to tracked files; commit or stash them first" }
}
& git @gitLong -C $SourceDir cat-file -e "$($commit)^{commit}" 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Log "fetching $commit"
    & git @gitLong -C $SourceDir fetch origin $commit
    if ($LASTEXITCODE -ne 0) { Stop-Build "git fetch of $commit failed" }
}
& git @gitLong -C $SourceDir checkout --detach $commit
if ($LASTEXITCODE -ne 0) { Stop-Build "git checkout $commit failed" }
$head = (& git @gitLong -C $SourceDir rev-parse HEAD | Select-Object -First 1)
if ($head -ne $commit) { Stop-Build "checkout landed on $head, expected $commit" }
# git checkout exits 0 even when it could not write some files; never build a partial tree.
$dirty = @(& git @gitLong -C $SourceDir status --porcelain --untracked-files=no)
if ($LASTEXITCODE -ne 0) { Stop-Build "git status failed in $SourceDir" }
if ($dirty.Count -gt 0) {
    Stop-Build "git checkout of $commit left tracked files missing or modified ($($dirty.Count), first: '$($dirty[0])'). Fix the cause git printed above (disk space, a file locked by another program, a file name clash), then delete $SourceDir and rerun."
}

# ------------------------------------------------------------------ configure + build

# Separate build trees per accelerator: the VS generator refuses to switch toolsets (-T) in place.
$buildDir = Join-Path $SourceDir "build-hypercompute-$accel"
$cmakeArgs = @('-S', $SourceDir, '-B', $buildDir, '-G', $Generator, '-A', 'x64', '-DGGML_RPC=ON', '-DLLAMA_BUILD_TESTS=OFF')
if ($accel -eq 'cuda') {
    if ($cudaToolset) { $cmakeArgs += @('-T', $cudaToolset) }
    $cmakeArgs += '-DGGML_CUDA=ON'
    if ($CudaArchitectures) { $cmakeArgs += "-DCMAKE_CUDA_ARCHITECTURES=$CudaArchitectures" }
} else {
    $cmakeArgs += '-DGGML_CUDA=OFF'
}
# Same version banner as the official release ("build 11770, commit 39da130fc"), so engine identity
# checks match prebuilt peers.
$shortCommit = $commit.Substring(0, 9)
if ($buildNumber) { $cmakeArgs += @("-DLLAMA_BUILD_NUMBER=$buildNumber", "-DLLAMA_BUILD_COMMIT=$shortCommit") }

Write-Log ("cmake " + ($cmakeArgs -join ' '))
& $cmakeExe @cmakeArgs
if ($LASTEXITCODE -ne 0) {
    Stop-Build 'cmake configure failed (see above). With CUDA, "No CUDA toolset found" means the CUDA Visual Studio integration is missing; rerun the CUDA installer with it selected, or pass -Cpu.'
}
& $cmakeExe --build $buildDir --config Release -j $Jobs --target @Targets
if ($LASTEXITCODE -ne 0) { Stop-Build 'build failed (see the compiler output above)' }

# ------------------------------------------------------------------ verify

$binDir = Join-Path $buildDir 'bin\Release'
if (-not (Test-Path -LiteralPath (Join-Path $binDir 'llama-server.exe'))) { $binDir = Join-Path $buildDir 'bin' }
foreach ($t in $Targets) {
    if (-not (Test-Path -LiteralPath (Join-Path $binDir "$t.exe"))) { Stop-Build "expected $binDir\$t.exe after the build" }
}
$server = Join-Path $binDir 'llama-server.exe'
Write-Log 'llama-server --version:'
$versionOut = (& $server --version 2>&1 | ForEach-Object { "$_" }) -join "`n"
$versionExit = $LASTEXITCODE
Write-Host $versionOut
if ($versionExit -ne 0) {
    # 0xC0000135 = STATUS_DLL_NOT_FOUND: the CUDA runtime DLLs are not on PATH.
    Stop-Build "llama-server --version exited $versionExit. If a DLL is missing, add `"$cudaRoot\bin`" (and its x64 subfolder when present) to PATH or copy cudart64_*.dll, cublas64_*.dll and cublasLt64_*.dll next to llama-server.exe."
}
if ($buildNumber -and $versionOut -notmatch [regex]::Escape("build $buildNumber, commit $shortCommit")) {
    Write-Log "WARNING: version banner does not say 'build $buildNumber, commit $shortCommit'"
}
Write-Log 'llama-server --list-devices:'
& $server --list-devices
if ($LASTEXITCODE -ne 0) { Write-Log "WARNING: --list-devices exited $LASTEXITCODE (driver or runtime library problem)" }

$quoted = $binDir -replace "'", "''"
Write-Host ''
Write-Host "Built $($Targets -join ', ') ($accel) at commit $commit."
Write-Host "Binaries: $binDir"
Write-Host 'Use them with:'
Write-Host "  `$env:HYPERCOMPUTE_LLAMA_BIN_DIR = '$quoted'"
