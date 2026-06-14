# Phase 18/19: start poll_and_finalize_overnight.ps1 in the background after manually
# launching production overnight (e.g. run_production_overnight.ps1).
# When -BuildDir is the default native/build and overnight is running, BuildDir is
# auto-detected from the run_production_overnight.ps1 command line (e.g. native/build_p13).
#
# Usage:
#   pwsh -File scripts/start_poll_finalize_background.ps1
#   pwsh -File scripts/start_poll_finalize_background.ps1 -BuildDir native/build
#   pwsh -File scripts/start_poll_finalize_background.ps1 -LogFile bench/results/poll_finalize.log
param(
    [string]$BuildDir = "native/build",
    [string]$LogFile = "bench/results/poll_finalize.log"
)

$ErrorActionPreference = "Stop"
$root = Split-Path $PSScriptRoot -Parent
$DEFAULT_BUILD_DIR = "native/build"
$pollScript = Join-Path $PSScriptRoot "poll_and_finalize_overnight.ps1"

if (-not (Test-Path $pollScript)) {
    throw "missing $pollScript"
}

function Test-OvernightStillRunning {
    foreach ($name in @("cyphalm_bench_native", "cypha_cell_hypothesis_sweep")) {
        if (Get-Process -Name "${name}*" -ErrorAction SilentlyContinue) {
            return $true
        }
    }

    $cim = Get-CimInstance Win32_Process -ErrorAction SilentlyContinue |
        Where-Object {
            $_.CommandLine -and $_.CommandLine -match 'run_production_overnight\.ps1'
        }
    if ($cim) {
        return $true
    }

    return $false
}

function Get-DetectedBuildDirFromOvernight {
    $cim = Get-CimInstance Win32_Process -ErrorAction SilentlyContinue |
        Where-Object {
            $_.CommandLine -and $_.CommandLine -match 'run_production_overnight\.ps1'
        }
    foreach ($p in $cim) {
        if ($p.CommandLine -match '-BuildDir\s+("([^"]+)"|[^\s]+)') {
            $raw = if ($Matches[2]) { $Matches[2] } else { $Matches[1] }
            if ($raw) { return $raw }
        }
    }
    return $null
}

function Resolve-PollBuildDir {
    param([string]$Requested)

    if ($Requested -ne $DEFAULT_BUILD_DIR) {
        return $Requested
    }

    if (-not (Test-OvernightStillRunning)) {
        return $Requested
    }

    $detected = Get-DetectedBuildDirFromOvernight
    if ($detected) {
        Write-Host "start_poll_finalize_background: auto-detected BuildDir from run_production_overnight.ps1: $detected" -ForegroundColor Yellow
        return $detected
    }

    return $Requested
}

$BuildDir = Resolve-PollBuildDir -Requested $BuildDir

if ([System.IO.Path]::IsPathRooted($LogFile)) {
    $resolvedLog = $LogFile
} else {
    $resolvedLog = Join-Path $root $LogFile
}

$logDir = Split-Path $resolvedLog -Parent
if ($logDir) {
    New-Item -ItemType Directory -Force -Path $logDir | Out-Null
}

$pwsh = Get-Command pwsh -ErrorAction SilentlyContinue
if ($pwsh) {
    $exe = $pwsh.Source
    $argList = @(
        "-NoProfile",
        "-File", $pollScript,
        "-BuildDir", $BuildDir,
        "-LogFile", $resolvedLog
    )
} else {
    $exe = "powershell.exe"
    $argList = @(
        "-NoProfile",
        "-ExecutionPolicy", "Bypass",
        "-File", $pollScript,
        "-BuildDir", $BuildDir,
        "-LogFile", $resolvedLog
    )
}

$proc = Start-Process -FilePath $exe -ArgumentList $argList -WorkingDirectory $root -PassThru -WindowStyle Hidden
if (-not $proc) {
    throw "failed to start poll_and_finalize_overnight.ps1"
}

Write-Host "Started poll_and_finalize_overnight.ps1 in background (PID $($proc.Id))" -ForegroundColor Green
Write-Host "  build: $BuildDir" -ForegroundColor DarkGray
Write-Host "  log:   $resolvedLog" -ForegroundColor DarkGray
Write-Host "Tail the log or run watch_production_overnight.ps1 while overnight processes run." -ForegroundColor Yellow
