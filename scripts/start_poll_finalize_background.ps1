# Phase 18/19/21/24: start poll_and_finalize_overnight.ps1 in the background after manually
# launching production overnight (e.g. run_production_overnight.ps1).
# When -BuildDir is the default native/build and overnight is running, BuildDir is
# auto-detected from the run_production_overnight.ps1 command line (e.g. native/build_p13).
#
# Usage:
#   pwsh -File scripts/start_poll_finalize_background.ps1
#   pwsh -File scripts/start_poll_finalize_background.ps1 -BuildDir native/build
#   pwsh -File scripts/start_poll_finalize_background.ps1 -LogFile bench/results/poll_finalize.log
#   pwsh -File scripts/start_poll_finalize_background.ps1 -AutoCommit
param(
    [string]$BuildDir = "",
    [string]$LogFile = "bench/results/poll_finalize.log",
    [switch]$AutoCommit
)

$ErrorActionPreference = "Stop"
. (Join-Path $PSScriptRoot "lib\NativeBenchCommon.ps1")
$root = Get-CyphaRepoRoot -ScriptRoot $PSScriptRoot
$DEFAULT_BUILD_DIR = ""
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

    if ($Requested -and $Requested -ne $DEFAULT_BUILD_DIR) {
        return $Requested
    }

    if (-not (Test-OvernightStillRunning)) {
        return (Get-DefaultNativeBuildDir -Override $Requested)
    }

    $detected = Get-DetectedBuildDirFromOvernight
    if ($detected) {
        Write-Host "start_poll_finalize_background: auto-detected BuildDir from run_production_overnight.ps1: $detected" -ForegroundColor Yellow
        return $detected
    }

    return (Get-DefaultNativeBuildDir -Override $Requested)
}

$BuildDir = Resolve-PollBuildDir -Requested (Get-DefaultNativeBuildDir -Override $BuildDir)

function Stop-ExistingPollFinalizeProcesses {
    $killed = @()
    $cim = Get-CimInstance Win32_Process -ErrorAction SilentlyContinue |
        Where-Object {
            $_.CommandLine -and $_.CommandLine -match 'poll_and_finalize_overnight\.ps1'
        }
    foreach ($p in $cim) {
        try {
            Stop-Process -Id $p.ProcessId -Force -ErrorAction Stop
            $killed += $p.ProcessId
        } catch {
            Write-Host ("start_poll_finalize_background: warn: could not stop PID {0}: {1}" -f $p.ProcessId, $_.Exception.Message) -ForegroundColor Yellow
        }
    }
    return $killed
}

$killedPids = @(Stop-ExistingPollFinalizeProcesses)
if ($killedPids.Count -gt 0) {
    Write-Host ("start_poll_finalize_background: killed existing poll_and_finalize_overnight.ps1 PID(s): {0}" -f ($killedPids -join ", ")) -ForegroundColor Yellow
}

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
    if ($AutoCommit) {
        $argList += "-AutoCommit"
    }
} else {
    $exe = "powershell.exe"
    $argList = @(
        "-NoProfile",
        "-ExecutionPolicy", "Bypass",
        "-File", $pollScript,
        "-BuildDir", $BuildDir,
        "-LogFile", $resolvedLog
    )
    if ($AutoCommit) {
        $argList += "-AutoCommit"
    }
}

$proc = Start-Process -FilePath $exe -ArgumentList $argList -WorkingDirectory $root -PassThru -WindowStyle Hidden
if (-not $proc) {
    throw "failed to start poll_and_finalize_overnight.ps1"
}

Write-Host "Started poll_and_finalize_overnight.ps1 in background (PID $($proc.Id))" -ForegroundColor Green
Write-Host "  build: $BuildDir" -ForegroundColor DarkGray
Write-Host "  log:   $resolvedLog" -ForegroundColor DarkGray
if ($AutoCommit) {
    Write-Host "  autocommit: enabled (-Force when lock n_train >= 300000 after finalize)" -ForegroundColor DarkGray
}
Write-Host "Tail the log or run watch_production_overnight.ps1 while overnight processes run." -ForegroundColor Yellow
