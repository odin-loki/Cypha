# Phase 17: poll until production overnight processes exit, then finalize + commit preview.
# Reuses overnight process detection from watch_production_overnight.ps1 (+ cypha_cell_hypothesis_sweep).
#
# Usage:
#   pwsh -File scripts/poll_and_finalize_overnight.ps1
#   pwsh -File scripts/poll_and_finalize_overnight.ps1 -Once
#   pwsh -File scripts/poll_and_finalize_overnight.ps1 -Force
#   pwsh -File scripts/poll_and_finalize_overnight.ps1 -LogFile bench/results/poll_finalize.log
param(
    [string]$BuildDir = "native/build",
    [string]$LockFile = "",
    [string]$LogFile = "",
    [int]$IntervalSeconds = 60,
    [switch]$Force,
    [switch]$Once
)

$ErrorActionPreference = "Stop"
$root = Split-Path $PSScriptRoot -Parent
$transcriptTemp = $null
$resolvedLogFile = $null

if ($LogFile) {
    if ([System.IO.Path]::IsPathRooted($LogFile)) {
        $resolvedLogFile = $LogFile
    } else {
        $resolvedLogFile = Join-Path $root $LogFile
    }
    $logDir = Split-Path $resolvedLogFile -Parent
    if ($logDir) {
        New-Item -ItemType Directory -Force -Path $logDir | Out-Null
    }
    "=== poll_and_finalize_overnight started $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') ===" |
        Out-File -FilePath $resolvedLogFile -Append -Encoding utf8
    $transcriptTemp = Join-Path $env:TEMP "cypha_poll_finalize_$([Guid]::NewGuid().ToString('N')).log"
    Start-Transcript -Path $transcriptTemp | Out-Null
}

if (-not $LockFile) {
    $LockFile = Join-Path $root "bench\BASELINE_LOCK.json"
} elseif (-not [System.IO.Path]::IsPathRooted($LockFile)) {
    $LockFile = Join-Path $root $LockFile
}

function Get-OvernightProcessInfo {
    $found = @()

    foreach ($proc in Get-Process -Name "cyphalm_bench_native*" -ErrorAction SilentlyContinue) {
        $found += [PSCustomObject]@{
            Id          = $proc.Id
            ProcessName = $proc.ProcessName
            CommandLine = "cyphalm_bench_native"
        }
    }

    foreach ($proc in Get-Process -Name "cypha_cell_hypothesis_sweep*" -ErrorAction SilentlyContinue) {
        $found += [PSCustomObject]@{
            Id          = $proc.Id
            ProcessName = $proc.ProcessName
            CommandLine = "cypha_cell_hypothesis_sweep"
        }
    }

    $cim = Get-CimInstance Win32_Process -ErrorAction SilentlyContinue |
        Where-Object {
            $_.CommandLine -and $_.CommandLine -match 'run_production_overnight\.ps1'
        }
    foreach ($p in $cim) {
        $found += [PSCustomObject]@{
            Id          = $p.ProcessId
            ProcessName = $p.Name
            CommandLine = $p.CommandLine
        }
    }

    return $found
}

function Show-RunningProcesses {
    $procs = Get-OvernightProcessInfo
    Write-Host "[$(Get-Date -Format 'HH:mm:ss')] overnight processes still running:" -ForegroundColor Yellow
    foreach ($p in $procs) {
        $cmd = $p.CommandLine
        if ($cmd.Length -gt 120) {
            $cmd = $cmd.Substring(0, 117) + "..."
        }
        Write-Host ("  PID={0,-8} {1,-28} {2}" -f $p.Id, $p.ProcessName, $cmd) -ForegroundColor DarkGray
    }
}

function Show-LockSummary {
    param([string]$Path)

    if (-not (Test-Path $Path)) {
        Write-Host "lock file not found: $Path" -ForegroundColor Red
        return
    }

    try {
        $lock = Get-Content $Path -Raw | ConvertFrom-Json
    } catch {
        Write-Host "invalid lock JSON: $($_.Exception.Message)" -ForegroundColor Red
        return
    }

    Write-Host ""
    Write-Host "== BASELINE_LOCK summary ==" -ForegroundColor Cyan
    foreach ($name in @("overnight_results", "rpsm_results", "cell_sweep_results")) {
        if ($lock.PSObject.Properties.Name -notcontains $name -or $null -eq $lock.$name) {
            Write-Host ("  {0,-22} (absent)" -f $name) -ForegroundColor DarkGray
            continue
        }
        $section = $lock.$name
        $nTrain = if ($section.PSObject.Properties.Name -contains "n_train") { $section.n_train } else { "?" }
        $status = if ($section.PSObject.Properties.Name -contains "status") { $section.status } else { "?" }
        $bpc = if ($section.PSObject.Properties.Name -contains "bpc") { $section.bpc } else { "?" }
        $runAt = if ($section.PSObject.Properties.Name -contains "run_at") { $section.run_at } else { "" }
        Write-Host ("  {0,-22} n_train={1,-8} status={2,-12} bpc={3}" -f $name, $nTrain, $status, $bpc)
        if ($runAt) {
            Write-Host ("    run_at={0}" -f $runAt) -ForegroundColor DarkGray
        }
    }
}

function Test-OvernightStillRunning {
    $procs = Get-OvernightProcessInfo
    return ($procs -and $procs.Count -gt 0)
}

$finalizeScript = Join-Path $PSScriptRoot "finalize_production_overnight.ps1"
$commitScript = Join-Path $PSScriptRoot "commit_production_lock.ps1"
$exitCode = 0

try {
Write-Host "poll_and_finalize_overnight: polling every ${IntervalSeconds}s for overnight process exit" -ForegroundColor Cyan
Write-Host "  lock:  $LockFile" -ForegroundColor DarkGray
Write-Host "  build: $BuildDir" -ForegroundColor DarkGray
if ($resolvedLogFile) {
    Write-Host "  log:   $resolvedLogFile (append)" -ForegroundColor DarkGray
}
if ($Force) {
    Write-Host "  commit: -Force (git add + commit after finalize)" -ForegroundColor DarkGray
} else {
    Write-Host "  commit: DryRun preview (pass -Force to git add + commit)" -ForegroundColor DarkGray
}

if ($Once) {
    if (Test-OvernightStillRunning) {
        Show-RunningProcesses
        Write-Host "poll_and_finalize_overnight: processes still running (-Once)" -ForegroundColor Yellow
        $exitCode = 1
    } else {
        Write-Host "[$(Get-Date -Format 'HH:mm:ss')] no overnight processes (-Once)" -ForegroundColor Green
    }
} else {
    while (Test-OvernightStillRunning) {
        Show-RunningProcesses
        Start-Sleep -Seconds $IntervalSeconds
    }
    Write-Host "[$(Get-Date -Format 'HH:mm:ss')] no overnight processes - finalizing" -ForegroundColor Green
}

if ($exitCode -eq 0) {
Write-Host ""
Write-Host "== finalize_production_overnight.ps1 ==" -ForegroundColor Cyan
& $finalizeScript -BuildDir $BuildDir -LockFile $LockFile
if ($LASTEXITCODE -ne 0) {
    Show-LockSummary -Path $LockFile
    $exitCode = $LASTEXITCODE
} else {
Write-Host ""
if ($Force) {
    Write-Host "== commit_production_lock.ps1 -Force ==" -ForegroundColor Cyan
    & $commitScript -BuildDir $BuildDir -LockFile $LockFile -Force
} else {
    Write-Host "== commit_production_lock.ps1 -DryRun ==" -ForegroundColor Cyan
    & $commitScript -BuildDir $BuildDir -LockFile $LockFile -DryRun
}
$commitCode = $LASTEXITCODE
Show-LockSummary -Path $LockFile
if ($commitCode -ne 0) {
    $exitCode = $commitCode
} else {
Write-Host ""
Write-Host "poll_and_finalize_overnight: OK" -ForegroundColor Green
}
}
}
} finally {
    if ($transcriptTemp) {
        Stop-Transcript | Out-Null
        if (Test-Path $transcriptTemp) {
            Get-Content -Path $transcriptTemp | Add-Content -Path $resolvedLogFile -Encoding utf8
            Remove-Item -Path $transcriptTemp -Force -ErrorAction SilentlyContinue
        }
        "=== poll_and_finalize_overnight finished $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') exit=$exitCode ===" |
            Out-File -FilePath $resolvedLogFile -Append -Encoding utf8
    }
}
exit $exitCode
