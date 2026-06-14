# Watch production overnight run: process + log growth + BASELINE_LOCK sections.
# Usage:
#   pwsh -File scripts/watch_production_overnight.ps1
#   pwsh -File scripts/watch_production_overnight.ps1 -Once
#   pwsh -File scripts/watch_production_overnight.ps1 -IntervalSeconds 120 -ProcessId 12345
param(
    [string]$LockFile = "",
    [string]$LogFile = "",
    [int]$IntervalSeconds = 60,
    [int]$StallMinutes = 30,
    [int]$ProcessId = 0,
    [switch]$Once
)

$ErrorActionPreference = "Stop"
$root = Split-Path $PSScriptRoot -Parent
if (-not $LockFile) {
    $LockFile = Join-Path $root "bench\BASELINE_LOCK.json"
} elseif (-not [System.IO.Path]::IsPathRooted($LockFile)) {
    $LockFile = Join-Path $root $LockFile
}

function Resolve-LogFile {
    param([string]$Path)

    if ($Path) {
        if ([System.IO.Path]::IsPathRooted($Path)) { return $Path }
        return Join-Path $root $Path
    }

    $resultsDir = Join-Path $root "bench\results"
    if (-not (Test-Path $resultsDir)) { return $null }
    $latest = Get-ChildItem -Path $resultsDir -Filter "production_overnight_*.log" -ErrorAction SilentlyContinue |
        Sort-Object LastWriteTime -Descending |
        Select-Object -First 1
    if ($latest) { return $latest.FullName }
    return $null
}

$sectionNames = @("overnight_results", "rpsm_results", "cell_sweep_results")
$lastRunAts = @{}
$lastPollSize = -1
$lastGrowthUtc = $null
$resolvedLogPath = Resolve-LogFile -Path $LogFile
$cellSweepProgressPath = Join-Path $root "bench\results\cell_sweep\overnight_progress.log"

function Format-Age([datetime]$UtcWhen) {
    $age = (Get-Date).ToUniversalTime() - $UtcWhen
    if ($age.TotalDays -ge 1) {
        return ("{0:F1}d ago" -f $age.TotalDays)
    }
    if ($age.TotalHours -ge 1) {
        return ("{0:F1}h ago" -f $age.TotalHours)
    }
    if ($age.TotalMinutes -ge 1) {
        return ("{0:F0}m ago" -f $age.TotalMinutes)
    }
    return ("{0:F0}s ago" -f $age.TotalSeconds)
}

function Show-MigrationNote {
    $legacySummary = Join-Path $root "results\summary.csv"
    if (Test-Path $legacySummary) {
        Write-Host ""
        Write-Host "NOTE: legacy results/summary.csv detected at repo root." -ForegroundColor Yellow
        Write-Host "      New cell-sweep artifacts live under bench/results/cell_sweep - migrate or archive the old path." -ForegroundColor Yellow
    }
}

function Get-OvernightProcessInfo {
    if ($ProcessId -gt 0) {
        try {
            $proc = Get-Process -Id $ProcessId -ErrorAction Stop
            return @(
                [PSCustomObject]@{
                    Id          = $proc.Id
                    ProcessName = $proc.ProcessName
                    CommandLine = "(explicit -ProcessId)"
                }
            )
        } catch {
            Write-Host "  process: PID $ProcessId not found" -ForegroundColor Red
            return @()
        }
    }

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

function Show-ProcessStatus {
    $procs = Get-OvernightProcessInfo
    Write-Host ""
    Write-Host "[$(Get-Date -Format 'HH:mm:ss')] overnight processes" -ForegroundColor Cyan
    if (-not $procs -or $procs.Count -eq 0) {
        Write-Host "  (none - run may be finished or not started)" -ForegroundColor DarkGray
        return
    }
    foreach ($p in $procs) {
        $cmd = $p.CommandLine
        if ($cmd.Length -gt 120) {
            $cmd = $cmd.Substring(0, 117) + "..."
        }
        Write-Host ("  PID={0,-8} {1,-24} {2}" -f $p.Id, $p.ProcessName, $cmd) -ForegroundColor DarkGray
    }
}

function Show-LogStatus {
    param([string]$Path)

    Write-Host ""
    Write-Host "[$(Get-Date -Format 'HH:mm:ss')] production overnight log" -ForegroundColor Cyan

    if (-not $Path -or -not (Test-Path $Path)) {
        Write-Host "  log: (none - start with scripts/run_production_overnight.ps1)" -ForegroundColor DarkGray
        return
    }

    $info = Get-Item $Path
    $size = $info.Length
    $delta = if ($lastPollSize -ge 0) { $size - $lastPollSize } else { 0 }
    $deltaLabel = if ($lastPollSize -ge 0) { "+$delta bytes since last poll" } else { "initial snapshot" }

    if ($null -eq $lastGrowthUtc -or $size -gt $lastPollSize) {
        $lastGrowthUtc = Get-Date
    }
    $lastPollSize = $size

    Write-Host ("  path: {0}" -f $Path) -ForegroundColor DarkGray
    Write-Host ("  size: {0} bytes ({1})" -f $size, $deltaLabel) -ForegroundColor DarkGray

    if ($lastGrowthUtc -and ((Get-Date) - $lastGrowthUtc).TotalMinutes -ge $StallMinutes) {
        $stallMin = [math]::Floor(((Get-Date) - $lastGrowthUtc).TotalMinutes)
        Write-Host ("  WARNING: log has not grown in {0}+ minutes (stalled?)" -f $stallMin) -ForegroundColor Yellow
    }

    try {
        $lastLine = Get-Content $Path -Tail 1 -ErrorAction Stop
        if ($lastLine) {
            Write-Host ("  last: {0}" -f $lastLine) -ForegroundColor DarkGray
        } else {
            Write-Host "  last: (empty)" -ForegroundColor DarkGray
        }
    } catch {
        Write-Host "  last: (could not read: $($_.Exception.Message))" -ForegroundColor DarkYellow
    }
}

function Show-CellSweepProgress {
    Write-Host ""
    Write-Host "[$(Get-Date -Format 'HH:mm:ss')] cell sweep progress" -ForegroundColor Cyan

    if (-not (Test-Path $cellSweepProgressPath)) {
        Write-Host "  (no overnight_progress.log yet)" -ForegroundColor DarkGray
        return
    }

    try {
        $lines = @(Get-Content $cellSweepProgressPath -Tail 2 -ErrorAction Stop)
        if ($lines.Count -eq 0) {
            Write-Host "  (empty)" -ForegroundColor DarkGray
            return
        }
        foreach ($line in $lines) {
            Write-Host ("  {0}" -f $line) -ForegroundColor DarkGray
        }
    } catch {
        Write-Host "  (could not read: $($_.Exception.Message))" -ForegroundColor DarkYellow
    }
}

function Show-LockStatus {
    param([string]$Path)

    if (-not (Test-Path $Path)) {
        Write-Host ""
        Write-Host "[$(Get-Date -Format 'HH:mm:ss')] BASELINE_LOCK.json missing: $Path" -ForegroundColor Red
        return
    }

    $fileInfo = Get-Item $Path
    try {
        $lock = Get-Content $Path -Raw | ConvertFrom-Json
    } catch {
        Write-Host ""
        Write-Host "[$(Get-Date -Format 'HH:mm:ss')] invalid lock JSON: $($_.Exception.Message)" -ForegroundColor Red
        return
    }

    Write-Host ""
    Write-Host "[$(Get-Date -Format 'HH:mm:ss')] BASELINE_LOCK.json (mtime $($fileInfo.LastWriteTime.ToString('yyyy-MM-dd HH:mm:ss')))" -ForegroundColor Cyan

    foreach ($name in $sectionNames) {
        if ($lock.PSObject.Properties.Name -notcontains $name -or $null -eq $lock.$name) {
            Write-Host "  $name : (absent)" -ForegroundColor DarkGray
            continue
        }

        $section = $lock.$name
        $status = if ($section.PSObject.Properties.Name -contains "status") { [string]$section.status } else { "?" }
        $runAtRaw = if ($section.PSObject.Properties.Name -contains "run_at") { [string]$section.run_at } else { "" }
        $nTrain = if ($section.PSObject.Properties.Name -contains "n_train") { $section.n_train } else { "?" }
        $nEval = if ($section.PSObject.Properties.Name -contains "n_eval") { $section.n_eval } else { "?" }
        $bpc = if ($section.PSObject.Properties.Name -contains "bpc") { $section.bpc } else { "?" }

        $ageLabel = ""
        if (-not [string]::IsNullOrWhiteSpace($runAtRaw)) {
            try {
                $runAt = [datetimeoffset]::Parse($runAtRaw, [Globalization.CultureInfo]::InvariantCulture)
                $ageLabel = Format-Age $runAt.UtcDateTime
            } catch {
                $ageLabel = "bad run_at"
            }
        } else {
            $ageLabel = "no run_at"
        }

        $changed = ""
        if ($lastRunAts.ContainsKey($name) -and $lastRunAts[$name] -ne $runAtRaw) {
            $changed = " [UPDATED]"
        }
        $lastRunAts[$name] = $runAtRaw

        $tier = if ([int]$nTrain -ge 300000) { "production" } elseif ([int]$nTrain -ge 5000) { "medium" } else { "smoke" }
        Write-Host ("  {0,-22} status={1,-12} n_train={2,-8} bpc={3,-8} run_at={4} ({5}){6}" -f `
            "$name", $status, $nTrain, $bpc, $runAtRaw, $ageLabel, $changed)
        Write-Host ("    tier={0} n_eval={1}" -f $tier, $nEval) -ForegroundColor DarkGray
    }
}

function Show-Snapshot {
    Show-ProcessStatus
    Show-LogStatus -Path $resolvedLogPath
    Show-CellSweepProgress
    Show-LockStatus -Path $LockFile
}

Write-Host "watch_production_overnight: polling every ${IntervalSeconds}s (stall warn after ${StallMinutes}m without log growth)" -ForegroundColor Cyan
Write-Host "  lock: $LockFile" -ForegroundColor DarkGray
if ($resolvedLogPath) {
    Write-Host "  log:  $resolvedLogPath" -ForegroundColor DarkGray
} else {
    Write-Host "  log:  (no bench/results/production_overnight_*.log yet)" -ForegroundColor DarkGray
}
if ($ProcessId -gt 0) {
    Write-Host "  tracking PID: $ProcessId" -ForegroundColor DarkGray
}

Show-MigrationNote

$hadOvernightProcesses = $false

if ($Once) {
    Show-Snapshot
    exit 0
}

while ($true) {
    $procsNow = Get-OvernightProcessInfo
    $runningNow = ($procsNow -and $procsNow.Count -gt 0)
    if ($hadOvernightProcesses -and -not $runningNow) {
        Write-Host ""
        Write-Host "HINT: overnight processes finished - run poll_and_finalize_overnight.ps1 to finalize + commit preview" -ForegroundColor Green
        Write-Host "      pwsh -File scripts/poll_and_finalize_overnight.ps1" -ForegroundColor DarkGray
    }
    $hadOvernightProcesses = $runningNow

    Show-Snapshot
    Start-Sleep -Seconds $IntervalSeconds
}
