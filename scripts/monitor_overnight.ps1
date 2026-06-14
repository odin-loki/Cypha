# Lightweight monitor: poll bench/BASELINE_LOCK.json run_at and print overnight status.
# Usage:
#   pwsh -File scripts/monitor_overnight.ps1
#   pwsh -File scripts/monitor_overnight.ps1 -IntervalSeconds 60
#   pwsh -File scripts/monitor_overnight.ps1 -Once
#   pwsh -File scripts/monitor_overnight.ps1 -LogFile bench/results/production_overnight_20260614_120000.log
param(
    [string]$LockFile = "",
    [string]$LogFile = "",
    [int]$IntervalSeconds = 30,
    [switch]$Once
)

$ErrorActionPreference = "Stop"
$root = Split-Path $PSScriptRoot -Parent
if (-not $LockFile) {
    $LockFile = Join-Path $root "bench\BASELINE_LOCK.json"
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

function Show-LockStatus {
    param([string]$Path)

    if (-not (Test-Path $Path)) {
        Write-Host "[$(Get-Date -Format 'HH:mm:ss')] lock file missing: $Path" -ForegroundColor Red
        return
    }

    $fileInfo = Get-Item $Path
    try {
        $lock = Get-Content $Path -Raw | ConvertFrom-Json
    } catch {
        Write-Host "[$(Get-Date -Format 'HH:mm:ss')] invalid JSON: $($_.Exception.Message)" -ForegroundColor Red
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

function Show-LogTail {
    param([string]$Path)

    if (-not $Path -or -not (Test-Path $Path)) {
        Write-Host "  log: (none)" -ForegroundColor DarkGray
        return
    }

    Write-Host "  log: $Path" -ForegroundColor DarkGray
    try {
        $lines = Get-Content $Path -Tail 3 -ErrorAction Stop
        foreach ($line in $lines) {
            Write-Host "    $line" -ForegroundColor DarkGray
        }
    } catch {
        Write-Host "    (could not read log: $($_.Exception.Message))" -ForegroundColor DarkYellow
    }
}

Write-Host "monitor_overnight: polling $LockFile every ${IntervalSeconds}s (Ctrl+C to stop)" -ForegroundColor Cyan
if ($LogFile) {
    Write-Host "  tailing log: $(Resolve-LogFile -Path $LogFile)" -ForegroundColor DarkGray
} else {
    Write-Host "  tailing log: latest bench/results/production_overnight_*.log" -ForegroundColor DarkGray
}
if ($Once) {
    Show-LockStatus -Path $LockFile
    Show-LogTail -Path (Resolve-LogFile -Path $LogFile)
    exit 0
}

while ($true) {
    Show-LockStatus -Path $LockFile
    Show-LogTail -Path (Resolve-LogFile -Path $LogFile)
    Start-Sleep -Seconds $IntervalSeconds
}
