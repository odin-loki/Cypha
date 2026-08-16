# Watch production overnight run: process + log growth + BASELINE_LOCK sections.
# Phase 24: variant-count stall detector (STALL_WARNING) while overnight processes run.
# Usage:
#   pwsh -File scripts/watch_production_overnight.ps1
#   pwsh -File scripts/watch_production_overnight.ps1 -Once
#   pwsh -File scripts/watch_production_overnight.ps1 -IntervalSeconds 120 -ProcessId 12345
#   pwsh -File scripts/watch_production_overnight.ps1 -LogFile bench/results/watch_stall.log
param(
    [string]$LockFile = "",
    [string]$ProductionLogFile = "",
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
$lastVariantCount = -1
$lastVariantGrowthUtc = $null
$resolvedLogPath = Resolve-LogFile -Path $ProductionLogFile
$resolvedStallLogFile = $null
$cellSweepProgressPath = Join-Path $root "bench\results\cell_sweep\overnight_progress.log"
$CELL_SWEEP_EXPECTED_VARIANTS = 36
$PRODUCTION_N_TRAIN_MIN = 300000

if ($LogFile) {
    if ([System.IO.Path]::IsPathRooted($LogFile)) {
        $resolvedStallLogFile = $LogFile
    } else {
        $resolvedStallLogFile = Join-Path $root $LogFile
    }
    $stallLogDir = Split-Path $resolvedStallLogFile -Parent
    if ($stallLogDir) {
        New-Item -ItemType Directory -Force -Path $stallLogDir | Out-Null
    }
}

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

function Get-CellSweepResultsDir {
    $primary = Join-Path $root "bench\results\cell_sweep"
    $legacy = Join-Path $root "results"

    $primaryCount = 0
    if (Test-Path $primary) {
        $primaryCount = @(Get-ChildItem -Path $primary -Filter "variant_*.json" -File -ErrorAction SilentlyContinue).Count
    }

    $legacyCount = 0
    if (Test-Path $legacy) {
        $legacyCount = @(Get-ChildItem -Path $legacy -Filter "variant_*.json" -File -ErrorAction SilentlyContinue).Count
    }

    if ($legacyCount -gt $primaryCount) {
        return @{
            Dir    = $legacy
            Source = "results (in-flight spill, $legacyCount variants)"
        }
    }

    if ($primaryCount -gt 0 -or (Test-Path $primary)) {
        $sourceLabel = "bench/results/cell_sweep"
        if ($legacyCount -gt 0 -and $legacyCount -eq $primaryCount) {
            $sourceLabel = "bench/results/cell_sweep (tied with results/)"
        }
        return @{
            Dir    = $primary
            Source = $sourceLabel
        }
    }

    if ($legacyCount -gt 0) {
        return @{
            Dir    = $legacy
            Source = "results (in-flight spill, $legacyCount variants)"
        }
    }

    return $null
}

function Get-CellSweepVariantCount {
    $sweepInfo = Get-CellSweepResultsDir
    if (-not $sweepInfo) {
        return 0
    }

    return @(Get-ChildItem -Path $sweepInfo.Dir -Filter "variant_*.json" -File -ErrorAction SilentlyContinue).Count
}

function Write-StallWarning {
    param([string]$Message)

    $line = "STALL_WARNING $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') $Message"
    Write-Host $line -ForegroundColor Yellow
    if ($script:resolvedStallLogFile) {
        $line | Out-File -FilePath $script:resolvedStallLogFile -Append -Encoding utf8
    }
}

function Test-VariantCountStall {
    param(
        [bool]$OvernightRunning,
        [int]$VariantCount
    )

    if (-not $OvernightRunning) {
        $script:lastVariantCount = -1
        $script:lastVariantGrowthUtc = $null
        return
    }

    if ($script:lastVariantCount -lt 0 -or $VariantCount -ne $script:lastVariantCount) {
        $script:lastVariantCount = $VariantCount
        $script:lastVariantGrowthUtc = Get-Date
        return
    }

    if ($null -eq $script:lastVariantGrowthUtc) {
        $script:lastVariantGrowthUtc = Get-Date
        return
    }

    $stallMin = [math]::Floor(((Get-Date) - $script:lastVariantGrowthUtc).TotalMinutes)
    if ($stallMin -ge $StallMinutes) {
        Write-StallWarning -Message "variant_count=$VariantCount unchanged for ${stallMin}m (threshold ${StallMinutes}m) while overnight running"
    }
}

function Get-VariantNTrain {
    param([System.IO.FileInfo]$VariantFile)

    try {
        $json = Get-Content $VariantFile.FullName -Raw | ConvertFrom-Json
    } catch {
        return $null
    }

    if ($json.PSObject.Properties.Name -contains "n_train") {
        return [int]$json.n_train
    }

    return $null
}

function Show-CellSweepProgress {
    param([bool]$OvernightRunning = $false)

    Write-Host ""
    Write-Host "[$(Get-Date -Format 'HH:mm:ss')] cell sweep progress" -ForegroundColor Cyan

    $sweepInfo = Get-CellSweepResultsDir
    if ($sweepInfo) {
        $variants = @(Get-ChildItem -Path $sweepInfo.Dir -Filter "variant_*.json" -File -ErrorAction SilentlyContinue)
        $variantCount = $variants.Count
        $manifestNTrain = $null
        $variantNTrain = $null
        $effectiveNTrain = $null
        $latestVariant = $null

        if ($variantCount -gt 0) {
            $latestVariant = $variants | Sort-Object LastWriteTime -Descending | Select-Object -First 1
            $variantNTrain = Get-VariantNTrain -VariantFile $latestVariant
        }

        $manifestPath = Join-Path $sweepInfo.Dir "manifest.json"
        if (Test-Path $manifestPath) {
            try {
                $manifest = Get-Content $manifestPath -Raw | ConvertFrom-Json
                if ($manifest.PSObject.Properties.Name -contains "n_train") {
                    $manifestNTrain = [int]$manifest.n_train
                }
            } catch {
                Write-Host "  manifest: (invalid JSON)" -ForegroundColor DarkYellow
            }
        }

        if ($OvernightRunning -and $null -ne $manifestNTrain -and $manifestNTrain -lt $PRODUCTION_N_TRAIN_MIN -and $null -ne $variantNTrain) {
            $effectiveNTrain = $variantNTrain
        } elseif ($null -ne $manifestNTrain) {
            $effectiveNTrain = $manifestNTrain
        } elseif ($null -ne $variantNTrain) {
            $effectiveNTrain = $variantNTrain
        }

        if ($OvernightRunning) {
            $progressLine = "  progress: {0}/{1}" -f $variantCount, $CELL_SWEEP_EXPECTED_VARIANTS
            if ($null -ne $effectiveNTrain) {
                $progressLine += " effective_n_train=$effectiveNTrain"
            }
            Write-Host $progressLine -ForegroundColor DarkGray
        } elseif ($variantCount -gt 0) {
            $variantLine = "  variants: {0} (expect {1} for full sweep)" -f $variantCount, $CELL_SWEEP_EXPECTED_VARIANTS
            if ($null -ne $effectiveNTrain) {
                $variantLine += " effective_n_train=$effectiveNTrain"
            }
            Write-Host $variantLine -ForegroundColor DarkGray
        } else {
            Write-Host "  variants: 0 (no variant_*.json yet)" -ForegroundColor DarkGray
        }

        if ($latestVariant) {
            Write-Host ("  latest: {0} mtime={1}" -f $latestVariant.Name, $latestVariant.LastWriteTime.ToString("yyyy-MM-dd HH:mm:ss")) -ForegroundColor DarkGray
        }

        if ($null -ne $manifestNTrain) {
            Write-Host ("  manifest: n_train={0}" -f $manifestNTrain) -ForegroundColor DarkGray
        } elseif (Test-Path $manifestPath) {
            Write-Host "  manifest: n_train=?" -ForegroundColor DarkGray
        }

        if ($sweepInfo.Source -match "in-flight|legacy|tied") {
            Write-Host ("  dir: {0}" -f $sweepInfo.Source) -ForegroundColor Yellow
        }
    } else {
        Write-Host "  (no cell sweep dir yet)" -ForegroundColor DarkGray
    }

    if (-not (Test-Path $cellSweepProgressPath)) {
        return
    }

    try {
        $lines = @(Get-Content $cellSweepProgressPath -Tail 2 -ErrorAction Stop)
        if ($lines.Count -eq 0) {
            return
        }
        foreach ($line in $lines) {
            Write-Host ("  log: {0}" -f $line) -ForegroundColor DarkGray
        }
    } catch {
        Write-Host ("  log: (could not read overnight_progress.log: $($_.Exception.Message))") -ForegroundColor DarkYellow
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
    param([bool]$OvernightRunning = $false)

    Show-ProcessStatus
    Show-LogStatus -Path $resolvedLogPath
    Show-CellSweepProgress -OvernightRunning:$OvernightRunning
    Test-VariantCountStall -OvernightRunning:$OvernightRunning -VariantCount (Get-CellSweepVariantCount)
    Show-LockStatus -Path $LockFile
}

Write-Host "watch_production_overnight: polling every ${IntervalSeconds}s (stall warn after ${StallMinutes}m without log growth or variant progress)" -ForegroundColor Cyan
Write-Host "  lock: $LockFile" -ForegroundColor DarkGray
if ($resolvedLogPath) {
    Write-Host "  log:  $resolvedLogPath" -ForegroundColor DarkGray
} else {
    Write-Host "  log:  (no bench/results/production_overnight_*.log yet)" -ForegroundColor DarkGray
}
if ($resolvedStallLogFile) {
    Write-Host "  stall log: $resolvedStallLogFile (append STALL_WARNING)" -ForegroundColor DarkGray
}
if ($ProcessId -gt 0) {
    Write-Host "  tracking PID: $ProcessId" -ForegroundColor DarkGray
}

Show-MigrationNote

$hadOvernightProcesses = $false

if ($Once) {
    $onceProcs = Get-OvernightProcessInfo
    $onceRunning = ($onceProcs -and $onceProcs.Count -gt 0)
    Show-Snapshot -OvernightRunning:$onceRunning
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

    Show-Snapshot -OvernightRunning:$runningNow
    Start-Sleep -Seconds $IntervalSeconds
}
