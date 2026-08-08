# Phase 17/19/20/21/24: poll until production overnight processes exit, then finalize + commit preview.
# Reuses overnight process detection from watch_production_overnight.ps1 (+ cypha_cell_hypothesis_sweep).
# When -BuildDir is the default native/build and overnight is running, BuildDir is auto-detected
# from the run_production_overnight.ps1 command line (e.g. native/build_p13).
# With -LogFile, each poll cycle appends HEARTBEAT (timestamp, process count, lock n_train).
# Poll query failures log ERROR and retry (does not treat failed query as "processes exited").
# Usage:
#   pwsh -File scripts/poll_and_finalize_overnight.ps1
#   pwsh -File scripts/poll_and_finalize_overnight.ps1 -Once
#   pwsh -File scripts/poll_and_finalize_overnight.ps1 -Force
#   pwsh -File scripts/poll_and_finalize_overnight.ps1 -LogFile bench/results/poll_finalize.log
#   pwsh -File scripts/poll_and_finalize_overnight.ps1 -AutoCommit
param(
    [string]$BuildDir = "",
    [string]$LockFile = "",
    [string]$LogFile = "",
    [int]$IntervalSeconds = 60,
    [switch]$Force,
    [switch]$AutoCommit,
    [switch]$Once
)

$ErrorActionPreference = "Stop"
. (Join-Path $PSScriptRoot "lib\NativeBenchCommon.ps1")
$root = Get-CyphaRepoRoot -ScriptRoot $PSScriptRoot
$DEFAULT_BUILD_DIR = ""
$PRODUCTION_N_TRAIN_MIN = 300000
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

function Get-LockOvernightNTrain {
    param([string]$Path)

    $value = Get-LockOvernightNTrainValue -Path $Path
    if ($null -eq $value) {
        return "?"
    }

    return [string]$value
}

function Get-LockOvernightNTrainValue {
    param([string]$Path)

    if (-not (Test-Path $Path)) {
        return $null
    }

    try {
        $lock = Get-Content $Path -Raw | ConvertFrom-Json
    } catch {
        return $null
    }

    if ($lock.PSObject.Properties.Name -notcontains "overnight_results" -or $null -eq $lock.overnight_results) {
        return $null
    }

    $section = $lock.overnight_results
    if ($section.PSObject.Properties.Name -contains "n_train") {
        return [int]$section.n_train
    }

    return $null
}

function Write-PollHeartbeat {
    param(
        [int]$ProcessCount,
        [string]$LockPath
    )

    $nTrain = Get-LockOvernightNTrain -Path $LockPath
    $line = "HEARTBEAT $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') processes=$ProcessCount lock_n_train=$nTrain"
    Write-Host $line -ForegroundColor DarkGray
    if ($script:resolvedLogFile) {
        $line | Out-File -FilePath $script:resolvedLogFile -Append -Encoding utf8
    }
}

function Write-PollError {
    param([string]$Message)

    $line = "ERROR $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') $Message"
    Write-Host $line -ForegroundColor Red
    if ($script:resolvedLogFile) {
        $line | Out-File -FilePath $script:resolvedLogFile -Append -Encoding utf8
    }
}

function Write-PollAutoCommit {
    param([string]$Message)

    $line = "AUTO_COMMIT $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') $Message"
    Write-Host $line -ForegroundColor DarkGray
    if ($script:resolvedLogFile) {
        $line | Out-File -FilePath $script:resolvedLogFile -Append -Encoding utf8
    }
}

function Get-OvernightProcessInfoSafe {
    try {
        $procs = Get-OvernightProcessInfo
        $count = if ($procs) { $procs.Count } else { 0 }
        return @{
            Ok        = $true
            Processes = $procs
            Count     = $count
            Error     = $null
        }
    } catch {
        return @{
            Ok        = $false
            Processes = @()
            Count     = -1
            Error     = $_.Exception.Message
        }
    }
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

    $info = Get-OvernightProcessInfoSafe
    if (-not $info.Ok -or $info.Count -eq 0) {
        return $Requested
    }

    $detected = Get-DetectedBuildDirFromOvernight
    if ($detected) {
        Write-Host "poll_and_finalize_overnight: auto-detected BuildDir from run_production_overnight.ps1: $detected" -ForegroundColor Yellow
        return $detected
    }

    return (Get-DefaultNativeBuildDir -Override $Requested)
}

$BuildDir = Resolve-PollBuildDir -Requested (Get-DefaultNativeBuildDir -Override $BuildDir)

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
} elseif ($AutoCommit) {
    Write-Host "  commit: AutoCommit (-Force when lock n_train >= $PRODUCTION_N_TRAIN_MIN, else DryRun preview)" -ForegroundColor DarkGray
} else {
    Write-Host "  commit: DryRun preview (pass -Force or -AutoCommit)" -ForegroundColor DarkGray
}

if ($Once) {
    $onceInfo = Get-OvernightProcessInfoSafe
    if (-not $onceInfo.Ok) {
        Write-PollError -Message "poll check failed: $($onceInfo.Error)"
        $exitCode = 1
    } else {
        Write-PollHeartbeat -ProcessCount $onceInfo.Count -LockPath $LockFile
        if ($onceInfo.Count -gt 0) {
            Show-RunningProcesses
            Write-Host "poll_and_finalize_overnight: processes still running (-Once)" -ForegroundColor Yellow
            $exitCode = 1
        } else {
            Write-Host "[$(Get-Date -Format 'HH:mm:ss')] no overnight processes (-Once)" -ForegroundColor Green
        }
    }
} else {
    $pollHadErrors = $false
    while ($true) {
        $pollInfo = Get-OvernightProcessInfoSafe
        if (-not $pollInfo.Ok) {
            Write-PollError -Message "poll loop query failed: $($pollInfo.Error)"
            $pollHadErrors = $true
            Start-Sleep -Seconds $IntervalSeconds
            continue
        }

        Write-PollHeartbeat -ProcessCount $pollInfo.Count -LockPath $LockFile

        if ($pollInfo.Count -eq 0) {
            break
        }

        Show-RunningProcesses
        Start-Sleep -Seconds $IntervalSeconds
    }

    if ($pollHadErrors) {
        Write-Host "poll_and_finalize_overnight: poll loop had query errors (continuing to finalize after clean exit)" -ForegroundColor Yellow
        if ($resolvedLogFile) {
            Write-PollError -Message "poll loop had query errors (continuing to finalize after clean exit)"
        }
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
} elseif ($AutoCommit) {
    $autoNTrain = Get-LockOvernightNTrainValue -Path $LockFile
    $autoNTrainLabel = if ($null -eq $autoNTrain) { "?" } else { [string]$autoNTrain }
    Write-PollAutoCommit -Message "attempt n_train=$autoNTrainLabel threshold=$PRODUCTION_N_TRAIN_MIN"
    if ($null -ne $autoNTrain -and $autoNTrain -ge $PRODUCTION_N_TRAIN_MIN) {
        Write-Host "== commit_production_lock.ps1 -Force (AutoCommit) ==" -ForegroundColor Cyan
        & $commitScript -BuildDir $BuildDir -LockFile $LockFile -Force
    } else {
        Write-Host "== commit_production_lock.ps1 -DryRun (AutoCommit: n_train below threshold) ==" -ForegroundColor Cyan
        & $commitScript -BuildDir $BuildDir -LockFile $LockFile -DryRun
    }
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
