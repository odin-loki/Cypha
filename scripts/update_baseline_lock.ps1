# Update bench/BASELINE_LOCK.json from a native bench run (Phase 8).
# Usage:
#   pwsh -File scripts/update_baseline_lock.ps1 -Run d17
#   pwsh -File scripts/update_baseline_lock.ps1 -Run d21 -NTrain 300000
#   pwsh -File scripts/update_baseline_lock.ps1 -Run cell-sweep -Fast -NTrain 200
#   pwsh -File scripts/update_baseline_lock.ps1 -Run all -Fast
param(
    [Parameter(Mandatory = $true)]
    [ValidateSet("d17", "d21", "cell-sweep", "all")]
    [string]$Run,

    [int]$NTrain = 0,
    [int]$NEval = 0,
    [int]$Threads = 1,
    [switch]$Fast,
    [string]$BuildDir = "native/build",
    [string]$LockFile = "bench/BASELINE_LOCK.json",
    [string]$OutputDir = ""
)

$ErrorActionPreference = "Stop"

$root = Split-Path $PSScriptRoot -Parent
$exe = Join-Path $root (Join-Path $BuildDir "cypha_baseline_lock.exe")
if (-not (Test-Path $exe)) {
    $exe = Join-Path $root (Join-Path $BuildDir "cypha_baseline_lock")
}
if (-not (Test-Path $exe)) {
    throw "missing cypha_baseline_lock under $BuildDir (build native first: cmake --build $BuildDir --target cypha_baseline_lock)"
}

$lockPath = if ([System.IO.Path]::IsPathRooted($LockFile)) { $LockFile } else { Join-Path $root $LockFile }
$buildAbs = Join-Path $root $BuildDir

function Invoke-BaselineLockRun {
    param([string]$RunName)

    $args = @(
        "--run", $RunName,
        "--threads", "$Threads",
        "--lock-file", $lockPath,
        "--exe-dir", $buildAbs
    )
    if ($Fast) {
        $args += "--fast"
    }
    if ($NTrain -gt 0) {
        $args += @("--n-train", "$NTrain")
    } elseif (-not $Fast) {
        $args += @("--n-train", "300000")
    }
    if ($NEval -gt 0) {
        $args += @("--n-eval", "$NEval")
    } elseif (-not $Fast) {
        $args += @("--n-eval", "2000")
    }
    if ($OutputDir -ne "") {
        $outAbs = if ([System.IO.Path]::IsPathRooted($OutputDir)) { $OutputDir } else { Join-Path $root $OutputDir }
        $args += @("--output-dir", $outAbs)
    }

    Write-Host "== baseline lock update (run=$RunName n_train=$(if ($NTrain -gt 0) { $NTrain } elseif ($Fast) { '200 (fast default)' } else { '300000' }) fast=$Fast) ==" -ForegroundColor Cyan
    & $exe @args
    if ($LASTEXITCODE -ne 0) {
        throw "cypha_baseline_lock failed exit=$LASTEXITCODE (run=$RunName)"
    }
}

if ($Run -eq "all") {
    foreach ($r in @("d17", "d21", "cell-sweep")) {
        Invoke-BaselineLockRun -RunName $r
    }
} else {
    Invoke-BaselineLockRun -RunName $Run
}

Write-Host "Updated $lockPath" -ForegroundColor Green
