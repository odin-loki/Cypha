# Update bench/BASELINE_LOCK.json from a native bench run (Phase 8).

# Usage:

#   pwsh -File scripts/update_baseline_lock.ps1 -Run d17

#   pwsh -File scripts/update_baseline_lock.ps1 -Run d21 -NTrain 300000

#   pwsh -File scripts/update_baseline_lock.ps1 -Run cell-sweep -Fast -NTrain 200

#   pwsh -File scripts/update_baseline_lock.ps1 -Run all -Fast
#   pwsh -File scripts/update_baseline_lock.ps1 -Run d17 -Medium
#   pwsh -File scripts/update_baseline_lock.ps1 -Run all -Production

param(

    [Parameter(Mandatory = $true)]

    [ValidateSet("d17", "d21", "cell-sweep", "all")]

    [string]$Run,



    [int]$NTrain = 0,

    [int]$NEval = 0,

    [int]$Threads = 1,

    [switch]$Fast,

    [switch]$Medium,

    [switch]$Production,

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

if ($OutputDir -eq "" -and ($Run -eq "cell-sweep" -or $Run -eq "all")) {
    $OutputDir = "bench/results/cell_sweep"
}

$buildAbs = Join-Path $root $BuildDir

$tierCount = @($Fast, $Medium, $Production | Where-Object { $_ }).Count
if ($tierCount -gt 1) {
    throw "cannot combine -Fast, -Medium, and -Production"
}

if ($Fast -and -not $env:CYPHA_BENCH_FAST) {
    $env:CYPHA_BENCH_FAST = "1"
}



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

    if ($Medium) {

        $args += "--medium"

    }

    if ($Production) {

        $args += "--production"

    }

    if ($NTrain -gt 0) {

        $args += @("--n-train", "$NTrain")

    } elseif ($Fast) {

        # cypha_baseline_lock --fast default n_train=200

    } elseif ($Medium) {

        $args += @("--n-train", "5000")

    } elseif ($Production) {

        # cypha_baseline_lock --production default n_train=300000

    } else {

        $args += @("--n-train", "300000")

    }

    if ($NEval -gt 0) {

        $args += @("--n-eval", "$NEval")

    } elseif ($Fast) {

        # cypha_baseline_lock --fast default n_eval=64

    } elseif ($Medium) {

        $args += @("--n-eval", "256")

    } elseif ($Production) {

        # cypha_baseline_lock --production default n_eval=2000

    } else {

        $args += @("--n-eval", "2000")

    }

    if ($OutputDir -ne "") {

        $outAbs = if ([System.IO.Path]::IsPathRooted($OutputDir)) { $OutputDir } else { Join-Path $root $OutputDir }

        $args += @("--output-dir", $outAbs)

    }



    $tierLabel = if ($Fast) { '200 (fast default)' } elseif ($Medium) { '5000 (medium default)' } elseif ($Production) { '300000 (production default)' } else { '300000' }
    Write-Host "== baseline lock update (run=$RunName n_train=$(if ($NTrain -gt 0) { $NTrain } else { $tierLabel }) fast=$Fast medium=$Medium production=$Production) ==" -ForegroundColor Cyan

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

