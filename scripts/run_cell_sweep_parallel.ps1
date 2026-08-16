# Run remaining cell-hypothesis variants in parallel.
# Each process writes variant_<id>.json; manifest.json is rebuilt at the end from those files.
param(
    [string]$BuildDir = "",
    [int]$NTrain = 300000,
    [int]$NEval = 2000,
    [int]$Threads = 1,
    [int]$Parallelism = 4,
    [string]$Profile = "d17",
    [switch]$Production,
    [switch]$MathIntegration,
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"
. (Join-Path $PSScriptRoot "lib\NativeBenchCommon.ps1")

$root = Get-CyphaRepoRoot -ScriptRoot $PSScriptRoot
$BuildDir = Get-DefaultNativeBuildDir -Override $BuildDir
$buildAbs = Resolve-NativeBuildDir -RepoRoot $root -BuildDir $BuildDir
$sweepExe = Resolve-NativeExePath -BuildDir $buildAbs -Stem "cypha_cell_hypothesis_sweep"
if (-not $sweepExe) {
    throw "missing cypha_cell_hypothesis_sweep under $buildAbs"
}

$outDir = Join-Path $root "bench\results\cell_sweep"
New-Item -ItemType Directory -Force -Path $outDir | Out-Null
$logDir = Join-Path $root "bench\results\cell_sweep_logs"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null

$prevEap = $ErrorActionPreference
$ErrorActionPreference = "Continue"
$listJson = & $sweepExe --list-variants 2>$null
$ErrorActionPreference = $prevEap
$variants = $listJson | ConvertFrom-Json
$ids = @($variants | Where-Object { $_.runnable } | ForEach-Object { $_.id })
if ($ids.Count -eq 0) {
    throw "cypha_cell_hypothesis_sweep --list-variants returned no runnable ids"
}

function Test-VariantCheckpoint {
    param([string]$Id)
    $path = Join-Path $outDir "variant_$Id.json"
    if (-not (Test-Path $path)) { return $false }
    try {
        $row = Get-Content $path -Raw | ConvertFrom-Json
        $trainOk = [int]$row.n_train -eq $NTrain
        $evalOk = [int]$row.n_eval -eq $NEval
        $mathOk = (-not $MathIntegration) -or ([bool]$row.math_integration)
        return ($trainOk -and $evalOk -and $mathOk)
    } catch {
        return $false
    }
}

$pending = @($ids | Where-Object { -not (Test-VariantCheckpoint -Id $_) })
$done = $ids.Count - $pending.Count
Write-Host "== parallel cell sweep pending=$($pending.Count)/$($ids.Count) done=$done n_train=$NTrain parallelism=$Parallelism ==" -ForegroundColor Cyan
if ($pending.Count -eq 0) {
    Write-Host "All variant checkpoints already match. Nothing to run." -ForegroundColor Green
    exit 0
}

if ($DryRun) {
    Write-Host ("Would run: " + ($pending -join ", "))
    exit 0
}

$env:CYPHA_BENCH_FULL_CORPUS = "1"
$env:CYPHA_BENCH_OVERNIGHT = "1"
$env:CYPHA_BENCH_FULL_N_TRAIN = "$NTrain"
if ($Production) {
    $env:CYPHA_BENCH_PRODUCTION = "1"
}

$queue = [System.Collections.Queue]::new()
foreach ($id in $pending) { $queue.Enqueue($id) }
$running = @{}

function Start-VariantJob {
    param([string]$Id)
    $logPath = Join-Path $logDir "$Id.log"
    $errPath = Join-Path $logDir "$Id.err.log"
    $argList = @(
        "--cell-variant", $Id,
        "--profile", $Profile,
        "--n-train", "$NTrain",
        "--n-eval", "$NEval",
        "--threads", "$Threads",
        "--output-dir", $outDir
    )
    if ($MathIntegration) {
        $argList += @("--intelligence-profile", "--math-integration")
    }
    $p = Start-Process -FilePath $sweepExe -ArgumentList $argList -WorkingDirectory $root -RedirectStandardOutput $logPath -RedirectStandardError $errPath -PassThru -WindowStyle Hidden
    $running[$Id] = $p
    Write-Host "$(Get-Date -Format o) started $Id pid=$($p.Id)" -ForegroundColor Yellow
}

function Wait-AnyJob {
    while ($true) {
        $finished = @()
        foreach ($id in @($running.Keys)) {
            $proc = $running[$id]
            if ($proc.HasExited) {
                $finished += $id
            }
        }
        if ($finished.Count -gt 0) {
            foreach ($id in $finished) {
                $code = $running[$id].ExitCode
                $ok = Test-VariantCheckpoint -Id $id
                if ($ok) {
                    Write-Host "$(Get-Date -Format o) finished $id ok" -ForegroundColor Green
                } else {
                    Write-Host "$(Get-Date -Format o) finished $id FAIL exit=$code" -ForegroundColor Red
                }
                $running.Remove($id)
            }
            return
        }
        Start-Sleep -Seconds 15
    }
}

while ($queue.Count -gt 0 -or $running.Count -gt 0) {
    while ($running.Count -lt $Parallelism -and $queue.Count -gt 0) {
        Start-VariantJob -Id ([string]$queue.Dequeue())
    }
    if ($running.Count -gt 0) {
        Wait-AnyJob
    }
}

$have = @(Get-ChildItem -Path $outDir -Filter "variant_*.json" -File).Count
Write-Host "== parallel sweep complete checkpoints=$have/$($ids.Count) ==" -ForegroundColor Cyan
if ($have -lt $ids.Count) {
    exit 1
}
exit 0