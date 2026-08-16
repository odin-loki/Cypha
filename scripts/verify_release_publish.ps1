# Phase 19 (shipped): release publish smoke - production-complete gate + d33 bench + publish -DryRun preview.
# Does not call gh; authenticate manually before a real publish:
#   gh auth login
#   gh auth status
# Then: pwsh -File scripts/publish_release.ps1 -Tag v2.4.0
#
# Usage:
#   pwsh -File scripts/verify_release_publish.ps1
#   pwsh -File scripts/verify_release_publish.ps1 -BuildDir native/build -Tag v2.4.0
#   pwsh -File scripts/verify_release_publish.ps1 -AllowPending
param(
    [string]$BuildDir = "",
    [string]$Tag = "",
    [switch]$AllowPending
)

$ErrorActionPreference = "Stop"
. (Join-Path $PSScriptRoot "lib\NativeBenchCommon.ps1")
$root = Get-CyphaRepoRoot -ScriptRoot $PSScriptRoot
$BuildDir = Get-DefaultNativeBuildDir -Override $BuildDir
$DEFAULT_TAG = "v2.4.0"
$PRODUCTION_N_TRAIN_MIN = 300000

$validateCompleteScript = Join-Path $PSScriptRoot "validate_production_complete.ps1"
$publishScript = Join-Path $PSScriptRoot "publish_release.ps1"
$lockFile = Join-Path $root "bench\BASELINE_LOCK.json"

function Resolve-ReleaseTag {
    param([string]$Requested)

    if ($Requested) {
        return $Requested
    }

    $git = Get-Command git -ErrorAction SilentlyContinue
    if (-not $git) {
        return $DEFAULT_TAG
    }

    Push-Location $root
    try {
        $prevEap = $ErrorActionPreference
        $ErrorActionPreference = "Continue"
        $described = & git describe --tags --abbrev=0 2>$null
        $ErrorActionPreference = $prevEap
        if ($LASTEXITCODE -eq 0 -and $described) {
            return $described.Trim()
        }
    } finally {
        Pop-Location
    }

    return $DEFAULT_TAG
}

function Get-LockNTrain {
    param([string]$Path)

    if (-not (Test-Path $Path)) {
        return 0
    }

    try {
        $lock = Get-Content $Path -Raw | ConvertFrom-Json
    } catch {
        return 0
    }

    if ($lock.PSObject.Properties.Name -notcontains "overnight_results" -or $null -eq $lock.overnight_results) {
        return 0
    }

    $overnight = $lock.overnight_results
    if ($overnight.PSObject.Properties.Name -notcontains "n_train") {
        return 0
    }

    return [int]$overnight.n_train
}

function Test-DomainTagExists {
    param([string]$Tag)

    $profiles = Get-ChildItem -Path (Join-Path $root "bench\config") -Filter "${Tag}_*_profile.json" -ErrorAction SilentlyContinue
    if ($profiles) { return $true }

    $indexPath = Join-Path $root "bench\config\profiles_index.json"
    if (Test-Path $indexPath) {
        try {
            $index = Get-Content $indexPath -Raw | ConvertFrom-Json
            foreach ($prop in $index.PSObject.Properties) {
                if ($prop.Value.domain -eq $Tag) { return $true }
            }
        } catch { }
    }

    return $false
}

function Resolve-BenchExe {
    param([string]$Dir)

    $exe = Join-Path $root (Join-Path $Dir "cypha_bench_run.exe")
    if (Test-Path $exe) { return $exe }

    $exe = Join-Path $root (Join-Path $Dir "cypha_bench_run")
    if (Test-Path $exe) { return $exe }

    return $null
}

$resolvedTag = Resolve-ReleaseTag -Requested $Tag
$useAllowPending = $AllowPending
if (-not $useAllowPending) {
    $nTrain = Get-LockNTrain -Path $lockFile
    if ($nTrain -lt $PRODUCTION_N_TRAIN_MIN) {
        $useAllowPending = $true
        Write-Host "verify_release_publish: lock n_train=$nTrain below $PRODUCTION_N_TRAIN_MIN - auto -AllowPending" -ForegroundColor Yellow
    }
}

Write-Host "== verify release publish (smoke) ==" -ForegroundColor Cyan
Write-Host "  build: $BuildDir"
Write-Host "  tag:   $resolvedTag"
Write-Host "  gh:    manual step - run 'gh auth login' before real publish (this script uses -DryRun only)" -ForegroundColor DarkGray
if ($useAllowPending) {
    Write-Host "  mode:  -AllowPending (smoke / pending_production)" -ForegroundColor DarkGray
}

Write-Host ""
Write-Host "== validate_production_complete.ps1 ==" -ForegroundColor Cyan
$validateArgs = @{
    BuildDir = $BuildDir
}
if ($useAllowPending) {
    $validateArgs.AllowPending = $true
}
& $validateCompleteScript @validateArgs
if ($LASTEXITCODE -ne 0) {
    Write-Host "verify_release_publish: FAIL (validate_production_complete)" -ForegroundColor Red
    exit $LASTEXITCODE
}

$benchExe = Resolve-BenchExe -Dir $BuildDir
if ($benchExe -and (Test-DomainTagExists -Tag "d33")) {
    Write-Host ""
    Write-Host "== cypha_bench_run --domain-tag d33 ==" -ForegroundColor Cyan
    Push-Location $root
    try {
        & $benchExe --domain-tag d33
        if ($LASTEXITCODE -ne 0) {
            Write-Host "verify_release_publish: FAIL (d33 bench)" -ForegroundColor Red
            exit $LASTEXITCODE
        }
    } finally {
        Pop-Location
    }
} elseif (-not $benchExe) {
    Write-Host ""
    Write-Host "== cypha_bench_run --domain-tag d33 (skipped - exe not found under $BuildDir) ==" -ForegroundColor DarkGray
} else {
    Write-Host ""
    Write-Host "== cypha_bench_run --domain-tag d33 (skipped - domain not present) ==" -ForegroundColor DarkGray
}

Write-Host ""
Write-Host "== publish_release.ps1 -DryRun ==" -ForegroundColor Cyan
& $publishScript -Tag $resolvedTag -DryRun
if ($LASTEXITCODE -ne 0) {
    Write-Host "verify_release_publish: FAIL (publish_release -DryRun)" -ForegroundColor Red
    exit $LASTEXITCODE
}

Write-Host ""
Write-Host "verify_release_publish: OK - smoke checks passed" -ForegroundColor Green
Write-Host "Next (manual): gh auth login; gh auth status" -ForegroundColor Yellow
Write-Host "Then publish:  pwsh -File scripts/publish_release.ps1 -Tag $resolvedTag" -ForegroundColor Yellow
exit 0
