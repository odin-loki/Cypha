# Validate bench/BASELINE_LOCK.json schema, d17 pin, and overnight/rpsm/cell-sweep sections.
param(
    [string]$LockFile = "",
    [switch]$Strict
)

$ErrorActionPreference = "Stop"
$root = Split-Path $PSScriptRoot -Parent
if (-not $LockFile) {
    $LockFile = Join-Path $root "bench\BASELINE_LOCK.json"
}

$D17_PIN_BPC = 2.873
$D17_PIN_TOLERANCE = 0.02
$VALID_STATUSES = @("fast_smoke", "completed")

function Fail([string]$Message) {
    Write-Host "validate_baseline_lock: FAIL - $Message" -ForegroundColor Red
    exit 1
}

function Require-Object([object]$Parent, [string]$Key, [string]$Context) {
    if ($null -eq $Parent -or -not ($Parent.PSObject.Properties.Name -contains $Key)) {
        Fail "$Context missing '$Key'"
    }
    $val = $Parent.$Key
    if ($null -eq $val) {
        Fail "$Context '$Key' is null"
    }
    return $val
}

function Validate-ResultSection {
    param(
        [object]$Section,
        [string]$Name,
        [string]$ExpectedProfile,
        [string]$ExpectedMode
    )

    if ($null -eq $Section -or $Section -isnot [pscustomobject]) {
        Fail "$Name is not an object"
    }

    $status = Require-Object $Section "status" $Name
    if ($status -isnot [string]) {
        Fail "$Name status must be a string"
    }
    if ($status -eq "pending") {
        Fail "$Name status is pending"
    }
    if ($VALID_STATUSES -notcontains $status) {
        Fail "$Name status '$status' is not a recognized value ($($VALID_STATUSES -join ', '))"
    }

    if ($Strict -and $Name -eq "overnight_results" -and $status -eq "fast_smoke") {
        Fail "overnight_results status is fast_smoke (-Strict requires completed overnight run)"
    }

    $bpc = Require-Object $Section "bpc" $Name
    try {
        [void][double]$bpc
    } catch {
        Fail "$Name bpc must be numeric"
    }

    $runAt = Require-Object $Section "run_at" $Name
    if ($runAt -isnot [string] -or [string]::IsNullOrWhiteSpace($runAt)) {
        Fail "$Name run_at must be a non-empty string"
    }

    foreach ($key in @("profile", "mode", "n_train", "n_eval", "runner", "env")) {
        Require-Object $Section $key $Name | Out-Null
    }

    if ($Section.profile -ne $ExpectedProfile) {
        Fail "$Name profile expected '$ExpectedProfile', got '$($Section.profile)'"
    }
    if ($Section.mode -ne $ExpectedMode) {
        Fail "$Name mode expected '$ExpectedMode', got '$($Section.mode)'"
    }
    if ($Section.env -isnot [pscustomobject]) {
        Fail "$Name env must be an object"
    }
}

if (-not (Test-Path $LockFile)) {
    Fail "lock file not found: $LockFile"
}

try {
    $lock = Get-Content $LockFile -Raw | ConvertFrom-Json
} catch {
    Fail "invalid JSON in $LockFile`: $($_.Exception.Message)"
}

if ($null -eq $lock.schema_version -or [int]$lock.schema_version -ne 1) {
    Fail "schema_version must be 1 (got '$($lock.schema_version)')"
}

$d17 = Require-Object $lock "d17_hybrid_baseline" "lock"
if ($d17 -isnot [pscustomobject]) {
    Fail "d17_hybrid_baseline is not an object"
}

foreach ($key in @("bpc", "profile", "mode", "n_train", "n_eval")) {
    Require-Object $d17 $key "d17_hybrid_baseline" | Out-Null
}

if ($d17.profile -ne "d17") {
    Fail "d17_hybrid_baseline profile must be d17"
}
if ($d17.mode -ne "hybrid") {
    Fail "d17_hybrid_baseline mode must be hybrid"
}

$bpcDelta = [Math]::Abs([double]$d17.bpc - $D17_PIN_BPC)
if ($bpcDelta -gt $D17_PIN_TOLERANCE) {
    Fail ("d17_hybrid_baseline bpc pin {0} out of tolerance (expected ~{1}, delta {2:F4})" -f $d17.bpc, $D17_PIN_BPC, $bpcDelta)
}

Validate-ResultSection $lock.overnight_results "overnight_results" "d17" "hybrid"
Validate-ResultSection $lock.rpsm_results "rpsm_results" "d21" "rpsm"

if ($lock.PSObject.Properties.Name -contains "cell_sweep_results" -and $null -ne $lock.cell_sweep_results) {
    Validate-ResultSection $lock.cell_sweep_results "cell_sweep_results" "d17" "cell-sweep"
}

Write-Host "validate_baseline_lock: OK $LockFile" -ForegroundColor Green
if ($Strict) {
    Write-Host "  (-Strict: overnight_results not fast_smoke)" -ForegroundColor DarkGray
}
exit 0
