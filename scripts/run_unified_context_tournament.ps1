# Unified-context BPC tournament (U01–U10) vs B2 control.
#
# Screen: all variants @ 40k/2k D17 WikiText
# Crown:  top 3 U-* + B2 @ 300k/2k
#
# Usage:
#   powershell -File scripts/run_unified_context_tournament.ps1
#   powershell -File scripts/run_unified_context_tournament.ps1 -BuildDir native/build-pgm -SkipBuild
#   powershell -File scripts/run_unified_context_tournament.ps1 -ScreenOnly
#   powershell -File scripts/run_unified_context_tournament.ps1 -CrownOnly -ScreenDir bench/results/unified_context_tournament/...

param(
    [string]$BuildDir = "native/build-pgm",
    [int]$ScreenTrain = 40000,
    [int]$CrownTrain = 300000,
    [int]$NEval = 2000,
    [int]$Threads = 1,
    [switch]$SkipBuild,
    [switch]$ScreenOnly,
    [switch]$CrownOnly,
    [string]$ScreenDir = ""
)

$ErrorActionPreference = "Stop"
. (Join-Path $PSScriptRoot "lib\NativeBenchCommon.ps1")
$root = Get-CyphaRepoRoot -ScriptRoot $PSScriptRoot

$variants = @("B2", "U01", "U02", "U03", "U04", "U05", "U06", "U07", "U08", "U09", "U10")

function Resolve-BenchExe {
    $candidates = @(
        (Join-Path $root (Join-Path $BuildDir "cyphalm_bench_native.exe")),
        (Join-Path $root (Join-Path $BuildDir "cyphalm_bench_native")),
        (Join-Path $root (Join-Path $BuildDir "Release\cyphalm_bench_native.exe"))
    )
    foreach ($c in $candidates) {
        if (Test-Path $c) { return $c }
    }
    return $null
}

function Extract-BenchJson {
    param([string]$Text)
    $idx = $Text.LastIndexOf("`n{")
    if ($idx -lt 0) { $idx = $Text.IndexOf("{") } else { $idx += 1 }
    if ($idx -lt 0) { throw "no JSON blob in bench stdout" }
    return $Text.Substring($idx)
}

if (-not $SkipBuild) {
    $buildAbs = Join-Path $root $BuildDir
    if (-not (Test-Path (Join-Path $buildAbs "CMakeCache.txt"))) {
        cmake -S (Join-Path $root "native") -B $buildAbs -DCMAKE_BUILD_TYPE=Release
        if ($LASTEXITCODE -ne 0) { throw "cmake configure failed" }
    }
    cmake --build $buildAbs --config Release --target cyphalm_bench_native
    if ($LASTEXITCODE -ne 0) { throw "build failed" }
}

$exe = Resolve-BenchExe
if (-not $exe) { throw "missing cyphalm_bench_native under $BuildDir" }

Remove-Item Env:CYPHA_BENCH_FAST -ErrorAction SilentlyContinue
$env:CYPHA_BENCH_FULL_CORPUS = "1"
$env:CYPHA_BENCH_OVERNIGHT = "1"

$resultsRoot = Join-Path $root "bench\results\unified_context_tournament"
New-Item -ItemType Directory -Force -Path $resultsRoot | Out-Null

if ($CrownOnly) {
    if (-not $ScreenDir) { throw "-CrownOnly requires -ScreenDir" }
    $runDir = $ScreenDir
    if (-not [System.IO.Path]::IsPathRooted($runDir)) {
        $runDir = Join-Path $root $runDir
    }
} else {
    $timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
    $runDir = Join-Path $resultsRoot $timestamp
    New-Item -ItemType Directory -Force -Path $runDir | Out-Null
}

$logPath = Join-Path $runDir "tournament.log"
Write-Host "Unified-context tournament -> $runDir" -ForegroundColor Cyan

function Write-Log([string]$msg) {
    $line = "[{0}] {1}" -f (Get-Date -Format "HH:mm:ss"), $msg
    Add-Content -Path $logPath -Value $line
    Write-Host $line
}

$screenRows = @()

if (-not $CrownOnly) {
    Write-Log "SCREEN n_train=$ScreenTrain n_eval=$NEval variants=$($variants -join ',')"
    foreach ($v in $variants) {
        Write-Log "screen $v"
        $outJson = Join-Path $runDir ("screen_{0}.json" -f $v)
        $outLog = Join-Path $runDir ("screen_{0}.log" -f $v)
        $args = @(
            "--profile", "d17",
            "--cell-variant", $v,
            "--n-train", "$ScreenTrain",
            "--n-eval", "$NEval",
            "--threads", "$Threads",
            "--bench-seed", "42"
        )
        $prev = $ErrorActionPreference
        $ErrorActionPreference = "Continue"
        try {
            $raw = & $exe @args 2>&1 | Tee-Object -FilePath $outLog
        } finally {
            $ErrorActionPreference = $prev
        }
        if ($LASTEXITCODE -ne 0) {
            Write-Log "FAIL screen $v exit=$LASTEXITCODE"
            $screenRows += [pscustomobject]@{ variant = $v; bpc = [double]::PositiveInfinity; ok = $false }
            continue
        }
        $text = ($raw | ForEach-Object { "$_" }) -join "`n"
        $blob = Extract-BenchJson $text
        Set-Content -Path $outJson -Value $blob -Encoding utf8
        $obj = $blob | ConvertFrom-Json
        $bpc = [double]$obj.bpc
        Write-Log ("screen {0} bpc={1:F4}" -f $v, $bpc)
        $screenRows += [pscustomobject]@{ variant = $v; bpc = $bpc; ok = $true; json = "screen_$v.json" }
    }

    $screenRows | ConvertTo-Json -Depth 4 | Set-Content (Join-Path $runDir "screen_summary.json") -Encoding utf8
    $rankedU = $screenRows |
        Where-Object { $_.ok -and $_.variant -ne "B2" } |
        Sort-Object bpc
    $top3 = @($rankedU | Select-Object -First 3 | ForEach-Object { $_.variant })
    @{ top3 = $top3; screen = $screenRows } | ConvertTo-Json -Depth 5 |
        Set-Content (Join-Path $runDir "screen_rank.json") -Encoding utf8
    Write-Log ("top3 U-variants: {0}" -f ($top3 -join ", "))
} else {
    $rankPath = Join-Path $runDir "screen_rank.json"
    if (-not (Test-Path $rankPath)) { throw "missing screen_rank.json in $runDir" }
    $rank = Get-Content $rankPath -Raw | ConvertFrom-Json
    $top3 = @($rank.top3)
    $screenRows = @($rank.screen)
}

if ($ScreenOnly) {
    Write-Log "ScreenOnly — stopping before crown"
    exit 0
}

$crownList = @("B2") + $top3 | Select-Object -Unique
Write-Log "CROWN n_train=$CrownTrain variants=$($crownList -join ',')"
$crownRows = @()
foreach ($v in $crownList) {
    Write-Log "crown $v"
    $outJson = Join-Path $runDir ("crown_{0}.json" -f $v)
    $outLog = Join-Path $runDir ("crown_{0}.log" -f $v)
    $args = @(
        "--profile", "d17",
        "--cell-variant", $v,
        "--n-train", "$CrownTrain",
        "--n-eval", "$NEval",
        "--threads", "$Threads",
        "--bench-seed", "42"
    )
    $prev = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        $raw = & $exe @args 2>&1 | Tee-Object -FilePath $outLog
    } finally {
        $ErrorActionPreference = $prev
    }
    if ($LASTEXITCODE -ne 0) {
        Write-Log "FAIL crown $v exit=$LASTEXITCODE"
        $crownRows += [pscustomobject]@{ variant = $v; bpc = [double]::PositiveInfinity; ok = $false }
        continue
    }
    $text = ($raw | ForEach-Object { "$_" }) -join "`n"
    $blob = Extract-BenchJson $text
    Set-Content -Path $outJson -Value $blob -Encoding utf8
    $obj = $blob | ConvertFrom-Json
    $bpc = [double]$obj.bpc
    Write-Log ("crown {0} bpc={1:F4}" -f $v, $bpc)
    $crownRows += [pscustomobject]@{ variant = $v; bpc = $bpc; ok = $true; json = "crown_$v.json" }
}

$b2Row = $crownRows | Where-Object { $_.variant -eq "B2" -and $_.ok } | Select-Object -First 1
$b2Bpc = if ($b2Row) { [double]$b2Row.bpc } else { 2.873 }
$winner = $crownRows | Where-Object { $_.ok } | Sort-Object bpc | Select-Object -First 1
$bestU = $crownRows | Where-Object { $_.ok -and $_.variant -ne "B2" } | Sort-Object bpc | Select-Object -First 1

$summary = [ordered]@{
    run_dir = $runDir
    screen_n_train = $ScreenTrain
    crown_n_train = $CrownTrain
    n_eval = $NEval
    screen = $screenRows
    top3 = $top3
    crown = $crownRows
    b2_crown_bpc = $b2Bpc
    winner = if ($winner) { $winner.variant } else { $null }
    winner_bpc = if ($winner) { $winner.bpc } else { $null }
    best_unified = if ($bestU) { $bestU.variant } else { $null }
    best_unified_bpc = if ($bestU) { $bestU.bpc } else { $null }
    beats_b2 = if ($winner -and $winner.variant -ne "B2") { $true } else { $false }
}
$summary | ConvertTo-Json -Depth 6 | Set-Content (Join-Path $runDir "summary.json") -Encoding utf8

$screenTable = ($screenRows | Sort-Object bpc | ForEach-Object {
    $bpcCell = if ($_.ok) { '{0:F4}' -f $_.bpc } else { 'FAIL' }
    '| {0} | {1} |' -f $_.variant, $bpcCell
}) -join "`n"
$crownTable = ($crownRows | Sort-Object bpc | ForEach-Object {
    if (-not $_.ok) {
        '| {0} | FAIL | - |' -f $_.variant
    } else {
        $delta = $_.bpc - $b2Bpc
        '| {0} | {1:F4} | {2:+0.4f} |' -f $_.variant, $_.bpc, $delta
    }
}) -join "`n"
$top3Text = ($top3 -join ', ')
$winnerLine = ''
if ($winner) {
    $winnerLine = '**Winner:** {0} @ {1:F4} BPC' -f $winner.variant, $winner.bpc
    if ($bestU) {
        $winnerLine += "`n`n" + ('**Best unified spine:** {0} @ {1:F4} BPC (delta vs B2 crown = {2:+0.4f})' -f `
            $bestU.variant, $bestU.bpc, ($bestU.bpc - $b2Bpc))
    }
}

$mdLines = New-Object System.Collections.Generic.List[string]
$mdLines.Add('# Unified-context BPC tournament')
$mdLines.Add('')
$mdLines.Add(('Artifacts: {0}' -f $runDir))
$mdLines.Add('')
$mdLines.Add(('## Screen ({0} train)' -f $ScreenTrain))
$mdLines.Add('')
$mdLines.Add('| Variant | BPC |')
$mdLines.Add('|---------|-----|')
foreach ($line in ($screenTable -split "`n")) { if ($line) { $mdLines.Add($line) } }
$mdLines.Add('')
$mdLines.Add(('Top 3 U-*: {0}' -f $top3Text))
$mdLines.Add('')
$mdLines.Add(('## Crown ({0} train)' -f $CrownTrain))
$mdLines.Add('')
$mdLines.Add('| Variant | BPC | vs B2 |')
$mdLines.Add('|---------|-----|-------|')
foreach ($line in ($crownTable -split "`n")) { if ($line) { $mdLines.Add($line) } }
$mdLines.Add('')
if ($winnerLine) { foreach ($line in ($winnerLine -split "`n")) { $mdLines.Add($line) } }
Set-Content -Path (Join-Path $runDir "SUMMARY.md") -Value ($mdLines -join "`n") -Encoding utf8

Write-Log ('WINNER={0} bpc={1}' -f $summary.winner, $summary.winner_bpc)
Write-Host "Done. summary.json + SUMMARY.md in $runDir" -ForegroundColor Green
