# Full native Cypha validation: Release build outside OneDrive, CTest, bench smoke.
# Phase 20 (v2.3.20, prep): d34 repo smoke hygiene validation, CYPHA_VALIDATE_REPO_SMOKE_HYGIENE, 112 CTests when d34 merged.
# Phase 19 (v2.3.19, shipped): d33 release publish validation, CYPHA_VALIDATE_RELEASE_PUBLISH, 111 CTests.
#
# Optional environment variables (Phase 13–20):
#   CYPHA_VALIDATE_PRODUCTION=1
#       Run validate_baseline_lock.ps1 -Production after the standard lock check.
#   CYPHA_VALIDATE_OVERNIGHT_COMPLETE=1
#       After baseline lock validate, run cypha_bench_run --domain-tag d28 (overnight completion).
#   CYPHA_VALIDATE_RELEASE_READINESS=1
#       After baseline lock validate, run cypha_bench_run --domain-tag d29 when its profile exists;
#       gracefully skipped when d29 is not built/merged yet.
#   CYPHA_VALIDATE_ARTIFACT_HYGIENE=1
#       After baseline lock validate, run cypha_bench_run --domain-tag d30 when its profile exists;
#       gracefully skipped when d30 is not built/merged yet.
#   CYPHA_VALIDATE_POST_OVERNIGHT_PIPELINE=1
#       After baseline lock validate, run cypha_bench_run --domain-tag d31 when its profile exists;
#       gracefully skipped when d31 is not built/merged yet.
#   CYPHA_VALIDATE_PRODUCTION_COMPLETE=1
#       After baseline lock validate, run cypha_bench_run --domain-tag d32 when its profile exists;
#       gracefully skipped when d32 is not built/merged yet.
#   CYPHA_VALIDATE_RELEASE_PUBLISH=1
#       After production complete (d32) step, run cypha_bench_run --domain-tag d33 when its profile exists;
#       gracefully skipped when d33 is not built/merged yet.
#   CYPHA_VALIDATE_REPO_SMOKE_HYGIENE=1
#       After release publish (d33) step, run cypha_bench_run --domain-tag d34 when its profile exists;
#       gracefully skipped when d34 is not built/merged yet.
#   CYPHA_STRICT_TEST_COUNT=1
#       Fail the ctest_native step when the parsed native_ test count does not match the expected
#       gate (112 with d34 merged, else 111 with d33 merged, else 110 with d32 merged, else 109 with d31 merged, else 108 with d30 merged, else 107). Default: warn in step detail only.
#
# Usage:
#   pwsh -File scripts/cypha_native_validate_all.ps1
#   $env:CYPHA_VALIDATE_PRODUCTION = "1"; pwsh -File scripts/cypha_native_validate_all.ps1 -SkipBuild
param(
    [string]$BuildDir = "C:\Temp\cypha_full_cpp_build",
    [switch]$SkipBuild,
    [switch]$SkipBench,
    [switch]$TuneSmoke
)

$ErrorActionPreference = "Stop"
$root = Split-Path $PSScriptRoot -Parent
$results = [ordered]@{}

function Step-Result {
    param([string]$Name, [bool]$Ok, [string]$Detail = "")
    $results[$Name] = @{ Ok = $Ok; Detail = $Detail }
}

function BinPath {
    param([string]$Stem)
    Join-Path $BuildDir "$Stem.exe"
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

function Get-ExpectedNativeTestCount {
    if (Test-DomainTagExists -Tag "d34") { return 112 }
    if (Test-DomainTagExists -Tag "d33") { return 111 }
    if (Test-DomainTagExists -Tag "d32") { return 110 }
    if (Test-DomainTagExists -Tag "d31") { return 109 }
    if (Test-DomainTagExists -Tag "d30") { return 108 }
    return 107
}

function Get-CtestPassedCount {
    param([object]$Output)
    $text = if ($Output -is [array]) { $Output -join "`n" } else { [string]$Output }
    foreach ($line in ($text -split "`n")) {
        if ($line -match '(\d+)\s+tests?\s+failed out of (\d+)' -and $line -match 'tests passed') {
            return [int]$Matches[2]
        }
    }
    if ($text -match '(?m)(\d+)\s+tests?\s+passed') {
        return [int]$Matches[1]
    }
    return $null
}

Write-Host "Cypha full native validation" -ForegroundColor Cyan
Write-Host "  repo:     $root"
Write-Host "  build:    $BuildDir"
Write-Host ""

# --- Build ---
if (-not $SkipBuild) {
    Write-Host "== CMake configure (Release) ==" -ForegroundColor Yellow
    New-Item -ItemType Directory -Force -Path $BuildDir | Out-Null
    cmake -S (Join-Path $root "native") -B $BuildDir `
        -DCMAKE_BUILD_TYPE=Release `
        -DCYPHA_BUILD_QT=OFF `
        -DCYPHA_BUILD_EXPERIMENT_DB=ON `
        -G Ninja
    if ($LASTEXITCODE -ne 0) {
        Step-Result "build_configure" $false "cmake configure failed"
        throw "cmake configure failed"
    }

    Write-Host "== CMake build (all targets) ==" -ForegroundColor Yellow
    cmake --build $BuildDir --parallel
    if ($LASTEXITCODE -ne 0) {
        Step-Result "build_compile" $false "cmake --build failed"
        throw "cmake --build failed"
    }
    Step-Result "build" $true "Release build at $BuildDir"
} else {
    if (-not (Test-Path $BuildDir)) {
        throw "BuildDir missing: $BuildDir (drop -SkipBuild or build first)"
    }
    Step-Result "build" $true "skipped (-SkipBuild)"
}

$env:QT_QPA_PLATFORM = "offscreen"

# --- CTest ---
Write-Host ""
Write-Host "== CTest (-R native_) ==" -ForegroundColor Yellow
$prevEap = $ErrorActionPreference
$ErrorActionPreference = "Continue"
try {
    $ctestOut = & ctest --test-dir $BuildDir -R native_ --output-on-failure 2>&1
    $ctestCode = $LASTEXITCODE
} finally {
    $ErrorActionPreference = $prevEap
}
$ctestOut | Write-Host
$ctestDetail = if ($ctestCode -eq 0) { "all native_ tests passed" } else { "exit $ctestCode" }
$expectedTestCount = Get-ExpectedNativeTestCount
$parsedTestCount = Get-CtestPassedCount -Output $ctestOut
if ($null -ne $parsedTestCount -and $parsedTestCount -ne $expectedTestCount) {
    $countWarn = "expected $expectedTestCount native_ tests, parsed $parsedTestCount"
    $ctestDetail = if ($ctestDetail) { "$ctestDetail; warn: $countWarn" } else { "warn: $countWarn" }
    if ($env:CYPHA_STRICT_TEST_COUNT -eq "1") {
        Step-Result "ctest_native" $false $ctestDetail
    } else {
        Step-Result "ctest_native" ($ctestCode -eq 0) $ctestDetail
    }
} else {
    if ($null -ne $parsedTestCount) {
        $ctestDetail = if ($ctestDetail) { "$ctestDetail ($parsedTestCount/$expectedTestCount)" } else { "$parsedTestCount/$expectedTestCount" }
    }
    Step-Result "ctest_native" ($ctestCode -eq 0) $ctestDetail
}

# --- Bench smoke ---
if (-not $SkipBench) {
    Write-Host ""
    Write-Host "== cypha_bench_run smoke (d01, d04, d17) ==" -ForegroundColor Yellow
    $benchExe = BinPath "cypha_bench_run"
    if (-not (Test-Path $benchExe)) {
        Step-Result "bench_smoke" $false "missing $benchExe"
        Step-Result "bench_report_png" $false "missing $benchExe"
    } else {
        $benchOk = $true
        $benchDetail = @()
        foreach ($domain in @(1, 4, 17)) {
            Write-Host "--domain $domain --"
            Push-Location $root
            try {
                & $benchExe --domain $domain
                $code = $LASTEXITCODE
            } finally {
                Pop-Location
            }
            $dTag = "d{0:D2}" -f $domain
            if ($code -ne 0) {
                $benchOk = $false
                $benchDetail += "$dTag`: exit $code"
            } else {
                $benchDetail += "$dTag`: ok"
            }
        }

        Write-Host ""
        Write-Host "== d03_xor kernel LLR smoke (fast) ==" -ForegroundColor Yellow
        $env:CYPHA_BENCH_FAST = "1"
        Push-Location $root
        try {
            & $benchExe --domain-tag d03_xor
            $xorCode = $LASTEXITCODE
        } finally {
            Pop-Location
            Remove-Item Env:CYPHA_BENCH_FAST -ErrorAction SilentlyContinue
        }
        if ($xorCode -ne 0) {
            $benchOk = $false
            $benchDetail += "d03_xor: exit $xorCode"
        } else {
            $benchDetail += "d03_xor: ok"
        }
        Step-Result "bench_smoke" $benchOk ($benchDetail -join "; ")

        Write-Host ""
        Write-Host "== cypha_bench_run --report-only (PNG figures) ==" -ForegroundColor Yellow
        Push-Location $root
        try {
            & $benchExe --report-only
            $reportCode = $LASTEXITCODE
        } finally {
            Pop-Location
        }
        if ($reportCode -ne 0) {
            Step-Result "bench_report_png" $false "exit $reportCode"
        } else {
            $figuresDir = Join-Path $root "bench\report\figures"
            $manifestPath = Join-Path $figuresDir "figures_manifest.json"
            $pngOk = $true
            $pngDetail = @()
            if (-not (Test-Path $manifestPath)) {
                $pngOk = $false
                $pngDetail += "missing figures_manifest.json"
            } else {
                $manifest = Get-Content $manifestPath -Raw | ConvertFrom-Json
                foreach ($fig in $manifest.figures) {
                    $pngPath = Join-Path $figuresDir "$fig.png"
                    if (Test-Path $pngPath) {
                        $pngDetail += "$fig.png: ok"
                    } else {
                        $pngOk = $false
                        $pngDetail += "$fig.png: missing"
                    }
                }
            }
            Step-Result "bench_report_png" $pngOk ($pngDetail -join "; ")
        }
    }
} else {
    Step-Result "bench_smoke" $true "skipped (-SkipBench)"
    Step-Result "bench_report_png" $true "skipped (-SkipBench)"
}

# --- cypha_tune_run smoke ---
Write-Host ""
if ($TuneSmoke) {
    Write-Host "== cypha_tune_smoke (all configs, live run) ==" -ForegroundColor Yellow
    $tuneScript = Join-Path $root "scripts\cypha_tune_smoke.ps1"
    if (-not (Test-Path $tuneScript)) {
        Step-Result "tune_run_smoke" $false "missing $tuneScript"
    } else {
        & $tuneScript -BuildDir $BuildDir -MaxCells 4 -Write
        $tuneCode = $LASTEXITCODE
        Step-Result "tune_run_smoke" ($tuneCode -eq 0) $(if ($tuneCode -eq 0) { "live smoke ok" } else { "exit $tuneCode" })
    }
} else {
    Write-Host "== cypha_tune_run smoke (--dry-run, all configs) ==" -ForegroundColor Yellow
    $tuneScript = Join-Path $root "scripts\cypha_tune_smoke.ps1"
    if (-not (Test-Path $tuneScript)) {
        Step-Result "tune_run_smoke" $false "missing $tuneScript"
    } else {
        & $tuneScript -BuildDir $BuildDir -MaxCells 4 -DryRun
        $tuneCode = $LASTEXITCODE
        Step-Result "tune_run_smoke" ($tuneCode -eq 0) $(if ($tuneCode -eq 0) { "dry-run ok (pass -TuneSmoke for live)" } else { "exit $tuneCode" })
    }
}

# --- cypha_rest /dif REST smoke (curl) ---
Write-Host ""
Write-Host "== cypha_rest /dif/retrieve smoke (curl) ==" -ForegroundColor Yellow
$restExe = BinPath "cypha_rest"
$refCypha = Join-Path $root "fixtures\reference.cypha"
$fField = Join-Path $root "fixtures\f_field.json"
$retrievalSidecar = Join-Path $root "fixtures\retrieval\sidecar.json"
if (-not (Test-Path $restExe)) {
    Step-Result "rest_dif_smoke" $false "missing $restExe"
} elseif (-not (Test-Path $refCypha) -or -not (Test-Path $fField) -or -not (Test-Path $retrievalSidecar)) {
    Step-Result "rest_dif_smoke" $false "missing fixtures (reference.cypha, f_field.json, retrieval/sidecar.json)"
} else {
    $listener = [System.Net.Sockets.TcpListener]::new([System.Net.IPAddress]::Loopback, 0)
    $listener.Start()
    try {
        $port = $listener.LocalEndpoint.Port
    } finally {
        $listener.Stop()
    }
    $listenAddr = "127.0.0.1:$port"
    $baseUrl = "http://$listenAddr"
    $restArgs = @(
        "--listen", $listenAddr,
        "--cypha", $refCypha,
        "--f-field-json", $fField
    )
    $restProc = Start-Process -FilePath $restExe -ArgumentList $restArgs -PassThru -WindowStyle Hidden
    $difOk = $false
    $difDetail = ""
    try {
        $deadline = (Get-Date).AddSeconds(20)
        $healthy = $false
        while ((Get-Date) -lt $deadline) {
            if ($restProc.HasExited) {
                $difDetail = "cypha_rest exited early (code $($restProc.ExitCode))"
                break
            }
            try {
                $health = curl.exe -s -o NUL -w "%{http_code}" "$baseUrl/health"
                if ($health -eq "200") {
                    $healthy = $true
                    break
                }
            } catch { }
            Start-Sleep -Milliseconds 200
        }
        if (-not $healthy -and -not $difDetail) {
            $difDetail = "cypha_rest health timeout"
        } elseif ($healthy) {
            $sidecar = Get-Content $retrievalSidecar -Raw | ConvertFrom-Json
            $case = $sidecar.cases[0]
            $inputArr = @()
            foreach ($v in $case.query_x) { $inputArr += [double]$v }
            $dbArr = @()
            foreach ($row in $case.database_x) {
                $r = @()
                foreach ($v in $row) { $r += [double]$v }
                $dbArr += ,$r
            }
            $payloadObj = [ordered]@{
                input    = $inputArr
                database = $dbArr
                top_k    = [int]$case.top_k
            }
            if ($case.PSObject.Properties.Name -contains "label" -and $null -ne $case.label) {
                $payloadObj.label = [string]$case.label
            }
            $payloadJson = $payloadObj | ConvertTo-Json -Compress -Depth 20
            $tmpBody = Join-Path $env:TEMP "cypha_dif_retrieve_smoke.json"
            [System.IO.File]::WriteAllText($tmpBody, $payloadJson)
            $curlOut = curl.exe -s -S -X POST "$baseUrl/dif/retrieve" `
                -H "Content-Type: application/json" `
                --data-binary "@$tmpBody"
            $curlCode = $LASTEXITCODE
            Remove-Item -Force $tmpBody -ErrorAction SilentlyContinue
            if ($curlCode -ne 0) {
                $difDetail = "curl exit $curlCode"
            } else {
                try {
                    $body = $curlOut | ConvertFrom-Json
                    if ($body.hits -and $body.hits.Count -gt 0) {
                        $difOk = $true
                        $difDetail = "hits=$($body.hits.Count)"
                    } else {
                        $difDetail = "empty hits in response"
                    }
                } catch {
                    $difDetail = "bad JSON: $($curlOut.Substring(0, [Math]::Min(120, $curlOut.Length)))"
                }
            }
        }
    } finally {
        if (-not $restProc.HasExited) {
            Stop-Process -Id $restProc.Id -Force -ErrorAction SilentlyContinue
            $restProc.WaitForExit(5000) | Out-Null
        }
    }
    Step-Result "rest_dif_smoke" $difOk $difDetail
}

# --- BASELINE_LOCK.json validation ---
Write-Host ""
Write-Host "== validate_baseline_lock.ps1 ==" -ForegroundColor Yellow
$baselineLockScript = Join-Path $root "scripts\validate_baseline_lock.ps1"
if (-not (Test-Path $baselineLockScript)) {
    Step-Result "baseline_lock_validate" $true "skipped (script missing)"
} else {
    $baselineLockArgs = @{}
    $productionValidate = ($env:CYPHA_VALIDATE_PRODUCTION -eq "1")
    if ($productionValidate) {
        $baselineLockArgs.Production = $true
        Write-Host "  CYPHA_VALIDATE_PRODUCTION=1 -> -Production tier checks" -ForegroundColor DarkGray
    }
    & $baselineLockScript @baselineLockArgs
    $lockCode = $LASTEXITCODE
    $lockDetail = if ($lockCode -eq 0) {
        if ($productionValidate) { "bench/BASELINE_LOCK.json ok (-Production)" } else { "bench/BASELINE_LOCK.json ok" }
    } else {
        "exit $lockCode"
    }
    Step-Result "baseline_lock_validate" ($lockCode -eq 0) $lockDetail
}

# --- d28 overnight completion (optional) ---
if ($env:CYPHA_VALIDATE_OVERNIGHT_COMPLETE -eq "1") {
    Write-Host ""
    Write-Host "== cypha_bench_run --domain-tag d28 (CYPHA_VALIDATE_OVERNIGHT_COMPLETE) ==" -ForegroundColor Yellow
    $benchExe = BinPath "cypha_bench_run"
    if (-not (Test-Path $benchExe)) {
        Step-Result "overnight_complete_d28" $false "missing $benchExe"
    } else {
        Push-Location $root
        try {
            & $benchExe --domain-tag d28
            $d28Code = $LASTEXITCODE
        } finally {
            Pop-Location
        }
        Step-Result "overnight_complete_d28" ($d28Code -eq 0) $(if ($d28Code -eq 0) { "d28 ok" } else { "exit $d28Code" })
    }
}

# --- d29 release readiness (optional) ---
if ($env:CYPHA_VALIDATE_RELEASE_READINESS -eq "1") {
    Write-Host ""
    if (Test-DomainTagExists -Tag "d29") {
        Write-Host "== cypha_bench_run --domain-tag d29 (CYPHA_VALIDATE_RELEASE_READINESS) ==" -ForegroundColor Yellow
        $benchExe = BinPath "cypha_bench_run"
        if (-not (Test-Path $benchExe)) {
            Step-Result "release_readiness_d29" $false "missing $benchExe"
        } else {
            Push-Location $root
            try {
                & $benchExe --domain-tag d29
                $d29Code = $LASTEXITCODE
            } finally {
                Pop-Location
            }
            Step-Result "release_readiness_d29" ($d29Code -eq 0) $(if ($d29Code -eq 0) { "d29 ok" } else { "exit $d29Code" })
        }
    } else {
        Write-Host "== cypha_bench_run --domain-tag d29 (skipped - profile not present) ==" -ForegroundColor DarkGray
        Step-Result "release_readiness_d29" $true "skipped (d29 profile not present)"
    }
}

# --- d30 artifact hygiene (optional) ---
if ($env:CYPHA_VALIDATE_ARTIFACT_HYGIENE -eq "1") {
    Write-Host ""
    if (Test-DomainTagExists -Tag "d30") {
        Write-Host "== cypha_bench_run --domain-tag d30 (CYPHA_VALIDATE_ARTIFACT_HYGIENE) ==" -ForegroundColor Yellow
        $benchExe = BinPath "cypha_bench_run"
        if (-not (Test-Path $benchExe)) {
            Step-Result "artifact_hygiene_d30" $false "missing $benchExe"
        } else {
            Push-Location $root
            try {
                & $benchExe --domain-tag d30
                $d30Code = $LASTEXITCODE
            } finally {
                Pop-Location
            }
            Step-Result "artifact_hygiene_d30" ($d30Code -eq 0) $(if ($d30Code -eq 0) { "d30 ok" } else { "exit $d30Code" })
        }
    } else {
        Write-Host "== cypha_bench_run --domain-tag d30 (skipped - profile not present) ==" -ForegroundColor DarkGray
        Step-Result "artifact_hygiene_d30" $true "skipped (d30 profile not present)"
    }
}

# --- d31 post-overnight pipeline (optional) ---
if ($env:CYPHA_VALIDATE_POST_OVERNIGHT_PIPELINE -eq "1") {
    Write-Host ""
    if (Test-DomainTagExists -Tag "d31") {
        Write-Host "== cypha_bench_run --domain-tag d31 (CYPHA_VALIDATE_POST_OVERNIGHT_PIPELINE) ==" -ForegroundColor Yellow
        $benchExe = BinPath "cypha_bench_run"
        if (-not (Test-Path $benchExe)) {
            Step-Result "post_overnight_pipeline_d31" $false "missing $benchExe"
        } else {
            Push-Location $root
            try {
                & $benchExe --domain-tag d31
                $d31Code = $LASTEXITCODE
            } finally {
                Pop-Location
            }
            Step-Result "post_overnight_pipeline_d31" ($d31Code -eq 0) $(if ($d31Code -eq 0) { "d31 ok" } else { "exit $d31Code" })
        }
    } else {
        Write-Host "== cypha_bench_run --domain-tag d31 (skipped - profile not present) ==" -ForegroundColor DarkGray
        Step-Result "post_overnight_pipeline_d31" $true "skipped (d31 profile not present)"
    }
}

# --- d32 production complete (optional) ---
if ($env:CYPHA_VALIDATE_PRODUCTION_COMPLETE -eq "1") {
    Write-Host ""
    if (Test-DomainTagExists -Tag "d32") {
        Write-Host "== cypha_bench_run --domain-tag d32 (CYPHA_VALIDATE_PRODUCTION_COMPLETE) ==" -ForegroundColor Yellow
        $benchExe = BinPath "cypha_bench_run"
        if (-not (Test-Path $benchExe)) {
            Step-Result "production_complete_d32" $false "missing $benchExe"
        } else {
            Push-Location $root
            try {
                & $benchExe --domain-tag d32
                $d32Code = $LASTEXITCODE
            } finally {
                Pop-Location
            }
            Step-Result "production_complete_d32" ($d32Code -eq 0) $(if ($d32Code -eq 0) { "d32 ok" } else { "exit $d32Code" })
        }
    } else {
        Write-Host "== cypha_bench_run --domain-tag d32 (skipped - profile not present) ==" -ForegroundColor DarkGray
        Step-Result "production_complete_d32" $true "skipped (d32 profile not present)"
    }
}

# --- d33 release publish (optional) ---
if ($env:CYPHA_VALIDATE_RELEASE_PUBLISH -eq "1") {
    Write-Host ""
    if (Test-DomainTagExists -Tag "d33") {
        Write-Host "== cypha_bench_run --domain-tag d33 (CYPHA_VALIDATE_RELEASE_PUBLISH) ==" -ForegroundColor Yellow
        $benchExe = BinPath "cypha_bench_run"
        if (-not (Test-Path $benchExe)) {
            Step-Result "release_publish_d33" $false "missing $benchExe"
        } else {
            Push-Location $root
            try {
                & $benchExe --domain-tag d33
                $d33Code = $LASTEXITCODE
            } finally {
                Pop-Location
            }
            Step-Result "release_publish_d33" ($d33Code -eq 0) $(if ($d33Code -eq 0) { "d33 ok" } else { "exit $d33Code" })
        }
    } else {
        Write-Host "== cypha_bench_run --domain-tag d33 (skipped - profile not present) ==" -ForegroundColor DarkGray
        Step-Result "release_publish_d33" $true "skipped (d33 profile not present)"
    }
}

# --- d34 repo smoke hygiene (optional) ---
if ($env:CYPHA_VALIDATE_REPO_SMOKE_HYGIENE -eq "1") {
    Write-Host ""
    if (Test-DomainTagExists -Tag "d34") {
        Write-Host "== cypha_bench_run --domain-tag d34 (CYPHA_VALIDATE_REPO_SMOKE_HYGIENE) ==" -ForegroundColor Yellow
        $benchExe = BinPath "cypha_bench_run"
        if (-not (Test-Path $benchExe)) {
            Step-Result "repo_smoke_hygiene_d34" $false "missing $benchExe"
        } else {
            Push-Location $root
            try {
                & $benchExe --domain-tag d34
                $d34Code = $LASTEXITCODE
            } finally {
                Pop-Location
            }
            Step-Result "repo_smoke_hygiene_d34" ($d34Code -eq 0) $(if ($d34Code -eq 0) { "d34 ok" } else { "exit $d34Code" })
        }
    } else {
        Write-Host "== cypha_bench_run --domain-tag d34 (skipped - profile not present) ==" -ForegroundColor DarkGray
        Step-Result "repo_smoke_hygiene_d34" $true "skipped (d34 profile not present)"
    }
}

# --- Summary ---
Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "VALIDATION SUMMARY" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
$allOk = $true
foreach ($entry in $results.GetEnumerator()) {
    $icon = if ($entry.Value.Ok) { "PASS" } else { "FAIL" }
    $color = if ($entry.Value.Ok) { "Green" } else { "Red"; $allOk = $false }
    $detail = if ($entry.Value.Detail) { " - $($entry.Value.Detail)" } else { "" }
    Write-Host ("  [{0}] {1}{2}" -f $icon, $entry.Key, $detail) -ForegroundColor $color
}
Write-Host "========================================" -ForegroundColor Cyan

if ($allOk) {
    Write-Host "OK cypha_native_validate_all" -ForegroundColor Green
    exit 0
}
Write-Host "FAILED cypha_native_validate_all" -ForegroundColor Red
exit 1
