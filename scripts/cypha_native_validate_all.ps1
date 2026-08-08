# Full native Cypha validation: Release build outside OneDrive, CTest, bench smoke.
# Phase 59 (v2.3.59, prep): d74-d76 structural grids, 160 CTests when d76 merged.
# Phase 53 (v2.3.53, prep): d67 kernel blend grid, 151 CTests when d67 merged.
# Phase 52 (v2.3.52, prep): d66 free energy beta grid, 150 CTests when d66 merged.
# Phase 51 (v2.3.51, prep): d65 navigation loss warmup grid, 149 CTests when d65 merged.
# Phase 50 (v2.3.50, prep): d64 kappa trajectory window grid, 148 CTests when d64 merged.
# Phase 49 (v2.3.49, prep): d63 reu forget blend grid, 147 CTests when d63 merged.
# Phase 48 (v2.3.48, prep): d62 math ablation stack complete, 146 CTests when d62 merged.
# Phase 47 (v2.3.47, prep): d61 excess grad scale grid joint, 145 CTests when d61 merged.
# Phase 46 (v2.3.46, prep): d60 excess grad margin grid joint, 144 CTests when d60 merged.
# Phase 45 (v2.3.45, prep): d59 kernel blend floor grid joint, 143 CTests when d59 merged.
# Phase 44 (v2.3.44, prep): d58 production overnight math complete, 142 CTests when d58 merged.
# Phase 43 (v2.3.43, prep): d57 production cell sweep math certificate, 141 CTests when d57 merged.
# Phase 42 (v2.3.42, prep): d56 cell sweep math integration gate, 140 CTests when d56 merged.
# Phase 41 (v2.3.41, prep): d54 production math certificate + d55 nav warmup grid, 139 CTests when d55 merged.
# Phase 40 (v2.3.40, prep): d53 production preset ship lock gate, 137 CTests when d53 + κ nav warmup merged.
# Phase 39 (v2.3.39, prep): d52 preset ship lock gate, 136 CTests when d52 + r_eu preset merged.
# Phase 38 (v2.3.38, prep): d51 opt-in lever joint gate, 135 CTests when d51 + κ kernel blend merged.
# Phase 37 (v2.3.37, prep): d50 pinned-seed joint lock gate, 134 CTests when d50 + excess grad margin merged.
# Phase 36 (v2.3.36, prep): d49 ceiling grid joint gate, 133 CTests when d49 + trajectory ceiling merged.
# Phase 35 (v2.3.35, prep): d48 κ ceiling ablation gate, 132 CTests when d48 + eigenvalue D_eff merged.
# Phase 34 (v2.3.34, prep): d47 span ablation gate, 131 CTests when d47 + kappa ceiling merged.
# Phase 33 (v2.3.33, prep): d46 math stack upgrade gate, 130 CTests when d46 + tau smoke merged.
# Phase 32 (v2.3.32, prep): d45 per-stat navigation λ gate, 128 CTests when d45 merged.
# Phase 31 (v2.3.31, prep): d44 Nyström kernel CyphaLM gate, 127 CTests when d44 merged.
# Phase 30 (v2.3.30, prep): d43 math integration lock gate, 126 CTests when d43 merged.
# Phase 28 (v2.3.28, prep): d42 production math gate, adaptive λ, CharLSTM nav loss, d20/d22 κ ranking, 123 CTests when d42 + nav smoke merged.
# Phase 27 (v2.3.27, prep): d41 math integration scale validation, CYPHA_VALIDATE_MATH_INTEGRATION_SCALE, 121 CTests when d41 + lm_self_correct merged.
# Phase 26 (v2.3.26, prep): d40 math integration validation, CYPHA_VALIDATE_MATH_INTEGRATION, 118 CTests when d40 merged.
# Phase 25 (v2.3.25, prep): d39 intelligence monitor profile validation, CYPHA_VALIDATE_INTELLIGENCE_MONITOR, 117 CTests when d39 merged.
# Phase 24 (v2.3.24, prep): d38 overnight completion certificate validation, CYPHA_VALIDATE_OVERNIGHT_CERTIFICATE, 116 CTests when d38 merged.
# Phase 23 (v2.3.23, shipped): d37 overnight lock refresh validation, CYPHA_VALIDATE_LOCK_REFRESH, 115 CTests.
# Phase 22 (v2.3.22, shipped): d36 production pipeline E2E validation, CYPHA_VALIDATE_PIPELINE_E2E, 114 CTests.
# Phase 21 (v2.3.21, shipped): d35 lock commit pipeline validation, CYPHA_VALIDATE_LOCK_COMMIT_PIPELINE, 113 CTests when d35 merged.
# Phase 20 (v2.3.20, shipped): d34 repo smoke hygiene validation, CYPHA_VALIDATE_REPO_SMOKE_HYGIENE, 112 CTests when d34 merged.
# Phase 19 (v2.3.19, shipped): d33 release publish validation, CYPHA_VALIDATE_RELEASE_PUBLISH, 111 CTests.
#
# Optional environment variables (Phase 13–24):
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
#   CYPHA_VALIDATE_LOCK_COMMIT_PIPELINE=1
#       After repo smoke hygiene (d34) step, run cypha_bench_run --domain-tag d35 when its profile exists;
#       gracefully skipped when d35 is not built/merged yet.
#   CYPHA_VALIDATE_PIPELINE_E2E=1
#       After lock commit pipeline (d35) step, run cypha_bench_run --domain-tag d36 when its profile exists;
#       gracefully skipped when d36 is not built/merged yet.
#   CYPHA_VALIDATE_LOCK_REFRESH=1
#       After pipeline E2E (d36) step, run cypha_bench_run --domain-tag d37 when its profile exists;
#       gracefully skipped when d37 is not built/merged yet.
#   CYPHA_VALIDATE_OVERNIGHT_CERTIFICATE=1
#       After lock refresh (d37) step, run cypha_bench_run --domain-tag d38 when its profile exists;
#       gracefully skipped when d38 is not built/merged yet.
#   CYPHA_VALIDATE_INTELLIGENCE_MONITOR=1
#       After overnight certificate (d38) step, run cypha_bench_run --domain-tag d39 when its profile exists;
#       gracefully skipped when d39 is not built/merged yet.
#
#   CYPHA_VALIDATE_MATH_INTEGRATION=1
#       After intelligence monitor (d39) step, run cypha_bench_run --domain-tag d40 when its profile exists;
#       gracefully skipped when d40 is not built/merged yet.
#   CYPHA_VALIDATE_MATH_INTEGRATION_SCALE=1
#       After math integration (d40) step, run cypha_bench_run --domain-tag d41 when its profile exists;
#       gracefully skipped when d41 is not built/merged yet.
#   CYPHA_VALIDATE_MATH_INTEGRATION_PRODUCTION=1
#       After math integration scale (d41) step, run cypha_bench_run --domain-tag d42 when its profile exists;
#       gracefully skipped when d42 is not built/merged yet.
#   CYPHA_VALIDATE_MATH_INTEGRATION_LOCK=1
#       After math integration production (d42) step, run cypha_bench_run --domain-tag d43 when its profile exists;
#       gracefully skipped when d43 is not built/merged yet.
#   CYPHA_VALIDATE_KERNEL_NYSTROM_CYPHALM=1
#       After math integration lock (d43) step, run cypha_bench_run --domain-tag d44 when its profile exists;
#       gracefully skipped when d44 is not built/merged yet.
#   CYPHA_VALIDATE_PER_STAT_NAVIGATION=1
#   CYPHA_VALIDATE_MATH_STACK_UPGRADE=1
#   CYPHA_VALIDATE_SPAN_ABLATION=1
#   CYPHA_VALIDATE_KAPPA_CEILING_ABLATION=1
#   CYPHA_VALIDATE_CEILING_GRID_JOINT=1
#   CYPHA_VALIDATE_MATH_JOINT_LOCK=1
#   CYPHA_VALIDATE_OPT_IN_LEVER_JOINT=1
#   CYPHA_VALIDATE_PRESET_SHIP_LOCK=1
#   CYPHA_VALIDATE_PRODUCTION_PRESET_SHIP_LOCK=1
#   CYPHA_VALIDATE_PRODUCTION_MATH_CERTIFICATE=1
#   CYPHA_VALIDATE_NAV_WARMUP_GRID_JOINT=1
#   CYPHA_VALIDATE_CELL_SWEEP_MATH_INTEGRATION=1
#   CYPHA_VALIDATE_PRODUCTION_CELL_SWEEP_MATH_CERTIFICATE=1
#   CYPHA_VALIDATE_PRODUCTION_OVERNIGHT_MATH_COMPLETE=1
#   CYPHA_VALIDATE_KERNEL_BLEND_FLOOR_GRID_JOINT=1
#   CYPHA_VALIDATE_EXCESS_GRAD_MARGIN_GRID_JOINT=1
#   CYPHA_VALIDATE_EXCESS_GRAD_SCALE_GRID_JOINT=1
#   CYPHA_VALIDATE_MATH_ABLATION_STACK_COMPLETE=1
#   CYPHA_VALIDATE_REU_FORGET_BLEND_GRID_JOINT=1
#   CYPHA_VALIDATE_KAPPA_TRAJECTORY_WINDOW_GRID_JOINT=1
#   CYPHA_VALIDATE_NAVIGATION_LOSS_WARMUP_GRID_JOINT=1
#   CYPHA_VALIDATE_FREE_ENERGY_BETA_GRID_JOINT=1
#   CYPHA_VALIDATE_KERNEL_BLEND_GRID_JOINT=1
#   CYPHA_VALIDATE_KERNEL_M_GRID_JOINT=1
#       After kernel Nyström (d44) step, run cypha_bench_run --domain-tag d45 when its profile exists;
#       gracefully skipped when d45 is not built/merged yet.
#   CYPHA_STRICT_TEST_COUNT=1
#       Fail the ctest_native step when the parsed native_ test count does not match the expected
#       gate (128 with d45 merged, else 127 with d44 merged, else 126 with d43 merged, else 124 with d42 merged, else 121 with d41 merged, else 118 with d40 merged, else 117 with d39 merged, else 116 with d38 merged, else 115 with d37 merged, else 114 with d36 merged, else 113 with d35 merged, else 112 with d34 merged, else 111 with d33 merged, else 110 with d32 merged, else 109 with d31 merged, else 108 with d30 merged, else 107). Default: warn in step detail only.
#
# Usage:
#   pwsh -File scripts/cypha_native_validate_all.ps1
#   $env:CYPHA_VALIDATE_PRODUCTION = "1"; pwsh -File scripts/cypha_native_validate_all.ps1 -SkipBuild
param(
    # Default resolved below via Get-DefaultNativeBuildDir when not passed explicitly.
    [string]$BuildDir = "",
    [switch]$SkipBuild,
    [switch]$SkipBench,
    [switch]$TuneSmoke
)

$ErrorActionPreference = "Stop"
. (Join-Path $PSScriptRoot "lib\NativeBenchCommon.ps1")

$root = Get-CyphaRepoRoot -ScriptRoot $PSScriptRoot
$BuildDir = Get-DefaultNativeBuildDir -Override $BuildDir
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
    param([string]$BuildDir = "")
    if ($BuildDir -and (Test-Path (Join-Path $BuildDir "CTestTestfile.cmake"))) {
        try {
            $listOut = & ctest --test-dir $BuildDir -N -R native_ 2>&1 | Out-String
            if ($listOut -match 'Total Tests:\s*(\d+)') {
                return [int]$Matches[1]
            }
        } catch { }
    }
    return 214
}

function Get-CtestPassedCount {
    param([object]$Output)
    $text = if ($Output -is [array]) { $Output -join "`n" } else { [string]$Output }
    foreach ($line in ($text -split "`n")) {
        if ($line -match '(\d+)\s+tests?\s+failed out of (\d+)') {
            $failed = [int]$Matches[1]
            $total = [int]$Matches[2]
            return $total - $failed
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
$expectedTestCount = Get-ExpectedNativeTestCount -BuildDir $BuildDir
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
    Write-Host "== cypha_bench_run smoke (d01, d04, d17, forecast) ==" -ForegroundColor Yellow
    $benchExe = BinPath "cypha_bench_run"
    if (-not (Test-Path $benchExe)) {
        Step-Result "bench_smoke" $false "missing $benchExe"
        Step-Result "bench_report_png" $false "missing $benchExe"
    } else {
        $fetchScript = Join-Path $root "scripts\fetch_forecast_data.ps1"
        if (Test-Path $fetchScript) {
            Write-Host "== fetch_forecast_data (sample aliases) ==" -ForegroundColor Yellow
            Push-Location $root
            try {
                & $fetchScript
                if ($LASTEXITCODE -ne 0) {
                    Write-Host "fetch_forecast_data warning: exit $LASTEXITCODE" -ForegroundColor DarkYellow
                }
            } finally {
                Pop-Location
            }
            Write-Host ""
        }
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

        Write-Host ""
        Write-Host "== forecast bench smoke ==" -ForegroundColor Yellow
        Push-Location $root
        try {
            & $benchExe --domain-tag forecast
            $forecastCode = $LASTEXITCODE
        } finally {
            Pop-Location
        }
        if ($forecastCode -ne 0) {
            $benchOk = $false
            $benchDetail += "forecast: exit $forecastCode"
        } else {
            $benchDetail += "forecast: ok"
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
Write-Host "== cypha_rest /retrieve smoke (curl) ==" -ForegroundColor Yellow
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
            $curlOut = curl.exe -s -S -X POST "$baseUrl/retrieve" `
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

# --- d35 lock commit pipeline (optional) ---
if ($env:CYPHA_VALIDATE_LOCK_COMMIT_PIPELINE -eq "1") {
    Write-Host ""
    if (Test-DomainTagExists -Tag "d35") {
        Write-Host "== cypha_bench_run --domain-tag d35 (CYPHA_VALIDATE_LOCK_COMMIT_PIPELINE) ==" -ForegroundColor Yellow
        $benchExe = BinPath "cypha_bench_run"
        if (-not (Test-Path $benchExe)) {
            Step-Result "lock_commit_pipeline_d35" $false "missing $benchExe"
        } else {
            Push-Location $root
            try {
                & $benchExe --domain-tag d35
                $d35Code = $LASTEXITCODE
            } finally {
                Pop-Location
            }
            Step-Result "lock_commit_pipeline_d35" ($d35Code -eq 0) $(if ($d35Code -eq 0) { "d35 ok" } else { "exit $d35Code" })
        }
    } else {
        Write-Host "== cypha_bench_run --domain-tag d35 (skipped - profile not present) ==" -ForegroundColor DarkGray
        Step-Result "lock_commit_pipeline_d35" $true "skipped (d35 profile not present)"
    }
}

# --- d36 pipeline E2E (optional) ---
if ($env:CYPHA_VALIDATE_PIPELINE_E2E -eq "1") {
    Write-Host ""
    if (Test-DomainTagExists -Tag "d36") {
        Write-Host "== cypha_bench_run --domain-tag d36 (CYPHA_VALIDATE_PIPELINE_E2E) ==" -ForegroundColor Yellow
        $benchExe = BinPath "cypha_bench_run"
        if (-not (Test-Path $benchExe)) {
            Step-Result "pipeline_e2e_d36" $false "missing $benchExe"
        } else {
            Push-Location $root
            try {
                & $benchExe --domain-tag d36
                $d36Code = $LASTEXITCODE
            } finally {
                Pop-Location
            }
            Step-Result "pipeline_e2e_d36" ($d36Code -eq 0) $(if ($d36Code -eq 0) { "d36 ok" } else { "exit $d36Code" })
        }
    } else {
        Write-Host "== cypha_bench_run --domain-tag d36 (skipped - profile not present) ==" -ForegroundColor DarkGray
        Step-Result "pipeline_e2e_d36" $true "skipped (d36 profile not present)"
    }
}

# --- d37 lock refresh (optional) ---
if ($env:CYPHA_VALIDATE_LOCK_REFRESH -eq "1") {
    Write-Host ""
    if (Test-DomainTagExists -Tag "d37") {
        Write-Host "== cypha_bench_run --domain-tag d37 (CYPHA_VALIDATE_LOCK_REFRESH) ==" -ForegroundColor Yellow
        $benchExe = BinPath "cypha_bench_run"
        if (-not (Test-Path $benchExe)) {
            Step-Result "lock_refresh_d37" $false "missing $benchExe"
        } else {
            Push-Location $root
            try {
                & $benchExe --domain-tag d37
                $d37Code = $LASTEXITCODE
            } finally {
                Pop-Location
            }
            Step-Result "lock_refresh_d37" ($d37Code -eq 0) $(if ($d37Code -eq 0) { "d37 ok" } else { "exit $d37Code" })
        }
    } else {
        Write-Host "== cypha_bench_run --domain-tag d37 (skipped - profile not present) ==" -ForegroundColor DarkGray
        Step-Result "lock_refresh_d37" $true "skipped (d37 profile not present)"
    }
}

# --- d38 overnight certificate (optional) ---
if ($env:CYPHA_VALIDATE_OVERNIGHT_CERTIFICATE -eq "1") {
    Write-Host ""
    if (Test-DomainTagExists -Tag "d38") {
        Write-Host "== cypha_bench_run --domain-tag d38 (CYPHA_VALIDATE_OVERNIGHT_CERTIFICATE) ==" -ForegroundColor Yellow
        $benchExe = BinPath "cypha_bench_run"
        if (-not (Test-Path $benchExe)) {
            Step-Result "overnight_certificate_d38" $false "missing $benchExe"
        } else {
            Push-Location $root
            try {
                & $benchExe --domain-tag d38
                $d38Code = $LASTEXITCODE
            } finally {
                Pop-Location
            }
            Step-Result "overnight_certificate_d38" ($d38Code -eq 0) $(if ($d38Code -eq 0) { "d38 ok" } else { "exit $d38Code" })
        }
    } else {
        Write-Host "== cypha_bench_run --domain-tag d38 (skipped - profile not present) ==" -ForegroundColor DarkGray
        Step-Result "overnight_certificate_d38" $true "skipped (d38 profile not present)"
    }
}

# --- d39 intelligence monitor profile (optional) ---
if ($env:CYPHA_VALIDATE_INTELLIGENCE_MONITOR -eq "1") {
    Write-Host ""
    if (Test-DomainTagExists -Tag "d39") {
        Write-Host "== cypha_bench_run --domain-tag d39 (CYPHA_VALIDATE_INTELLIGENCE_MONITOR) ==" -ForegroundColor Yellow
        $benchExe = BinPath "cypha_bench_run"
        if (-not (Test-Path $benchExe)) {
            Step-Result "intelligence_monitor_d39" $false "missing $benchExe"
        } else {
            Push-Location $root
            try {
                & $benchExe --domain-tag d39
                $d39Code = $LASTEXITCODE
            } finally {
                Pop-Location
            }
            Step-Result "intelligence_monitor_d39" ($d39Code -eq 0) $(if ($d39Code -eq 0) { "d39 ok" } else { "exit $d39Code" })
        }
    } else {
        Write-Host "== cypha_bench_run --domain-tag d39 (skipped - profile not present) ==" -ForegroundColor DarkGray
        Step-Result "intelligence_monitor_d39" $true "skipped (d39 profile not present)"
    }
}

# --- d40 math integration (optional) ---
if ($env:CYPHA_VALIDATE_MATH_INTEGRATION -eq "1") {
    Write-Host ""
    if (Test-DomainTagExists -Tag "d40") {
        Write-Host "== cypha_bench_run --domain-tag d40 (CYPHA_VALIDATE_MATH_INTEGRATION) ==" -ForegroundColor Yellow
        $benchExe = BinPath "cypha_bench_run"
        if (-not (Test-Path $benchExe)) {
            Step-Result "math_integration_d40" $false "missing $benchExe"
        } else {
            Push-Location $root
            try {
                & $benchExe --domain-tag d40
                $d40Code = $LASTEXITCODE
            } finally {
                Pop-Location
            }
            Step-Result "math_integration_d40" ($d40Code -eq 0) $(if ($d40Code -eq 0) { "d40 ok" } else { "exit $d40Code" })
        }
    } else {
        Write-Host "== cypha_bench_run --domain-tag d40 (skipped - profile not present) ==" -ForegroundColor DarkGray
        Step-Result "math_integration_d40" $true "skipped (d40 profile not present)"
    }
}

# --- d41 math integration scale (optional) ---
if ($env:CYPHA_VALIDATE_MATH_INTEGRATION_SCALE -eq "1") {
    Write-Host ""
    if (Test-DomainTagExists -Tag "d41") {
        Write-Host "== cypha_bench_run --domain-tag d41 (CYPHA_VALIDATE_MATH_INTEGRATION_SCALE) ==" -ForegroundColor Yellow
        $benchExe = BinPath "cypha_bench_run"
        if (-not (Test-Path $benchExe)) {
            Step-Result "math_integration_scale_d41" $false "missing $benchExe"
        } else {
            Push-Location $root
            try {
                & $benchExe --domain-tag d41
                $d41Code = $LASTEXITCODE
            } finally {
                Pop-Location
            }
            Step-Result "math_integration_scale_d41" ($d41Code -eq 0) $(if ($d41Code -eq 0) { "d41 ok" } else { "exit $d41Code" })
        }
    } else {
        Write-Host "== cypha_bench_run --domain-tag d41 (skipped - profile not present) ==" -ForegroundColor DarkGray
        Step-Result "math_integration_scale_d41" $true "skipped (d41 profile not present)"
    }
}

# --- d42 math integration production (optional) ---
if ($env:CYPHA_VALIDATE_MATH_INTEGRATION_PRODUCTION -eq "1") {
    Write-Host ""
    if (Test-DomainTagExists -Tag "d42") {
        Write-Host "== cypha_bench_run --domain-tag d42 (CYPHA_VALIDATE_MATH_INTEGRATION_PRODUCTION) ==" -ForegroundColor Yellow
        $benchExe = BinPath "cypha_bench_run"
        if (-not (Test-Path $benchExe)) {
            Step-Result "math_integration_production_d42" $false "missing $benchExe"
        } else {
            Push-Location $root
            try {
                & $benchExe --domain-tag d42
                $d42Code = $LASTEXITCODE
            } finally {
                Pop-Location
            }
            Step-Result "math_integration_production_d42" ($d42Code -eq 0) $(if ($d42Code -eq 0) { "d42 ok" } else { "exit $d42Code" })
        }
    } else {
        Write-Host "== cypha_bench_run --domain-tag d42 (skipped - profile not present) ==" -ForegroundColor DarkGray
        Step-Result "math_integration_production_d42" $true "skipped (d42 profile not present)"
    }
}

# --- d43 math integration lock (optional) ---
if ($env:CYPHA_VALIDATE_MATH_INTEGRATION_LOCK -eq "1") {
    Write-Host ""
    if (Test-DomainTagExists -Tag "d43") {
        Write-Host "== cypha_bench_run --domain-tag d43 (CYPHA_VALIDATE_MATH_INTEGRATION_LOCK) ==" -ForegroundColor Yellow
        $benchExe = BinPath "cypha_bench_run"
        if (-not (Test-Path $benchExe)) {
            Step-Result "math_integration_lock_d43" $false "missing $benchExe"
        } else {
            Push-Location $root
            try {
                & $benchExe --domain-tag d43
                $d43Code = $LASTEXITCODE
            } finally {
                Pop-Location
            }
            Step-Result "math_integration_lock_d43" ($d43Code -eq 0) $(if ($d43Code -eq 0) { "d43 ok" } else { "exit $d43Code" })
        }
    } else {
        Write-Host "== cypha_bench_run --domain-tag d43 (skipped - profile not present) ==" -ForegroundColor DarkGray
        Step-Result "math_integration_lock_d43" $true "skipped (d43 profile not present)"
    }
}

# --- d44 Nyström kernel CyphaLM (optional) ---
if ($env:CYPHA_VALIDATE_KERNEL_NYSTROM_CYPHALM -eq "1") {
    Write-Host ""
    if (Test-DomainTagExists -Tag "d44") {
        Write-Host "== cypha_bench_run --domain-tag d44 (CYPHA_VALIDATE_KERNEL_NYSTROM_CYPHALM) ==" -ForegroundColor Yellow
        $benchExe = BinPath "cypha_bench_run"
        if (-not (Test-Path $benchExe)) {
            Step-Result "kernel_nystrom_cyphalm_d44" $false "missing $benchExe"
        } else {
            Push-Location $root
            try {
                & $benchExe --domain-tag d44
                $d44Code = $LASTEXITCODE
            } finally {
                Pop-Location
            }
            Step-Result "kernel_nystrom_cyphalm_d44" ($d44Code -eq 0) $(if ($d44Code -eq 0) { "d44 ok" } else { "exit $d44Code" })
        }
    } else {
        Write-Host "== cypha_bench_run --domain-tag d44 (skipped - profile not present) ==" -ForegroundColor DarkGray
        Step-Result "kernel_nystrom_cyphalm_d44" $true "skipped (d44 profile not present)"
    }
}

# --- d45 per-stat navigation λ (optional) ---
if ($env:CYPHA_VALIDATE_PER_STAT_NAVIGATION -eq "1") {
    Write-Host ""
    if (Test-DomainTagExists -Tag "d45") {
        Write-Host "== cypha_bench_run --domain-tag d45 (CYPHA_VALIDATE_PER_STAT_NAVIGATION) ==" -ForegroundColor Yellow
        $benchExe = BinPath "cypha_bench_run"
        if (-not (Test-Path $benchExe)) {
            Step-Result "per_stat_navigation_d45" $false "missing $benchExe"
        } else {
            Push-Location $root
            try {
                & $benchExe --domain-tag d45
                $d45Code = $LASTEXITCODE
            } finally {
                Pop-Location
            }
            Step-Result "per_stat_navigation_d45" ($d45Code -eq 0) $(if ($d45Code -eq 0) { "d45 ok" } else { "exit $d45Code" })
        }
    } else {
        Write-Host "== cypha_bench_run --domain-tag d45 (skipped - profile not present) ==" -ForegroundColor DarkGray
        Step-Result "per_stat_navigation_d45" $true "skipped (d45 profile not present)"
    }
}

# --- d46 math stack upgrade (optional) ---
if ($env:CYPHA_VALIDATE_MATH_STACK_UPGRADE -eq "1") {
    Write-Host ""
    if (Test-DomainTagExists -Tag "d46") {
        Write-Host "== cypha_bench_run --domain-tag d46 (CYPHA_VALIDATE_MATH_STACK_UPGRADE) ==" -ForegroundColor Yellow
        $benchExe = BinPath "cypha_bench_run"
        if (-not (Test-Path $benchExe)) {
            Step-Result "math_stack_upgrade_d46" $false "missing $benchExe"
        } else {
            Push-Location $root
            try {
                & $benchExe --domain-tag d46
                $d46Code = $LASTEXITCODE
            } finally {
                Pop-Location
            }
            Step-Result "math_stack_upgrade_d46" ($d46Code -eq 0) $(if ($d46Code -eq 0) { "d46 ok" } else { "exit $d46Code" })
        }
    } else {
        Write-Host "== cypha_bench_run --domain-tag d46 (skipped - profile not present) ==" -ForegroundColor DarkGray
        Step-Result "math_stack_upgrade_d46" $true "skipped (d46 profile not present)"
    }
}

# --- d47 span ablation (optional) ---
if ($env:CYPHA_VALIDATE_SPAN_ABLATION -eq "1") {
    Write-Host ""
    if (Test-DomainTagExists -Tag "d47") {
        Write-Host "== cypha_bench_run --domain-tag d47 (CYPHA_VALIDATE_SPAN_ABLATION) ==" -ForegroundColor Yellow
        $benchExe = BinPath "cypha_bench_run"
        if (-not (Test-Path $benchExe)) {
            Step-Result "span_ablation_d47" $false "missing $benchExe"
        } else {
            Push-Location $root
            try {
                & $benchExe --domain-tag d47
                $d47Code = $LASTEXITCODE
            } finally {
                Pop-Location
            }
            Step-Result "span_ablation_d47" ($d47Code -eq 0) $(if ($d47Code -eq 0) { "d47 ok" } else { "exit $d47Code" })
        }
    } else {
        Write-Host "== cypha_bench_run --domain-tag d47 (skipped - profile not present) ==" -ForegroundColor DarkGray
        Step-Result "span_ablation_d47" $true "skipped (d47 profile not present)"
    }
}

# --- d48 kappa ceiling ablation (optional) ---
if ($env:CYPHA_VALIDATE_KAPPA_CEILING_ABLATION -eq "1") {
    Write-Host ""
    if (Test-DomainTagExists -Tag "d48") {
        Write-Host "== cypha_bench_run --domain-tag d48 (CYPHA_VALIDATE_KAPPA_CEILING_ABLATION) ==" -ForegroundColor Yellow
        $benchExe = BinPath "cypha_bench_run"
        if (-not (Test-Path $benchExe)) {
            Step-Result "kappa_ceiling_ablation_d48" $false "missing $benchExe"
        } else {
            Push-Location $root
            try {
                & $benchExe --domain-tag d48
                $d48Code = $LASTEXITCODE
            } finally {
                Pop-Location
            }
            Step-Result "kappa_ceiling_ablation_d48" ($d48Code -eq 0) $(if ($d48Code -eq 0) { "d48 ok" } else { "exit $d48Code" })
        }
    } else {
        Write-Host "== cypha_bench_run --domain-tag d48 (skipped - profile not present) ==" -ForegroundColor DarkGray
        Step-Result "kappa_ceiling_ablation_d48" $true "skipped (d48 profile not present)"
    }
}

# --- d49 ceiling grid joint (optional) ---
if ($env:CYPHA_VALIDATE_CEILING_GRID_JOINT -eq "1") {
    Write-Host ""
    if (Test-DomainTagExists -Tag "d49") {
        Write-Host "== cypha_bench_run --domain-tag d49 (CYPHA_VALIDATE_CEILING_GRID_JOINT) ==" -ForegroundColor Yellow
        $benchExe = BinPath "cypha_bench_run"
        if (-not (Test-Path $benchExe)) {
            Step-Result "ceiling_grid_joint_d49" $false "missing $benchExe"
        } else {
            Push-Location $root
            try {
                & $benchExe --domain-tag d49
                $d49Code = $LASTEXITCODE
            } finally {
                Pop-Location
            }
            Step-Result "ceiling_grid_joint_d49" ($d49Code -eq 0) $(if ($d49Code -eq 0) { "d49 ok" } else { "exit $d49Code" })
        }
    } else {
        Write-Host "== cypha_bench_run --domain-tag d49 (skipped - profile not present) ==" -ForegroundColor DarkGray
        Step-Result "ceiling_grid_joint_d49" $true "skipped (d49 profile not present)"
    }
}

# --- d50 math joint lock (optional) ---
if ($env:CYPHA_VALIDATE_MATH_JOINT_LOCK -eq "1") {
    Write-Host ""
    if (Test-DomainTagExists -Tag "d50") {
        Write-Host "== cypha_bench_run --domain-tag d50 (CYPHA_VALIDATE_MATH_JOINT_LOCK) ==" -ForegroundColor Yellow
        $benchExe = BinPath "cypha_bench_run"
        if (-not (Test-Path $benchExe)) {
            Step-Result "math_joint_lock_d50" $false "missing $benchExe"
        } else {
            Push-Location $root
            try {
                & $benchExe --domain-tag d50
                $d50Code = $LASTEXITCODE
            } finally {
                Pop-Location
            }
            Step-Result "math_joint_lock_d50" ($d50Code -eq 0) $(if ($d50Code -eq 0) { "d50 ok" } else { "exit $d50Code" })
        }
    } else {
        Write-Host "== cypha_bench_run --domain-tag d50 (skipped - profile not present) ==" -ForegroundColor DarkGray
        Step-Result "math_joint_lock_d50" $true "skipped (d50 profile not present)"
    }
}

# --- d51 opt-in lever joint (optional) ---
if ($env:CYPHA_VALIDATE_OPT_IN_LEVER_JOINT -eq "1") {
    Write-Host ""
    if (Test-DomainTagExists -Tag "d51") {
        Write-Host "== cypha_bench_run --domain-tag d51 (CYPHA_VALIDATE_OPT_IN_LEVER_JOINT) ==" -ForegroundColor Yellow
        $benchExe = BinPath "cypha_bench_run"
        if (-not (Test-Path $benchExe)) {
            Step-Result "opt_in_lever_joint_d51" $false "missing $benchExe"
        } else {
            Push-Location $root
            try {
                & $benchExe --domain-tag d51
                $d51Code = $LASTEXITCODE
            } finally {
                Pop-Location
            }
            Step-Result "opt_in_lever_joint_d51" ($d51Code -eq 0) $(if ($d51Code -eq 0) { "d51 ok" } else { "exit $d51Code" })
        }
    } else {
        Write-Host "== cypha_bench_run --domain-tag d51 (skipped - profile not present) ==" -ForegroundColor DarkGray
        Step-Result "opt_in_lever_joint_d51" $true "skipped (d51 profile not present)"
    }
}

# --- d52 preset ship lock (optional) ---
if ($env:CYPHA_VALIDATE_PRESET_SHIP_LOCK -eq "1") {
    Write-Host ""
    if (Test-DomainTagExists -Tag "d52") {
        Write-Host "== cypha_bench_run --domain-tag d52 (CYPHA_VALIDATE_PRESET_SHIP_LOCK) ==" -ForegroundColor Yellow
        $benchExe = BinPath "cypha_bench_run"
        if (-not (Test-Path $benchExe)) {
            Step-Result "preset_ship_lock_d52" $false "missing $benchExe"
        } else {
            Push-Location $root
            try {
                & $benchExe --domain-tag d52
                $d52Code = $LASTEXITCODE
            } finally {
                Pop-Location
            }
            Step-Result "preset_ship_lock_d52" ($d52Code -eq 0) $(if ($d52Code -eq 0) { "d52 ok" } else { "exit $d52Code" })
        }
    } else {
        Write-Host "== cypha_bench_run --domain-tag d52 (skipped - profile not present) ==" -ForegroundColor DarkGray
        Step-Result "preset_ship_lock_d52" $true "skipped (d52 profile not present)"
    }
}

# --- d53 production preset ship lock (optional) ---
if ($env:CYPHA_VALIDATE_PRODUCTION_PRESET_SHIP_LOCK -eq "1") {
    Write-Host ""
    if (Test-DomainTagExists -Tag "d53") {
        Write-Host "== cypha_bench_run --domain-tag d53 (CYPHA_VALIDATE_PRODUCTION_PRESET_SHIP_LOCK) ==" -ForegroundColor Yellow
        $benchExe = BinPath "cypha_bench_run"
        if (-not (Test-Path $benchExe)) {
            Step-Result "production_preset_ship_lock_d53" $false "missing $benchExe"
        } else {
            Push-Location $root
            try {
                & $benchExe --domain-tag d53
                $d53Code = $LASTEXITCODE
            } finally {
                Pop-Location
            }
            Step-Result "production_preset_ship_lock_d53" ($d53Code -eq 0) $(if ($d53Code -eq 0) { "d53 ok" } else { "exit $d53Code" })
        }
    } else {
        Write-Host "== cypha_bench_run --domain-tag d53 (skipped - profile not present) ==" -ForegroundColor DarkGray
        Step-Result "production_preset_ship_lock_d53" $true "skipped (d53 profile not present)"
    }
}

# --- d54 production math certificate (optional) ---
if ($env:CYPHA_VALIDATE_PRODUCTION_MATH_CERTIFICATE -eq "1") {
    Write-Host ""
    if (Test-DomainTagExists -Tag "d54") {
        Write-Host "== cypha_bench_run --domain-tag d54 (CYPHA_VALIDATE_PRODUCTION_MATH_CERTIFICATE) ==" -ForegroundColor Yellow
        $benchExe = BinPath "cypha_bench_run"
        if (-not (Test-Path $benchExe)) {
            Step-Result "production_math_certificate_d54" $false "missing $benchExe"
        } else {
            Push-Location $root
            try {
                & $benchExe --domain-tag d54
                $d54Code = $LASTEXITCODE
            } finally {
                Pop-Location
            }
            Step-Result "production_math_certificate_d54" ($d54Code -eq 0) $(if ($d54Code -eq 0) { "d54 ok" } else { "exit $d54Code" })
        }
    } else {
        Write-Host "== cypha_bench_run --domain-tag d54 (skipped - profile not present) ==" -ForegroundColor DarkGray
        Step-Result "production_math_certificate_d54" $true "skipped (d54 profile not present)"
    }
}

# --- d55 nav warmup grid joint (optional) ---
if ($env:CYPHA_VALIDATE_NAV_WARMUP_GRID_JOINT -eq "1") {
    Write-Host ""
    if (Test-DomainTagExists -Tag "d55") {
        Write-Host "== cypha_bench_run --domain-tag d55 (CYPHA_VALIDATE_NAV_WARMUP_GRID_JOINT) ==" -ForegroundColor Yellow
        $benchExe = BinPath "cypha_bench_run"
        if (-not (Test-Path $benchExe)) {
            Step-Result "nav_warmup_grid_joint_d55" $false "missing $benchExe"
        } else {
            Push-Location $root
            try {
                & $benchExe --domain-tag d55
                $d55Code = $LASTEXITCODE
            } finally {
                Pop-Location
            }
            Step-Result "nav_warmup_grid_joint_d55" ($d55Code -eq 0) $(if ($d55Code -eq 0) { "d55 ok" } else { "exit $d55Code" })
        }
    } else {
        Write-Host "== cypha_bench_run --domain-tag d55 (skipped - profile not present) ==" -ForegroundColor DarkGray
        Step-Result "nav_warmup_grid_joint_d55" $true "skipped (d55 profile not present)"
    }
}

# --- d56 cell sweep math integration (optional) ---
if ($env:CYPHA_VALIDATE_CELL_SWEEP_MATH_INTEGRATION -eq "1") {
    Write-Host ""
    if (Test-DomainTagExists -Tag "d56") {
        Write-Host "== cypha_bench_run --domain-tag d56 (CYPHA_VALIDATE_CELL_SWEEP_MATH_INTEGRATION) ==" -ForegroundColor Yellow
        $benchExe = BinPath "cypha_bench_run"
        if (-not (Test-Path $benchExe)) {
            Step-Result "cell_sweep_math_integration_d56" $false "missing $benchExe"
        } else {
            Push-Location $root
            try {
                & $benchExe --domain-tag d56
                $d56Code = $LASTEXITCODE
            } finally {
                Pop-Location
            }
            Step-Result "cell_sweep_math_integration_d56" ($d56Code -eq 0) $(if ($d56Code -eq 0) { "d56 ok" } else { "exit $d56Code" })
        }
    } else {
        Write-Host "== cypha_bench_run --domain-tag d56 (skipped - profile not present) ==" -ForegroundColor DarkGray
        Step-Result "cell_sweep_math_integration_d56" $true "skipped (d56 profile not present)"
    }
}

# --- d57 production cell sweep math certificate (optional) ---
if ($env:CYPHA_VALIDATE_PRODUCTION_CELL_SWEEP_MATH_CERTIFICATE -eq "1") {
    Write-Host ""
    if (Test-DomainTagExists -Tag "d57") {
        Write-Host "== cypha_bench_run --domain-tag d57 (CYPHA_VALIDATE_PRODUCTION_CELL_SWEEP_MATH_CERTIFICATE) ==" -ForegroundColor Yellow
        $benchExe = BinPath "cypha_bench_run"
        if (-not (Test-Path $benchExe)) {
            Step-Result "production_cell_sweep_math_certificate_d57" $false "missing $benchExe"
        } else {
            Push-Location $root
            try {
                & $benchExe --domain-tag d57
                $d57Code = $LASTEXITCODE
            } finally {
                Pop-Location
            }
            Step-Result "production_cell_sweep_math_certificate_d57" ($d57Code -eq 0) $(if ($d57Code -eq 0) { "d57 ok" } else { "exit $d57Code" })
        }
    } else {
        Write-Host "== cypha_bench_run --domain-tag d57 (skipped - profile not present) ==" -ForegroundColor DarkGray
        Step-Result "production_cell_sweep_math_certificate_d57" $true "skipped (d57 profile not present)"
    }
}

# --- d58 production overnight math complete (optional) ---
if ($env:CYPHA_VALIDATE_PRODUCTION_OVERNIGHT_MATH_COMPLETE -eq "1") {
    Write-Host ""
    if (Test-DomainTagExists -Tag "d58") {
        Write-Host "== cypha_bench_run --domain-tag d58 (CYPHA_VALIDATE_PRODUCTION_OVERNIGHT_MATH_COMPLETE) ==" -ForegroundColor Yellow
        $benchExe = BinPath "cypha_bench_run"
        if (-not (Test-Path $benchExe)) {
            Step-Result "production_overnight_math_complete_d58" $false "missing $benchExe"
        } else {
            Push-Location $root
            try {
                & $benchExe --domain-tag d58
                $d58Code = $LASTEXITCODE
            } finally {
                Pop-Location
            }
            Step-Result "production_overnight_math_complete_d58" ($d58Code -eq 0) $(if ($d58Code -eq 0) { "d58 ok" } else { "exit $d58Code" })
        }
    } else {
        Write-Host "== cypha_bench_run --domain-tag d58 (skipped - profile not present) ==" -ForegroundColor DarkGray
        Step-Result "production_overnight_math_complete_d58" $true "skipped (d58 profile not present)"
    }
}

# --- d59 kernel blend floor grid joint (optional) ---
if ($env:CYPHA_VALIDATE_KERNEL_BLEND_FLOOR_GRID_JOINT -eq "1") {
    Write-Host ""
    if (Test-DomainTagExists -Tag "d59") {
        Write-Host "== cypha_bench_run --domain-tag d59 (CYPHA_VALIDATE_KERNEL_BLEND_FLOOR_GRID_JOINT) ==" -ForegroundColor Yellow
        $benchExe = BinPath "cypha_bench_run"
        if (-not (Test-Path $benchExe)) {
            Step-Result "kernel_blend_floor_grid_joint_d59" $false "missing $benchExe"
        } else {
            Push-Location $root
            try {
                & $benchExe --domain-tag d59
                $d59Code = $LASTEXITCODE
            } finally {
                Pop-Location
            }
            Step-Result "kernel_blend_floor_grid_joint_d59" ($d59Code -eq 0) $(if ($d59Code -eq 0) { "d59 ok" } else { "exit $d59Code" })
        }
    } else {
        Write-Host "== cypha_bench_run --domain-tag d59 (skipped - profile not present) ==" -ForegroundColor DarkGray
        Step-Result "kernel_blend_floor_grid_joint_d59" $true "skipped (d59 profile not present)"
    }
}

# --- d60 excess grad margin grid joint (optional) ---
if ($env:CYPHA_VALIDATE_EXCESS_GRAD_MARGIN_GRID_JOINT -eq "1") {
    Write-Host ""
    if (Test-DomainTagExists -Tag "d60") {
        Write-Host "== cypha_bench_run --domain-tag d60 (CYPHA_VALIDATE_EXCESS_GRAD_MARGIN_GRID_JOINT) ==" -ForegroundColor Yellow
        $benchExe = BinPath "cypha_bench_run"
        if (-not (Test-Path $benchExe)) {
            Step-Result "excess_grad_margin_grid_joint_d60" $false "missing $benchExe"
        } else {
            Push-Location $root
            try {
                & $benchExe --domain-tag d60
                $d60Code = $LASTEXITCODE
            } finally {
                Pop-Location
            }
            Step-Result "excess_grad_margin_grid_joint_d60" ($d60Code -eq 0) $(if ($d60Code -eq 0) { "d60 ok" } else { "exit $d60Code" })
        }
    } else {
        Write-Host "== cypha_bench_run --domain-tag d60 (skipped - profile not present) ==" -ForegroundColor DarkGray
        Step-Result "excess_grad_margin_grid_joint_d60" $true "skipped (d60 profile not present)"
    }
}

# --- d61 excess grad scale grid joint (optional) ---
if ($env:CYPHA_VALIDATE_EXCESS_GRAD_SCALE_GRID_JOINT -eq "1") {
    Write-Host ""
    if (Test-DomainTagExists -Tag "d61") {
        Write-Host "== cypha_bench_run --domain-tag d61 (CYPHA_VALIDATE_EXCESS_GRAD_SCALE_GRID_JOINT) ==" -ForegroundColor Yellow
        $benchExe = BinPath "cypha_bench_run"
        if (-not (Test-Path $benchExe)) {
            Step-Result "excess_grad_scale_grid_joint_d61" $false "missing $benchExe"
        } else {
            Push-Location $root
            try {
                & $benchExe --domain-tag d61
                $d61Code = $LASTEXITCODE
            } finally {
                Pop-Location
            }
            Step-Result "excess_grad_scale_grid_joint_d61" ($d61Code -eq 0) $(if ($d61Code -eq 0) { "d61 ok" } else { "exit $d61Code" })
        }
    } else {
        Write-Host "== cypha_bench_run --domain-tag d61 (skipped - profile not present) ==" -ForegroundColor DarkGray
        Step-Result "excess_grad_scale_grid_joint_d61" $true "skipped (d61 profile not present)"
    }
}

# --- d62 math ablation stack complete (optional) ---
if ($env:CYPHA_VALIDATE_MATH_ABLATION_STACK_COMPLETE -eq "1") {
    Write-Host ""
    if (Test-DomainTagExists -Tag "d62") {
        Write-Host "== cypha_bench_run --domain-tag d62 (CYPHA_VALIDATE_MATH_ABLATION_STACK_COMPLETE) ==" -ForegroundColor Yellow
        $benchExe = BinPath "cypha_bench_run"
        if (-not (Test-Path $benchExe)) {
            Step-Result "math_ablation_stack_complete_d62" $false "missing $benchExe"
        } else {
            Push-Location $root
            try {
                & $benchExe --domain-tag d62
                $d62Code = $LASTEXITCODE
            } finally {
                Pop-Location
            }
            Step-Result "math_ablation_stack_complete_d62" ($d62Code -eq 0) $(if ($d62Code -eq 0) { "d62 ok" } else { "exit $d62Code" })
        }
    } else {
        Write-Host "== cypha_bench_run --domain-tag d62 (skipped - profile not present) ==" -ForegroundColor DarkGray
        Step-Result "math_ablation_stack_complete_d62" $true "skipped (d62 profile not present)"
    }
}

# --- d63 reu forget blend grid joint (optional) ---
if ($env:CYPHA_VALIDATE_REU_FORGET_BLEND_GRID_JOINT -eq "1") {
    Write-Host ""
    if (Test-DomainTagExists -Tag "d63") {
        Write-Host "== cypha_bench_run --domain-tag d63 (CYPHA_VALIDATE_REU_FORGET_BLEND_GRID_JOINT) ==" -ForegroundColor Yellow
        $benchExe = BinPath "cypha_bench_run"
        if (-not (Test-Path $benchExe)) {
            Step-Result "reu_forget_blend_grid_joint_d63" $false "missing $benchExe"
        } else {
            Push-Location $root
            try {
                & $benchExe --domain-tag d63
                $d63Code = $LASTEXITCODE
            } finally {
                Pop-Location
            }
            Step-Result "reu_forget_blend_grid_joint_d63" ($d63Code -eq 0) $(if ($d63Code -eq 0) { "d63 ok" } else { "exit $d63Code" })
        }
    } else {
        Write-Host "== cypha_bench_run --domain-tag d63 (skipped - profile not present) ==" -ForegroundColor DarkGray
        Step-Result "reu_forget_blend_grid_joint_d63" $true "skipped (d63 profile not present)"
    }
}

# --- d64 kappa trajectory window grid joint (optional) ---
if ($env:CYPHA_VALIDATE_KAPPA_TRAJECTORY_WINDOW_GRID_JOINT -eq "1") {
    Write-Host ""
    if (Test-DomainTagExists -Tag "d64") {
        Write-Host "== cypha_bench_run --domain-tag d64 (CYPHA_VALIDATE_KAPPA_TRAJECTORY_WINDOW_GRID_JOINT) ==" -ForegroundColor Yellow
        $benchExe = BinPath "cypha_bench_run"
        if (-not (Test-Path $benchExe)) {
            Step-Result "kappa_trajectory_window_grid_joint_d64" $false "missing $benchExe"
        } else {
            Push-Location $root
            try {
                & $benchExe --domain-tag d64
                $d64Code = $LASTEXITCODE
            } finally {
                Pop-Location
            }
            Step-Result "kappa_trajectory_window_grid_joint_d64" ($d64Code -eq 0) $(if ($d64Code -eq 0) { "d64 ok" } else { "exit $d64Code" })
        }
    } else {
        Write-Host "== cypha_bench_run --domain-tag d64 (skipped - profile not present) ==" -ForegroundColor DarkGray
        Step-Result "kappa_trajectory_window_grid_joint_d64" $true "skipped (d64 profile not present)"
    }
}

# --- d65 navigation loss warmup grid joint (optional) ---
if ($env:CYPHA_VALIDATE_NAVIGATION_LOSS_WARMUP_GRID_JOINT -eq "1") {
    Write-Host ""
    if (Test-DomainTagExists -Tag "d65") {
        Write-Host "== cypha_bench_run --domain-tag d65 (CYPHA_VALIDATE_NAVIGATION_LOSS_WARMUP_GRID_JOINT) ==" -ForegroundColor Yellow
        $benchExe = BinPath "cypha_bench_run"
        if (-not (Test-Path $benchExe)) {
            Step-Result "navigation_loss_warmup_grid_joint_d65" $false "missing $benchExe"
        } else {
            Push-Location $root
            try {
                & $benchExe --domain-tag d65
                $d65Code = $LASTEXITCODE
            } finally {
                Pop-Location
            }
            Step-Result "navigation_loss_warmup_grid_joint_d65" ($d65Code -eq 0) $(if ($d65Code -eq 0) { "d65 ok" } else { "exit $d65Code" })
        }
    } else {
        Write-Host "== cypha_bench_run --domain-tag d65 (skipped - profile not present) ==" -ForegroundColor DarkGray
        Step-Result "navigation_loss_warmup_grid_joint_d65" $true "skipped (d65 profile not present)"
    }
}

# --- d66 free energy beta grid joint (optional) ---
if ($env:CYPHA_VALIDATE_FREE_ENERGY_BETA_GRID_JOINT -eq "1") {
    Write-Host ""
    if (Test-DomainTagExists -Tag "d66") {
        Write-Host "== cypha_bench_run --domain-tag d66 (CYPHA_VALIDATE_FREE_ENERGY_BETA_GRID_JOINT) ==" -ForegroundColor Yellow
        $benchExe = BinPath "cypha_bench_run"
        if (-not (Test-Path $benchExe)) {
            Step-Result "free_energy_beta_grid_joint_d66" $false "missing $benchExe"
        } else {
            Push-Location $root
            try {
                & $benchExe --domain-tag d66
                $d66Code = $LASTEXITCODE
            } finally {
                Pop-Location
            }
            Step-Result "free_energy_beta_grid_joint_d66" ($d66Code -eq 0) $(if ($d66Code -eq 0) { "d66 ok" } else { "exit $d66Code" })
        }
    } else {
        Write-Host "== cypha_bench_run --domain-tag d66 (skipped - profile not present) ==" -ForegroundColor DarkGray
        Step-Result "free_energy_beta_grid_joint_d66" $true "skipped (d66 profile not present)"
    }
}

# --- d67 kernel blend grid joint (optional) ---
if ($env:CYPHA_VALIDATE_KERNEL_BLEND_GRID_JOINT -eq "1") {
    Write-Host ""
    if (Test-DomainTagExists -Tag "d67") {
        Write-Host "== cypha_bench_run --domain-tag d67 (CYPHA_VALIDATE_KERNEL_BLEND_GRID_JOINT) ==" -ForegroundColor Yellow
        $benchExe = BinPath "cypha_bench_run"
        if (-not (Test-Path $benchExe)) {
            Step-Result "kernel_blend_grid_joint_d67" $false "missing $benchExe"
        } else {
            Push-Location $root
            try {
                & $benchExe --domain-tag d67
                $d67Code = $LASTEXITCODE
            } finally {
                Pop-Location
            }
            Step-Result "kernel_blend_grid_joint_d67" ($d67Code -eq 0) $(if ($d67Code -eq 0) { "d67 ok" } else { "exit $d67Code" })
        }
    } else {
        Write-Host "== cypha_bench_run --domain-tag d67 (skipped - profile not present) ==" -ForegroundColor DarkGray
        Step-Result "kernel_blend_grid_joint_d67" $true "skipped (d67 profile not present)"
    }
}

# --- d68 kernel m grid joint (optional) ---
if ($env:CYPHA_VALIDATE_KERNEL_M_GRID_JOINT -eq "1") {
    Write-Host ""
    if (Test-DomainTagExists -Tag "d68") {
        Write-Host "== cypha_bench_run --domain-tag d68 (CYPHA_VALIDATE_KERNEL_M_GRID_JOINT) ==" -ForegroundColor Yellow
        $benchExe = BinPath "cypha_bench_run"
        if (-not (Test-Path $benchExe)) {
            Step-Result "kernel_m_grid_joint_d68" $false "missing $benchExe"
        } else {
            Push-Location $root
            try {
                & $benchExe --domain-tag d68
                $d68Code = $LASTEXITCODE
            } finally {
                Pop-Location
            }
            Step-Result "kernel_m_grid_joint_d68" ($d68Code -eq 0) $(if ($d68Code -eq 0) { "d68 ok" } else { "exit $d68Code" })
        }
    } else {
        Write-Host "== cypha_bench_run --domain-tag d68 (skipped - profile not present) ==" -ForegroundColor DarkGray
        Step-Result "kernel_m_grid_joint_d68" $true "skipped (d68 profile not present)"
    }
}

# --- d69 hybrid blend logit grid joint (optional) ---
if ($env:CYPHA_VALIDATE_HYBRID_BLEND_LOGIT_GRID_JOINT -eq "1") {
    Write-Host ""
    if (Test-DomainTagExists -Tag "d69") {
        Write-Host "== cypha_bench_run --domain-tag d69 (CYPHA_VALIDATE_HYBRID_BLEND_LOGIT_GRID_JOINT) ==" -ForegroundColor Yellow
        $benchExe = BinPath "cypha_bench_run"
        if (-not (Test-Path $benchExe)) {
            Step-Result "hybrid_blend_logit_grid_joint_d69" $false "missing $benchExe"
        } else {
            Push-Location $root
            try {
                & $benchExe --domain-tag d69
                $d69Code = $LASTEXITCODE
            } finally {
                Pop-Location
            }
            Step-Result "hybrid_blend_logit_grid_joint_d69" ($d69Code -eq 0) $(if ($d69Code -eq 0) { "d69 ok" } else { "exit $d69Code" })
        }
    } else {
        Write-Host "== cypha_bench_run --domain-tag d69 (skipped - profile not present) ==" -ForegroundColor DarkGray
        Step-Result "hybrid_blend_logit_grid_joint_d69" $true "skipped (d69 profile not present)"
    }
}

# --- d70 mdl forget max norm grid joint (optional) ---
if ($env:CYPHA_VALIDATE_MDL_FORGET_MAX_NORM_GRID_JOINT -eq "1") {
    Write-Host ""
    if (Test-DomainTagExists -Tag "d70") {
        Write-Host "== cypha_bench_run --domain-tag d70 (CYPHA_VALIDATE_MDL_FORGET_MAX_NORM_GRID_JOINT) ==" -ForegroundColor Yellow
        $benchExe = BinPath "cypha_bench_run"
        if (-not (Test-Path $benchExe)) {
            Step-Result "mdl_forget_max_norm_grid_joint_d70" $false "missing $benchExe"
        } else {
            Push-Location $root
            try {
                & $benchExe --domain-tag d70
                $d70Code = $LASTEXITCODE
            } finally {
                Pop-Location
            }
            Step-Result "mdl_forget_max_norm_grid_joint_d70" ($d70Code -eq 0) $(if ($d70Code -eq 0) { "d70 ok" } else { "exit $d70Code" })
        }
    } else {
        Write-Host "== cypha_bench_run --domain-tag d70 (skipped - profile not present) ==" -ForegroundColor DarkGray
        Step-Result "mdl_forget_max_norm_grid_joint_d70" $true "skipped (d70 profile not present)"
    }
}

# --- d71 kernel lr scale grid joint (optional) ---
if ($env:CYPHA_VALIDATE_KERNEL_LR_SCALE_GRID_JOINT -eq "1") {
    Write-Host ""
    if (Test-DomainTagExists -Tag "d71") {
        Write-Host "== cypha_bench_run --domain-tag d71 (CYPHA_VALIDATE_KERNEL_LR_SCALE_GRID_JOINT) ==" -ForegroundColor Yellow
        $benchExe = BinPath "cypha_bench_run"
        if (-not (Test-Path $benchExe)) {
            Step-Result "kernel_lr_scale_grid_joint_d71" $false "missing $benchExe"
        } else {
            Push-Location $root
            try {
                & $benchExe --domain-tag d71
                $d71Code = $LASTEXITCODE
            } finally {
                Pop-Location
            }
            Step-Result "kernel_lr_scale_grid_joint_d71" ($d71Code -eq 0) $(if ($d71Code -eq 0) { "d71 ok" } else { "exit $d71Code" })
        }
    } else {
        Write-Host "== cypha_bench_run --domain-tag d71 (skipped - profile not present) ==" -ForegroundColor DarkGray
        Step-Result "kernel_lr_scale_grid_joint_d71" $true "skipped (d71 profile not present)"
    }
}

# --- d72 alpha init grid joint (optional) ---
if ($env:CYPHA_VALIDATE_ALPHA_INIT_GRID_JOINT -eq "1") {
    Write-Host ""
    if (Test-DomainTagExists -Tag "d72") {
        Write-Host "== cypha_bench_run --domain-tag d72 (CYPHA_VALIDATE_ALPHA_INIT_GRID_JOINT) ==" -ForegroundColor Yellow
        $benchExe = BinPath "cypha_bench_run"
        if (-not (Test-Path $benchExe)) {
            Step-Result "alpha_init_grid_joint_d72" $false "missing $benchExe"
        } else {
            Push-Location $root
            try {
                & $benchExe --domain-tag d72
                $d72Code = $LASTEXITCODE
            } finally {
                Pop-Location
            }
            Step-Result "alpha_init_grid_joint_d72" ($d72Code -eq 0) $(if ($d72Code -eq 0) { "d72 ok" } else { "exit $d72Code" })
        }
    } else {
        Write-Host "== cypha_bench_run --domain-tag d72 (skipped - profile not present) ==" -ForegroundColor DarkGray
        Step-Result "alpha_init_grid_joint_d72" $true "skipped (d72 profile not present)"
    }
}

# --- d73 hybrid blend lr grid joint (optional) ---
if ($env:CYPHA_VALIDATE_HYBRID_BLEND_LR_GRID_JOINT -eq "1") {
    Write-Host ""
    if (Test-DomainTagExists -Tag "d73") {
        Write-Host "== cypha_bench_run --domain-tag d73 (CYPHA_VALIDATE_HYBRID_BLEND_LR_GRID_JOINT) ==" -ForegroundColor Yellow
        $benchExe = BinPath "cypha_bench_run"
        if (-not (Test-Path $benchExe)) {
            Step-Result "hybrid_blend_lr_grid_joint_d73" $false "missing $benchExe"
        } else {
            Push-Location $root
            try {
                & $benchExe --domain-tag d73
                $d73Code = $LASTEXITCODE
            } finally {
                Pop-Location
            }
            Step-Result "hybrid_blend_lr_grid_joint_d73" ($d73Code -eq 0) $(if ($d73Code -eq 0) { "d73 ok" } else { "exit $d73Code" })
        }
    } else {
        Write-Host "== cypha_bench_run --domain-tag d73 (skipped - profile not present) ==" -ForegroundColor DarkGray
        Step-Result "hybrid_blend_lr_grid_joint_d73" $true "skipped (d73 profile not present)"
    }
}

# --- d74 n experts grid joint (optional) ---
if ($env:CYPHA_VALIDATE_N_EXPERTS_GRID_JOINT -eq "1") {
    Write-Host ""
    if (Test-DomainTagExists -Tag "d74") {
        Write-Host "== cypha_bench_run --domain-tag d74 (CYPHA_VALIDATE_N_EXPERTS_GRID_JOINT) ==" -ForegroundColor Yellow
        $benchExe = BinPath "cypha_bench_run"
        if (-not (Test-Path $benchExe)) {
            Step-Result "n_experts_grid_joint_d74" $false "missing $benchExe"
        } else {
            Push-Location $root
            try {
                & $benchExe --domain-tag d74
                $d74Code = $LASTEXITCODE
            } finally {
                Pop-Location
            }
            Step-Result "n_experts_grid_joint_d74" ($d74Code -eq 0) $(if ($d74Code -eq 0) { "d74 ok" } else { "exit $d74Code" })
        }
    } else {
        Write-Host "== cypha_bench_run --domain-tag d74 (skipped - profile not present) ==" -ForegroundColor DarkGray
        Step-Result "n_experts_grid_joint_d74" $true "skipped (d74 profile not present)"
    }
}

# --- d75 max memory slots grid joint (optional) ---
if ($env:CYPHA_VALIDATE_MAX_MEMORY_SLOTS_GRID_JOINT -eq "1") {
    Write-Host ""
    if (Test-DomainTagExists -Tag "d75") {
        Write-Host "== cypha_bench_run --domain-tag d75 (CYPHA_VALIDATE_MAX_MEMORY_SLOTS_GRID_JOINT) ==" -ForegroundColor Yellow
        $benchExe = BinPath "cypha_bench_run"
        if (-not (Test-Path $benchExe)) {
            Step-Result "max_memory_slots_grid_joint_d75" $false "missing $benchExe"
        } else {
            Push-Location $root
            try {
                & $benchExe --domain-tag d75
                $d75Code = $LASTEXITCODE
            } finally {
                Pop-Location
            }
            Step-Result "max_memory_slots_grid_joint_d75" ($d75Code -eq 0) $(if ($d75Code -eq 0) { "d75 ok" } else { "exit $d75Code" })
        }
    } else {
        Write-Host "== cypha_bench_run --domain-tag d75 (skipped - profile not present) ==" -ForegroundColor DarkGray
        Step-Result "max_memory_slots_grid_joint_d75" $true "skipped (d75 profile not present)"
    }
}

# --- d76 compress interval grid joint (optional) ---
if ($env:CYPHA_VALIDATE_COMPRESS_INTERVAL_GRID_JOINT -eq "1") {
    Write-Host ""
    if (Test-DomainTagExists -Tag "d76") {
        Write-Host "== cypha_bench_run --domain-tag d76 (CYPHA_VALIDATE_COMPRESS_INTERVAL_GRID_JOINT) ==" -ForegroundColor Yellow
        $benchExe = BinPath "cypha_bench_run"
        if (-not (Test-Path $benchExe)) {
            Step-Result "compress_interval_grid_joint_d76" $false "missing $benchExe"
        } else {
            Push-Location $root
            try {
                & $benchExe --domain-tag d76
                $d76Code = $LASTEXITCODE
            } finally {
                Pop-Location
            }
            Step-Result "compress_interval_grid_joint_d76" ($d76Code -eq 0) $(if ($d76Code -eq 0) { "d76 ok" } else { "exit $d76Code" })
        }
    } else {
        Write-Host "== cypha_bench_run --domain-tag d76 (skipped - profile not present) ==" -ForegroundColor DarkGray
        Step-Result "compress_interval_grid_joint_d76" $true "skipped (d76 profile not present)"
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
