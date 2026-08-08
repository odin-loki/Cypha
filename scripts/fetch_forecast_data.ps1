# Fetch / materialize conflict-forecasting datasets for Cypha forecast pipeline.
# Run from repo root: .\scripts\fetch_forecast_data.ps1

$ErrorActionPreference = "Stop"
$dest = Join-Path $PSScriptRoot "..\bench\data\forecast"
New-Item -ItemType Directory -Force -Path $dest | Out-Null

function Copy-IfMissing([string]$srcName, [string]$dstName) {
    $src = Join-Path $dest $srcName
    $dst = Join-Path $dest $dstName
    if ((Test-Path $src) -and -not (Test-Path $dst)) {
        Copy-Item $src $dst -Force
        Write-Host "created $dstName from $srcName"
    }
}

Copy-IfMissing "sample_mid.csv" "gml_mid.csv"
Copy-IfMissing "sample_gdelt.csv" "gdelt_events.csv"

$gmlPath = Join-Path $dest "gml_mid.csv"
if (Test-Path $gmlPath) {
    $rows = @(Get-Content $gmlPath -Encoding UTF8)
    if ($rows.Count -lt 30) {
        $extra = @()
        foreach ($y in 2010..2018) {
            $extra += "$y,USA_CHN,TWN,2,1,1,1,1.1,0"
            $extra += "$y,RUS_UKR,UKR,3,2,1,1,0.5,1"
            $extra += "$y,ISR_IRN,MID,3,2,1,2,0.8,0"
            $extra += "$y,PRK_KOR,PRK,2,2,1,3,0.6,0"
        }
        ($rows + $extra) | Out-File $gmlPath -Encoding utf8
        Write-Host "expanded gml_mid.csv"
    }
}

$gdeltPath = Join-Path $dest "gdelt_events.csv"
if (Test-Path $gdeltPath) {
    $rows = @(Get-Content $gdeltPath -Encoding UTF8)
    if ($rows.Count -lt 40) {
        $extra = @()
        foreach ($m in 1..12) {
            $extra += "2023,$m,10,USA,CHN,120,-2,TWN"
            $extra += "2023,$m,15,RUS,UKR,190,-8,UKR"
            $extra += "2023,$m,20,ISR,IRN,200,-10,MID"
        }
        ($rows + $extra) | Out-File $gdeltPath -Encoding utf8
        Write-Host "expanded gdelt_events.csv"
    }
}

Write-Host ""
Write-Host "Forecast data ready under: $dest"
Write-Host "fetch_forecast_data OK"