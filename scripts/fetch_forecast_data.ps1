# Materialize conflict-forecasting sample CSVs under bench/data/forecast.
# Run from repo root: .\scripts\fetch_forecast_data.ps1

$ErrorActionPreference = "Stop"
$root = Split-Path $PSScriptRoot -Parent
$dest = Join-Path $root "bench\data\forecast"
New-Item -ItemType Directory -Force -Path $dest | Out-Null

function Copy-IfMissing([string]$srcName, [string]$dstName) {
    $src = Join-Path $dest $srcName
    $dst = Join-Path $dest $dstName
    if ((Test-Path $src) -and -not (Test-Path $dst)) {
        Copy-Item $src $dst -Force
        Write-Host "copied $srcName -> $dstName"
    }
}

function Write-AsciiFile([string]$path, [string]$text) {
    [System.IO.File]::WriteAllText($path, $text, [System.Text.UTF8Encoding]::new($false))
}

Copy-IfMissing "sample_mid.csv" "gml_mid.csv"
Copy-IfMissing "sample_gdelt.csv" "gdelt_events.csv"

$gmlPath = Join-Path $dest "gml_mid.csv"
if (-not (Test-Path $gmlPath)) {
    $rows = @("year,dyad,theater,hostility,prev_hostility,great_power,duration,gdp_ratio,escalated")
    foreach ($y in 2010..2018) {
        $rows += "$y,USA_CHN,TWN,2,1,1,1,1.1,0"
        $rows += "$y,RUS_UKR,UKR,3,2,1,1,0.5,1"
    }
    Write-AsciiFile $gmlPath (($rows -join "`n") + "`n")
    Write-Host "created gml_mid.csv"
}

$gdeltPath = Join-Path $dest "gdelt_events.csv"
if (-not (Test-Path $gdeltPath)) {
    $rows = @("year,month,day,actor1,actor2,cameo_code,goldstein,theater")
    foreach ($m in 1..12) {
        $rows += "2023,$m,10,USA,CHN,120,-2,TWN"
        $rows += "2023,$m,15,RUS,UKR,190,-8,UKR"
        $rows += "2023,$m,20,ISR,IRN,200,-10,MID"
    }
    Write-AsciiFile $gdeltPath (($rows -join "`n") + "`n")
    Write-Host "created gdelt_events.csv"
}

Write-Host ""
Write-Host "Forecast data ready under $dest"
Write-Host "fetch_forecast_data OK"