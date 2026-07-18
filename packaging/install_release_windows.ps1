# Install Cypha native release bundle into %LOCALAPPDATA%\Cypha\<version> (Windows).
param(
    [string]$InstallRoot = ""
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$BundleRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$Version = if (Test-Path (Join-Path $BundleRoot "VERSION")) {
    (Get-Content (Join-Path $BundleRoot "VERSION") -Raw).Trim()
} else { "unknown" }

if ($InstallRoot -eq "") {
    $InstallRoot = Join-Path $env:LOCALAPPDATA "Cypha\cypha-$Version"
}

$BinSrc = Join-Path $BundleRoot "bin"
$BinDest = Join-Path $InstallRoot "bin"
New-Item -ItemType Directory -Force -Path $BinDest | Out-Null

Get-ChildItem -Path $BinSrc -File | ForEach-Object {
    Copy-Item -Path $_.FullName -Destination (Join-Path $BinDest $_.Name) -Force
}

$required = @(
    "cypha_rest.exe",
    "cypha_bench_run.exe",
    "cypha_bench_report.exe",
    "cypha_diagnostics_run.exe",
    "cypha_tune_run.exe"
)
foreach ($name in $required) {
    if (-not (Test-Path (Join-Path $BinDest $name))) {
        throw "Release bundle missing bin\$name (re-package with scripts/package_release_windows.ps1)"
    }
}

$DevSrc = Join-Path $BinSrc "dev"
$DevDest = Join-Path $BinDest "dev"
if (Test-Path $DevSrc) {
    New-Item -ItemType Directory -Force -Path $DevDest | Out-Null
    Copy-Item -Path (Join-Path $DevSrc "*") -Destination $DevDest -Force
}

$ShareSrc = Join-Path $BundleRoot "share"
$ShareDest = Join-Path $InstallRoot "share"
if (Test-Path $ShareSrc) {
    Copy-Item -Path $ShareSrc -Destination $ShareDest -Recurse -Force
}

$userPath = [Environment]::GetEnvironmentVariable("Path", "User")
if ($userPath -notlike "*$BinDest*") {
    [Environment]::SetEnvironmentVariable("Path", "$userPath;$BinDest", "User")
    $env:Path = "$env:Path;$BinDest"
}

Write-Host ""
Write-Host "Cypha $Version installed to:" -ForegroundColor Green
Write-Host "  $InstallRoot"
Write-Host ""
Write-Host "Production binaries are on PATH; dev parity tools (if bundled) live in bin\dev\."
Write-Host ""
Write-Host "Open a new terminal, then try:"
Write-Host "  cypha_rest.exe --listen 127.0.0.1:8099 ``"
Write-Host "    --cypha `"$InstallRoot\share\demo_fixtures\reference.cypha`" ``"
Write-Host "    --f-field-json `"$InstallRoot\share\demo_fixtures\f_field.json`""
Write-Host ""
Write-Host "  cypha_bench_run.exe --list-domains"
Write-Host "  cypha_bench_run.exe --domain 17"
Write-Host "  cypha_bench_run.exe --report-only"
Write-Host "  cypha_bench_report.exe --output .\bench_report"
Write-Host "  cypha_tune_run.exe --config path\to\sweep.json --dry-run"
Write-Host "  cypha_diagnostics_run.exe --fixtures C:\path\to\fixtures"
Write-Host ""
Write-Host "CyphaDIF REST (POST JSON on cypha_rest): /dif/retrieve, /dif/generate"
