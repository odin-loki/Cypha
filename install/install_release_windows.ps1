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

$BinDest = Join-Path $InstallRoot "bin"
New-Item -ItemType Directory -Force -Path $BinDest | Out-Null
Copy-Item -Path (Join-Path $BundleRoot "bin\*") -Destination $BinDest -Force

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
Write-Host "Open a new terminal, then try:"
Write-Host "  cypha_rest.exe --listen 127.0.0.1:8099 ``"
Write-Host "    --cypha `"$InstallRoot\share\demo_fixtures\reference.cypha`" ``"
Write-Host "    --f-field-json `"$InstallRoot\share\demo_fixtures\f_field.json`""
