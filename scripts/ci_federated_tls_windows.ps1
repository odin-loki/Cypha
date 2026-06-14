# Local mirror of the optional "Federated TLS smoke (OpenSSL)" job from .github/workflows/ci.yml.
# Blocking CI gate elsewhere: 98+ CTests (`ctest -R native_`; see scripts/cypha_native_validate_all.ps1).
# Builds with -DCYPHA_ENABLE_OPENSSL=ON when OpenSSL is found via vcpkg or OPENSSL_ROOT_DIR.
# Runs ctest -R native_federated_tls_smoke. Exits 0 (skip) when OpenSSL is unavailable.
param(
  [string]$BuildDir = "",
  [int]$Parallel = 0
)

$ErrorActionPreference = "Stop"
$root = Split-Path $PSScriptRoot -Parent
if ([string]::IsNullOrWhiteSpace($BuildDir)) {
  $BuildDir = if ($env:CYPHA_FEDERATED_TLS_BUILD_DIR) {
    $env:CYPHA_FEDERATED_TLS_BUILD_DIR
  } else {
    Join-Path $root "native/build-federated-tls-ci"
  }
}
if ($Parallel -le 0) {
  $Parallel = if ($env:CI_NATIVE_J) { [int]$env:CI_NATIVE_J } else { $env:NUMBER_OF_PROCESSORS }
  if (-not $Parallel -or $Parallel -lt 1) { $Parallel = 4 }
}

function Test-OpenSslRoot {
  param([string]$Dir)
  if ([string]::IsNullOrWhiteSpace($Dir)) { return $false }
  return (Test-Path (Join-Path $Dir "include/openssl/ssl.h"))
}

function Resolve-VcpkgRoot {
  if ($env:VCPKG_ROOT -and (Test-Path (Join-Path $env:VCPKG_ROOT "vcpkg.exe"))) {
    return $env:VCPKG_ROOT
  }
  if ($env:CMAKE_TOOLCHAIN_FILE) {
    $tc = $env:CMAKE_TOOLCHAIN_FILE -replace "\\", "/"
    if ($tc -match "/scripts/buildsystems/vcpkg\.cmake$") {
      $candidate = Split-Path (Split-Path (Split-Path $tc -Parent) -Parent) -Parent
      if (Test-Path (Join-Path $candidate "vcpkg.exe")) { return $candidate }
    }
  }
  return $null
}

function Test-VcpkgOpenSsl {
  param([string]$VcpkgRoot)
  if (-not $VcpkgRoot) { return $false }
  $triplet = if ($env:VCPKG_DEFAULT_TRIPLET) { $env:VCPKG_DEFAULT_TRIPLET } else { "x64-windows" }
  $installed = Join-Path $VcpkgRoot "installed/$triplet"
  return (Test-Path (Join-Path $installed "include/openssl/ssl.h"))
}

$buildType = if ($env:CMAKE_BUILD_TYPE) { $env:CMAKE_BUILD_TYPE } else { "Release" }
$cmakeArgs = @(
  "-S", (Join-Path $root "native"),
  "-B", $BuildDir,
  "-DCMAKE_BUILD_TYPE=$buildType",
  "-DCYPHA_ENABLE_OPENSSL=ON"
)

$opensslHint = $false
if (Test-OpenSslRoot $env:OPENSSL_ROOT_DIR) {
  $opensslHint = $true
  $cmakeArgs += "-DOPENSSL_ROOT_DIR=$($env:OPENSSL_ROOT_DIR)"
  Write-Host "OpenSSL: OPENSSL_ROOT_DIR=$($env:OPENSSL_ROOT_DIR)"
}

$vcpkgRoot = Resolve-VcpkgRoot
if ($vcpkgRoot) {
  $toolchain = Join-Path $vcpkgRoot "scripts/buildsystems/vcpkg.cmake"
  if (Test-Path $toolchain) {
    $cmakeArgs += "-DCMAKE_TOOLCHAIN_FILE=$toolchain"
    if (Test-VcpkgOpenSsl $vcpkgRoot) {
      $opensslHint = $true
      Write-Host "OpenSSL: vcpkg at $vcpkgRoot"
    }
  }
} elseif ($env:CMAKE_TOOLCHAIN_FILE) {
  $cmakeArgs += "-DCMAKE_TOOLCHAIN_FILE=$($env:CMAKE_TOOLCHAIN_FILE)"
}

if (-not $opensslHint) {
  Write-Host "SKIP: OpenSSL not found (set OPENSSL_ROOT_DIR or install openssl via vcpkg)"
  exit 0
}

& cmake @cmakeArgs
if ($LASTEXITCODE -ne 0) {
  throw "cmake configure failed exit=$LASTEXITCODE"
}

$cacheFile = Join-Path $BuildDir "CMakeCache.txt"
if (-not (Test-Path $cacheFile) -or -not (Select-String -Path $cacheFile -Pattern '^CYPHA_ENABLE_OPENSSL:BOOL=ON' -Quiet)) {
  Write-Host "SKIP: CYPHA_ENABLE_OPENSSL not enabled (OpenSSL not found by CMake)"
  exit 0
}

& cmake --build $BuildDir --parallel $Parallel --target federated_tls_smoke
if ($LASTEXITCODE -ne 0) {
  throw "cmake --build federated_tls_smoke failed exit=$LASTEXITCODE"
}

$opensslCli = Get-Command openssl -ErrorAction SilentlyContinue
if (-not $opensslCli) {
  Write-Host "SKIP: openssl CLI not on PATH (install OpenSSL or add its bin dir to PATH)"
  exit 0
}

& ctest --test-dir $BuildDir --output-on-failure -R native_federated_tls_smoke
if ($LASTEXITCODE -ne 0) {
  throw "ctest native_federated_tls_smoke failed exit=$LASTEXITCODE"
}

Write-Host "OK: federated TLS smoke $BuildDir" -ForegroundColor Green
