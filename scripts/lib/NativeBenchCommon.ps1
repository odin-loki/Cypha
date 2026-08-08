# Shared helpers for the native CyphaLM bench/sweep/validate scripts under scripts/.
# Dot-source this file from a script that lives directly under scripts/, e.g.:
#   . (Join-Path $PSScriptRoot "lib\NativeBenchCommon.ps1")
#   $root = Get-CyphaRepoRoot -ScriptRoot $PSScriptRoot
#
# NOTE ON $PSScriptRoot: inside a function defined in a dot-sourced file, $PSScriptRoot
# resolves to the directory of *this* file (scripts\lib), not the caller's directory.
# That's why Get-CyphaRepoRoot below takes the caller's $PSScriptRoot explicitly instead
# of assuming a fixed number of Split-Path hops - passing the wrong hop count from a
# function is exactly the class of bug this helper exists to prevent.

function Get-CyphaRepoRoot {
    <#
    .SYNOPSIS
        Resolves the Cypha repo root reliably.
    .PARAMETER ScriptRoot
        Pass the calling script's $PSScriptRoot. For scripts living directly under
        scripts/ (the common case), the repo root is one level up - the same
        "$root = Split-Path $PSScriptRoot -Parent" pattern used by the working
        overnight scripts.
    #>
    param(
        [string]$ScriptRoot
    )

    $candidates = @()
    if ($ScriptRoot) {
        $candidates += (Split-Path $ScriptRoot -Parent)
    }
    # This file always lives at <repo>\scripts\lib\NativeBenchCommon.ps1, so two levels
    # up from its own directory is a reliable fallback/self-check independent of the caller.
    $candidates += (Split-Path (Split-Path $PSScriptRoot -Parent) -Parent)

    foreach ($candidate in $candidates) {
        if ($candidate -and (Test-Path (Join-Path $candidate "native")) -and (Test-Path (Join-Path $candidate "bench"))) {
            return $candidate
        }
    }
    if ($candidates.Count -gt 0 -and $candidates[0]) {
        return $candidates[0]
    }
    throw "Get-CyphaRepoRoot: could not resolve repo root from ScriptRoot='$ScriptRoot'"
}

function Get-DefaultNativeBuildDir {
    <#
    .SYNOPSIS
        Returns a sensible default native build directory OUTSIDE the OneDrive-synced
        repo tree, to avoid OneDrive sync churn on build artifacts (object files, exes).
    .PARAMETER Override
        If non-empty, returned as-is (explicit caller override wins).
    .NOTES
        Falls back to $env:CYPHA_NATIVE_BUILD_DIR when set, then to
        $env:LOCALAPPDATA\cypha_native_build.
    #>
    param(
        [string]$Override
    )

    if ($Override) {
        return $Override
    }
    if ($env:CYPHA_NATIVE_BUILD_DIR) {
        return $env:CYPHA_NATIVE_BUILD_DIR
    }
    return (Join-Path $env:LOCALAPPDATA "cypha_native_build")
}

function Resolve-NativeBuildDir {
    <#
    .SYNOPSIS
        Resolves BuildDir to an absolute path (supports repo-relative or absolute overrides).
    #>
    param(
        [Parameter(Mandatory = $true)][string]$RepoRoot,
        [Parameter(Mandatory = $true)][string]$BuildDir
    )

    if ([System.IO.Path]::IsPathRooted($BuildDir)) {
        return $BuildDir
    }
    return (Join-Path $RepoRoot $BuildDir)
}

function Resolve-NativeExePath {
    <#
    .SYNOPSIS
        Finds a native tool executable under an absolute build directory.
    #>
    param(
        [Parameter(Mandatory = $true)][string]$BuildDir,
        [Parameter(Mandatory = $true)][string]$Stem
    )

    foreach ($candidate in @(
        (Join-Path $BuildDir "$Stem.exe"),
        (Join-Path $BuildDir $Stem),
        (Join-Path $BuildDir "Release\$Stem.exe"),
        (Join-Path $BuildDir "RelWithDebInfo\$Stem.exe")
    )) {
        if (Test-Path $candidate) {
            return $candidate
        }
    }
    return $null
}

function Invoke-NativeWithProgressLog {
    <#
    .SYNOPSIS
        Safely invokes a native .exe that emits progress lines to stderr, without
        $ErrorActionPreference = 'Stop' treating those progress lines as fatal errors.
    .DESCRIPTION
        Native tools log progress on stderr (e.g. [cyphalm], [cell_sweep]). With 2>&1,
        PowerShell turns each stderr line into an ErrorRecord; ErrorActionPreference
        'Stop' would abort on the very first progress line. This temporarily switches
        to 'Continue' around the call and restores the previous value afterwards.
    .PARAMETER LogPath
        Optional. When set, output is also teed (appended) to this file.
    #>
    param(
        [Parameter(Mandatory = $true)][string]$Exe,
        [string[]]$NativeArgs = @(),
        [string]$LogPath
    )

    $prevEap = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        if ($LogPath) {
            & $Exe @NativeArgs 2>&1 | Tee-Object -FilePath $LogPath -Append
        } else {
            & $Exe @NativeArgs 2>&1
        }
    } finally {
        $ErrorActionPreference = $prevEap
    }
}
