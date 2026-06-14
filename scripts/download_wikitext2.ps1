# Download WikiText-2 raw tokens into bench/data/wikitext2/wikitext-2/.
# PowerShell 5 compatible (Windows 10+).
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$DestDir = Join-Path $Root "bench\data\wikitext2"
$WikitextDir = Join-Path $DestDir "wikitext-2"
$Urls = @(
    "https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-2-v1.zip",
    "https://raw.githubusercontent.com/LogSSim/deeplearning_d2l_classes/main/class14_BERT/wikitext-2-v1.zip"
)
$ZipFile = Join-Path $env:TEMP "wikitext-2-v1.zip"

function Test-ValidZip {
    param([string]$Path)
    if (-not (Test-Path -LiteralPath $Path)) { return $false }
    $Info = Get-Item -LiteralPath $Path
    if ($Info.Length -lt 1024) { return $false }
    $Bytes = [System.IO.File]::ReadAllBytes($Path)
    return ($Bytes.Length -ge 4 -and $Bytes[0] -eq 0x50 -and $Bytes[1] -eq 0x4B)
}

$Required = @(
    (Join-Path $WikitextDir "wiki.train.tokens"),
    (Join-Path $WikitextDir "wiki.valid.tokens"),
    (Join-Path $WikitextDir "wiki.test.tokens")
)
$AllPresent = $true
foreach ($Path in $Required) {
    if (-not (Test-Path -LiteralPath $Path)) {
        $AllPresent = $false
        break
    }
}
if ($AllPresent) {
    Write-Host "WikiText-2 already present at $WikitextDir"
    exit 0
}

New-Item -ItemType Directory -Force -Path $DestDir | Out-Null
$Downloaded = $false
foreach ($Url in $Urls) {
    Write-Host "Downloading WikiText-2 from $Url ..."
    try {
        if (Get-Command curl.exe -ErrorAction SilentlyContinue) {
            & curl.exe -fsSL $Url -o $ZipFile
            if ($LASTEXITCODE -ne 0) { throw "curl exit $LASTEXITCODE" }
        } else {
            Invoke-WebRequest -Uri $Url -OutFile $ZipFile -UseBasicParsing -MaximumRedirection 10
        }
        if (-not (Test-ValidZip -Path $ZipFile)) {
            throw "Downloaded file is not a valid zip archive"
        }
        $Downloaded = $true
        break
    } catch {
        Write-Warning "Download failed from $Url : $_"
        Remove-Item -LiteralPath $ZipFile -Force -ErrorAction SilentlyContinue
    }
}
if (-not $Downloaded) {
    throw "Could not download WikiText-2 from any configured URL"
}

$ExtractRoot = Join-Path $env:TEMP ("wikitext2_extract_" + [guid]::NewGuid().ToString("N"))
New-Item -ItemType Directory -Force -Path $ExtractRoot | Out-Null
try {
    Expand-Archive -LiteralPath $ZipFile -DestinationPath $ExtractRoot -Force
    $Inner = Join-Path $ExtractRoot "wikitext-2"
    if (-not (Test-Path -LiteralPath $Inner)) {
        throw "Expected wikitext-2/ inside archive"
    }
    if (Test-Path -LiteralPath $WikitextDir) {
        Remove-Item -LiteralPath $WikitextDir -Recurse -Force
    }
    New-Item -ItemType Directory -Force -Path $WikitextDir | Out-Null
    Copy-Item -Path (Join-Path $Inner "*") -Destination $WikitextDir -Recurse -Force
}
finally {
    Remove-Item -LiteralPath $ExtractRoot -Recurse -Force -ErrorAction SilentlyContinue
    Remove-Item -LiteralPath $ZipFile -Force -ErrorAction SilentlyContinue
}

foreach ($Path in $Required) {
    if (-not (Test-Path -LiteralPath $Path)) {
        throw "Missing after extract: $Path"
    }
}
Write-Host "WikiText-2 installed to $WikitextDir"
exit 0
