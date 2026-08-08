# Materialize conflict-forecasting CSVs under bench/data/forecast.
# Run from repo root:
#   .\scripts\fetch_forecast_data.ps1           # ensure sample + canonical aliases
#   .\scripts\fetch_forecast_data.ps1 -Bulk     # also fetch public bulk snapshots
#   .\scripts\fetch_forecast_data.ps1 -Repair   # fix corrupted gml_mid / gdelt aliases

param(
    [switch]$Bulk,
    [switch]$Repair,
    [int]$GdeltMaxRows = 5000,
    [int]$ViewsMaxRows = 4000
)

$ErrorActionPreference = "Stop"
$root = Split-Path $PSScriptRoot -Parent
$dest = Join-Path $root "bench\data\forecast"
$bulkDir = Join-Path $dest "bulk"
New-Item -ItemType Directory -Force -Path $dest | Out-Null
New-Item -ItemType Directory -Force -Path $bulkDir | Out-Null

function Write-Utf8File([string]$path, [string]$text) {
    [System.IO.File]::WriteAllText($path, $text, [System.Text.UTF8Encoding]::new($false))
}

function Test-MidHeader([string]$path) {
    if (-not (Test-Path $path)) { return $false }
    try {
        $line = [System.IO.File]::ReadAllText($path, [System.Text.UTF8Encoding]::new($false)).Split("`n")[0].Trim()
        return $line.StartsWith("year,dyad,theater")
    } catch { return $false }
}

function Test-GdeltHeader([string]$path) {
    if (-not (Test-Path $path)) { return $false }
    try {
        $line = [System.IO.File]::ReadAllText($path, [System.Text.UTF8Encoding]::new($false)).Split("`n")[0].Trim()
        return $line.StartsWith("year,month,day,actor1")
    } catch { return $false }
}

function Copy-IfMissing([string]$srcName, [string]$dstName) {
    $src = Join-Path $dest $srcName
    $dst = Join-Path $dest $dstName
    if ((Test-Path $src) -and -not (Test-Path $dst)) {
        Copy-Item $src $dst -Force
        Write-Host "copied $srcName -> $dstName"
    }
}

function Ensure-SampleMid {
    $sample = Join-Path $dest "sample_mid.csv"
    if (-not (Test-Path $sample)) {
        $rows = @("year,dyad,theater,hostility,prev_hostility,great_power,duration,gdp_ratio,escalated")
        foreach ($y in 2016..2024) {
            $rows += "$y,USA_CHN,TWN,2,1,1,3,1.1,0"
            $rows += "$y,RUS_UKR,UKR,4,2,1,2,0.4,1"
            $rows += "$y,ISR_IRN,MID,3,1,1,1,0.8,1"
        }
        Write-Utf8File $sample (($rows -join "`n") + "`n")
        Write-Host "created sample_mid.csv"
    }
}

function Ensure-SampleGdelt {
    $sample = Join-Path $dest "sample_gdelt.csv"
    if (-not (Test-Path $sample)) {
        $rows = @("year,month,day,actor1,actor2,cameo_code,goldstein,theater")
        foreach ($m in 1..12) {
            $rows += "2024,$m,10,USA,CHN,120,-2,TWN"
            $rows += "2024,$m,15,RUS,UKR,190,-8,UKR"
            $rows += "2024,$m,20,ISR,IRN,200,-10,MID"
        }
        Write-Utf8File $sample (($rows -join "`n") + "`n")
        Write-Host "created sample_gdelt.csv"
    }
}

function Repair-Aliases {
    $gml = Join-Path $dest "gml_mid.csv"
    $gdelt = Join-Path $dest "gdelt_events.csv"
    $sampleMid = Join-Path $dest "sample_mid.csv"
    $sampleGdelt = Join-Path $dest "sample_gdelt.csv"
    if ((Test-Path $sampleMid) -and (-not (Test-MidHeader $gml) -or $Repair)) {
        Copy-Item $sampleMid $gml -Force
        Write-Host "repaired gml_mid.csv from sample_mid.csv"
    }
    if ((Test-Path $sampleGdelt) -and (-not (Test-GdeltHeader $gdelt) -or $Repair)) {
        Copy-Item $sampleGdelt $gdelt -Force
        Write-Host "repaired gdelt_events.csv from sample_gdelt.csv"
    }
}

function Convert-GdeltExportLine([string]$line) {
    if ([string]::IsNullOrWhiteSpace($line)) { return $null }
    $fields = $line -split "`t"
    if ($fields.Count -lt 31) { return $null }
    $sql = $fields[1]
    if ($sql.Length -lt 8) { return $null }
    $year = $sql.Substring(0, 4)
    $month = [int]$sql.Substring(4, 2)
    $day = [int]$sql.Substring(6, 2)
    $a1 = ($fields[6] -replace '"', '').Trim()
    $a2 = ($fields[16] -replace '"', '').Trim()
    $cameo = ($fields[26] -replace '"', '').Trim()
    $gold = ($fields[30] -replace '"', '').Trim()
    if ($a1 -eq '' -or $a2 -eq '') { return $null }
    $theater = "GLB"
    $u = ($a1 + $a2).ToUpperInvariant()
    if ($u -match 'CHN|TWN|CHINA|TAIWAN') { $theater = 'TWN' }
    elseif ($u -match 'RUS|UKR|RUSSIA|UKRAINE') { $theater = 'UKR' }
    elseif ($u -match 'ISR|IRN|PSE|ISRAEL|IRAN|PALEST') { $theater = 'MID' }
    elseif ($u -match 'PRK|KOR|KOREA') { $theater = 'PRK' }
    elseif ($u -match 'IND|PAK|INDIA|PAKIST') { $theater = 'SAS' }
    return "$year,$month,$day,$a1,$a2,$cameo,$gold,$theater"
}

function Fetch-GdeltBulk {
    param([int]$MaxRows = 5000)
    $outRaw = Join-Path $bulkDir "gdelt_export_latest.tab"
    $outCsv = Join-Path $dest "gdelt_events.csv"
    $dates = @()
    for ($i = 0; $i -lt 7; $i++) {
        $dates += (Get-Date).AddDays(-$i).ToString('yyyyMMdd')
    }
    $downloaded = $false
    foreach ($d in $dates) {
        $zipUrl = "http://data.gdeltproject.org/events/$d.export.CSV.zip"
        $zipPath = Join-Path $bulkDir "$d.export.CSV.zip"
        try {
            Write-Host "fetching $zipUrl"
            Invoke-WebRequest -Uri $zipUrl -OutFile $zipPath -UseBasicParsing -TimeoutSec 120
            Expand-Archive -Path $zipPath -DestinationPath $bulkDir -Force
            $tab = Join-Path $bulkDir "$d.export.CSV"
            if (Test-Path $tab) {
                Copy-Item $tab $outRaw -Force
                $downloaded = $true
                break
            }
        } catch {
            Write-Host "  skip $d ($($_.Exception.Message))"
        }
    }
    if (-not $downloaded) {
        Write-Host "GDELT bulk: no recent export found (using sample alias)"
        return
    }
    $rows = New-Object System.Collections.Generic.List[string]
    $rows.Add("year,month,day,actor1,actor2,cameo_code,goldstein,theater")
    $count = 0
    foreach ($line in [System.IO.File]::ReadLines($outRaw)) {
        $conv = Convert-GdeltExportLine $line
        if ($null -ne $conv) {
            $rows.Add($conv)
            $count++
            if ($count -ge $MaxRows) { break }
        }
    }
    if ($count -lt 10) {
        Write-Host "GDELT bulk: too few rows converted ($count); keeping sample"
        return
    }
    Write-Utf8File $outCsv (($rows -join "`n") + "`n")
    Write-Host "GDELT bulk: wrote $count rows -> gdelt_events.csv"
}

function Fetch-MidBulk {
    $zipUrl = "https://correlatesofwar.org/wp-content/uploads/dyadic_mid_4.03.zip"
    $zipPath = Join-Path $bulkDir "dyadic_mid_4.03.zip"
    $outCsv = Join-Path $dest "gml_mid.csv"
    try {
        Write-Host "fetching $zipUrl"
        Invoke-WebRequest -Uri $zipUrl -OutFile $zipPath -UseBasicParsing -TimeoutSec 180
        Expand-Archive -Path $zipPath -DestinationPath (Join-Path $bulkDir "cow_mid") -Force
        $src = Get-ChildItem -Path (Join-Path $bulkDir "cow_mid") -Recurse -Filter "*.csv" | Where-Object { $_.FullName -notmatch '__MACOSX' } | Select-Object -First 1
        if ($null -eq $src) { throw "no CSV inside MID zip" }
        $rows = New-Object System.Collections.Generic.List[string]
        $rows.Add("year,dyad,theater,hostility,prev_hostility,great_power,duration,gdp_ratio,escalated")
        $header = $null
        $prev = @{}
        $count = 0
        foreach ($line in [System.IO.File]::ReadLines($src.FullName)) {
            if ($null -eq $header) { $header = ($line -split ',').ForEach({ $_.Trim('"').ToLowerInvariant() }); continue }
            $fields = $line -split ',(?=(?:[^"]*"[^"]*")*[^"]*$)'
            if ($fields.Count -lt $header.Count) { continue }
            $map = @{}
            for ($i = 0; $i -lt $header.Count; $i++) { $map[$header[$i]] = ($fields[$i] -replace '"', '').Trim() }
            if (-not $map.ContainsKey('year')) { continue }
            $year = [int]$map['year']
            if ($map.ContainsKey('statea')) {
                $cc1 = $map['statea']
                $cc2 = $map['stateb']
            } else {
                $cc1 = $map['ccode1']
                $cc2 = $map['ccode2']
            }
            if ([string]::IsNullOrWhiteSpace($cc1) -or [string]::IsNullOrWhiteSpace($cc2)) { continue }
            $dyad = "${cc1}_${cc2}"
            $hostility = 1
            foreach ($k in @('hihost','hostlev','hostlevel','highestaction','hostility','highact')) {
                if ($map.ContainsKey($k) -and $map[$k] -match '^\d+$') { $hostility = [int]$map[$k]; break }
            }
            $key = $dyad
            $prevHost = if ($prev.ContainsKey($key)) { $prev[$key] } else { $hostility }
            $prev[$key] = $hostility
            $esc = if ($hostility -gt $prevHost) { 1 } else { 0 }
            $rows.Add("$year,$dyad,GLB,$hostility,$prevHost,1,1,1.0,$esc")
            $count++
            if ($count -ge 8000) { break }
        }
        if ($count -lt 50) { throw "too few MID rows mapped ($count)" }
        Write-Utf8File $outCsv (($rows -join "`n") + "`n")
        Write-Host "MID bulk: wrote $count rows -> gml_mid.csv"
    } catch {
        Write-Host "MID bulk: $($_.Exception.Message) (keeping sample alias)"
    }
}

function Get-LatestViewsPredictorRun {
    $index = Invoke-RestMethod -Uri "https://api.viewsforecasting.org/" -TimeoutSec 90
    $pred = @($index.runs | Where-Object { $_ -match '^predictors_fatalities' } | Sort-Object -Descending)
    if ($pred.Count -gt 0) {
        return $pred[0]
    }
    $fat = @($index.runs | Where-Object { $_ -match '^fatalities00' } | Sort-Object -Descending)
    if ($fat.Count -gt 0) {
        return $fat[0]
    }
    return $null
}

function Copy-ViewsSampleScaffold([string]$outCsv) {
    $sample = Join-Path $dest "sample_views.csv"
    if (Test-Path $sample) {
        Copy-Item $sample $outCsv -Force
        Write-Host "VIEWS bulk: scaffold copied sample_views.csv -> views_bulk.csv"
        Write-Host "  full replication sets: https://viewsforecasting.org/data/"
    }
}

function Fetch-ViewsBulk {
    param([int]$MaxRows = 4000, [int]$PageSize = 500)
    $outCsv = Join-Path $dest "views_bulk.csv"
    try {
        $run = Get-LatestViewsPredictorRun
        if (-not $run) {
            throw "no fatalities/predictors run in VIEWS API index"
        }
        Write-Host "fetching VIEWS API predictors run $run"
        $obs = New-Object System.Collections.Generic.List[object]
        $url = "https://api.viewsforecasting.org/${run}/cm?pagesize=$PageSize&page=1"
        while ($url -and $obs.Count -lt $MaxRows) {
            $page = Invoke-RestMethod -Uri $url -TimeoutSec 120
            foreach ($row in $page.data) {
                $fat = 0.0
                foreach ($k in @('ucdp_ged_sb_best_sum', 'ucdp_ged_os_best_sum', 'ucdp_ged_ns_best_sum')) {
                    if ($row.PSObject.Properties.Name -contains $k -and $null -ne $row.$k) {
                        $fat += [double]$row.$k
                    }
                }
                if ($fat -le 0.0 -or $row.year -lt 2020) { continue }
                $country = if ($row.isoab) { [string]$row.isoab } else { [string]$row.name }
                if ([string]::IsNullOrWhiteSpace($country)) { continue }
                $obs.Add([pscustomobject]@{
                    country    = $country.ToUpperInvariant()
                    year       = [int]$row.year
                    month      = [int]$row.month
                    fatalities = [math]::Round($fat, 2)
                    key        = ([int]$row.year * 100) + [int]$row.month
                })
                if ($obs.Count -ge $MaxRows) { break }
            }
            $url = if ($page.next_page) { [string]$page.next_page } else { $null }
        }
        if ($obs.Count -lt 80) {
            throw "too few VIEWS rows with fatalities ($($obs.Count))"
        }
        $sorted = $obs | Sort-Object key
        $holdoutKeys = @($sorted | Select-Object -ExpandProperty key -Unique | Sort-Object | Select-Object -Last 3)
        $holdoutSet = @{}
        foreach ($k in $holdoutKeys) { $holdoutSet[$k] = $true }
        $rows = New-Object System.Collections.Generic.List[string]
        $rows.Add("country,year,month,fatalities,split")
        foreach ($o in $sorted) {
            $split = if ($holdoutSet.ContainsKey($o.key)) { "holdout" } else { "train" }
            $rows.Add("$($o.country),$($o.year),$($o.month),$($o.fatalities),$split")
        }
        Write-Utf8File $outCsv (($rows -join "`n") + "`n")
        $trainN = @($sorted | Where-Object { -not $holdoutSet.ContainsKey($_.key) }).Count
        $holdN = $sorted.Count - $trainN
        Write-Host "VIEWS bulk: wrote $($sorted.Count) rows (train=$trainN holdout=$holdN) -> views_bulk.csv"
    } catch {
        Write-Host "VIEWS bulk: $($_.Exception.Message) (sample scaffold fallback)"
        Copy-ViewsSampleScaffold $outCsv
    }
}

Ensure-SampleMid
Ensure-SampleGdelt
Copy-IfMissing "sample_mid.csv" "gml_mid.csv"
Copy-IfMissing "sample_gdelt.csv" "gdelt_events.csv"
Repair-Aliases

if ($Bulk) {
    Write-Host ""
    Write-Host "== bulk fetch (public snapshots) ==" -ForegroundColor Yellow
    Fetch-GdeltBulk -MaxRows $GdeltMaxRows
    Fetch-MidBulk
    Fetch-ViewsBulk -MaxRows $ViewsMaxRows
}

Write-Host ""
Write-Host "Forecast data ready under $dest"
Write-Host "fetch_forecast_data OK"
