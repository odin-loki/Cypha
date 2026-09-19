# Sequential proxy256k RSS sweep of leftover champs v50..v78 (not m26).
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
if (-not (Test-Path (Join-Path $root "hp\src\main.cpp"))) {
    $root = "C:\Users\odinl\OneDrive\Desktop\Compression Algorithm"
}
$build = Join-Path $root "hp\build"
$infile = Join-Path $root "hp\data\proxy256k.xml"
$outdir = Join-Path $env:LOCALAPPDATA "hp_lab\rss256k"
$log = Join-Path $outdir "sweep.csv"
New-Item -ItemType Directory -Force -Path $outdir | Out-Null

$exes = Get-ChildItem (Join-Path $build "hp_v*.exe") |
    Where-Object { $_.Name -match '^hp_v(\d+)\.exe$' } |
    ForEach-Object {
        $n = [int]$Matches[1]
        if ($n -ge 50 -and $n -le 78) { $_ }
    } | Sort-Object { [int]([regex]::Match($_.Name, '\d+').Value) }

"name,archive_bytes,peak_ws_bytes,peak_ws_gb,wall_s,exit" | Set-Content -Encoding utf8 $log
Write-Host "infile=$infile"
Write-Host "n=$($exes.Count) log=$log"

foreach ($exe in $exes) {
    $arc = Join-Path $outdir ($exe.BaseName + ".hp")
    if (Test-Path $arc) { Remove-Item $arc -Force }
    $sw = [System.Diagnostics.Stopwatch]::StartNew()
    $psi = New-Object System.Diagnostics.ProcessStartInfo
    $psi.FileName = $exe.FullName
    $psi.Arguments = "c --mem 22 `"$infile`" `"$arc`""
    $psi.UseShellExecute = $false
    $psi.CreateNoWindow = $true
    $psi.RedirectStandardOutput = $true
    $psi.RedirectStandardError = $true
    $p = New-Object System.Diagnostics.Process
    $p.StartInfo = $psi
    [void]$p.Start()
    $peak = [int64]0
    while (-not $p.HasExited) {
        try {
            $p.Refresh()
            if ($p.WorkingSet64 -gt $peak) { $peak = $p.WorkingSet64 }
            if ($p.PeakWorkingSet64 -gt $peak) { $peak = $p.PeakWorkingSet64 }
        } catch { }
        Start-Sleep -Milliseconds 80
    }
    $sw.Stop()
    $code = $p.ExitCode
    [void]$p.StandardOutput.ReadToEnd()
    [void]$p.StandardError.ReadToEnd()
    $asz = 0
    if (Test-Path $arc) { $asz = (Get-Item $arc).Length }
    $gb = [math]::Round($peak / 1GB, 2)
    $line = "{0},{1},{2},{3},{4:N1},{5}" -f $exe.Name, $asz, $peak, $gb, $sw.Elapsed.TotalSeconds, $code
    Add-Content -Encoding utf8 $log $line
    Write-Host $line
    if (Test-Path $arc) { Remove-Item $arc -Force }
}
Write-Host "DONE $log"
