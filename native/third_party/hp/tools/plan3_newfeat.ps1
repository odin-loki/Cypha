$ErrorActionPreference = "Continue"
function Run([string]$label, [string]$exe, [string[]]$hpArgs) {
  Write-Host "==== $label ===="
  $sw = [Diagnostics.Stopwatch]::StartNew()
  & $exe @hpArgs
  $sw.Stop()
  $out = $hpArgs[-1]
  $inp = $hpArgs[-2]
  if (Test-Path $out) {
    $c = (Get-Item $out).Length
    $n = (Get-Item $inp).Length
    Write-Host "$label  $n -> $c  $([math]::Round(8.0*$c/$n,4)) bpc  $($sw.Elapsed.TotalSeconds) s"
  }
}

Write-Host "==== compile skip24 ===="
& g++ -std=c++17 -O2 -I hp/include -DHP_LINKWORD=0 -DHP_NUMERIC=0 "-DHP_MIXER_SKIP=24" -o hp/build/hp_skip24.exe hp/src/main.cpp
if ($LASTEXITCODE -eq 0) {
  Run "H3.1 skip24 1mb" "hp\build\hp_skip24.exe" @("c","--mem","22","data\enwik8.1mb","hp\build\e8_1mb_skip24.hp")
  Run "H3.1 skip24 8mb" "hp\build\hp_skip24.exe" @("c","--mem","22","data\enwik8.8mb","hp\build\e8_8mb_skip24.hp")
}

Write-Host "==== compile utf8 match ===="
& g++ -std=c++17 -O2 -I hp/include -DHP_LINKWORD=0 -DHP_NUMERIC=0 -DHP_SPARSE_UTF8=1 -o hp/build/hp_utf8.exe hp/src/main.cpp
if ($LASTEXITCODE -eq 0) {
  Run "H2.3 utf8 1mb" "hp\build\hp_utf8.exe" @("c","--mem","22","data\enwik8.1mb","hp\build\e8_1mb_utf8.hp")
  Run "H2.3 utf8 8mb" "hp\build\hp_utf8.exe" @("c","--mem","22","data\enwik8.8mb","hp\build\e8_8mb_utf8.hp")
}

Write-Host "==== H1.5 wiki xform ===="
python hp/tools/wiki_xform.py check data/enwik8.8mb
python hp/tools/wiki_xform.py e data/enwik8.8mb data/enwik8.8mb.xf
python hp/tools/wiki_xform.py e data/enwik8.1mb data/enwik8.1mb.xf
if (Test-Path hp\build\hp_v3b.exe) {
  Run "H1.5 xf 8mb" "hp\build\hp_v3b.exe" @("c","--mem","22","data\enwik8.8mb.xf","hp\build\e8_8mb_xf.hp")
  Run "H1.5 xf 1mb" "hp\build\hp_v3b.exe" @("c","--mem","22","data\enwik8.1mb.xf","hp\build\e8_1mb_xf.hp")
}
