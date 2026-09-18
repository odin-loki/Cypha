$ErrorActionPreference = "Continue"
$v3b = "hp\build\hp_v3b.exe"
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

Run "H1.4 dict 8mb" $v3b @("c","--mem","22","--dict","data\enwik8.8mb","hp\build\e8_8mb_dict.hp")
Run "H1.4 dict 1mb" $v3b @("c","--mem","22","--dict","data\enwik8.1mb","hp\build\e8_1mb_dict.hp")

Write-Host "==== compile v3b+num ===="
& g++ -std=c++17 -O2 -I hp/include -DHP_LINKWORD=0 -DHP_NUMERIC=1 -o hp/build/hp_num.exe hp/src/main.cpp
if ($LASTEXITCODE -eq 0) {
  Run "H2.8 num 8mb" "hp\build\hp_num.exe" @("c","--mem","22","data\enwik8.8mb","hp\build\e8_8mb_num.hp")
  Run "H2.8 num 1mb" "hp\build\hp_num.exe" @("c","--mem","22","data\enwik8.1mb","hp\build\e8_1mb_num.hp")
}

foreach ($pair in @(
  @{ n = "nowiki"; d = "-DHP_WIKI_STATES=0" },
  @{ n = "nowm";   d = "-DHP_WORD_MATCH=0" },
  @{ n = "nobrk";  d = "-DHP_BRACKET=0" }
)) {
  Write-Host "==== compile $($pair.n) ===="
  & g++ -std=c++17 -O2 -I hp/include -DHP_LINKWORD=0 -DHP_NUMERIC=0 "$($pair.d)" -o "hp\build\hp_$($pair.n).exe" hp/src/main.cpp
  if ($LASTEXITCODE -ne 0) { Write-Host "COMPILE FAIL $($pair.n)"; continue }
  Run "H0.3 $($pair.n) 8mb" "hp\build\hp_$($pair.n).exe" @("c","--mem","22","data\enwik8.8mb","hp\build\e8_8mb_$($pair.n).hp")
}
