$ErrorActionPreference = "Continue"
New-Item -ItemType Directory -Force -Path hp\build | Out-Null
$jobs = @(
  @{ s = 16; e = 1024; n = "d16e1024" },
  @{ s = 24; e = 1024; n = "d24e1024" },
  @{ s = 32; e = 1024; n = "d32e1024" },
  @{ s = 12; e = 512;  n = "d12e512" },
  @{ s = 12; e = 2048; n = "d12e2048" }
)
foreach ($j in $jobs) {
  $slots = [string]$j.s
  $eval = [string]$j.e
  $name = [string]$j.n
  $exe = "hp\build\hp_$name.exe"
  Write-Host "==== compile $name slots=$slots eval=$eval ===="
  & g++ -std=c++17 -O2 -I hp/include -DHP_LINKWORD=0 -DHP_NUMERIC=0 "-DHP_DISC_SLOTS=$slots" "-DHP_DISC_EVAL=$eval" -o $exe hp/src/main.cpp
  if ($LASTEXITCODE -ne 0) { Write-Host "COMPILE FAIL $name"; continue }
  Write-Host "==== run $name 1MB mem22 ===="
  & $exe c --mem 22 data\enwik8.1mb "hp\build\e8_$name.hp"
  $c = (Get-Item "hp\build\e8_$name.hp").Length
  Write-Host "$name $c B  $([math]::Round(8.0*$c/1048576,4)) bpc  vs v3b-mem22 232151"
}
