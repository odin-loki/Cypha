# Combinatorial 1MB A.3 of Stats-for-Compression mixer gates on v5 base.
$ErrorActionPreference = "Stop"
$root = Split-Path (Split-Path $PSScriptRoot)
Set-Location $root
$base = @("-std=c++17", "-O2", "-I", "hp/include", "-DHP_MIXER_SKIP=24", "-DHP_SPARSE_UTF8=1")
$jobs = @(
  @{ id = "shape";   d = @("-DHP_GATE_SHAPE=1") },
  @{ id = "branch";  d = @("-DHP_GATE_BRANCH=1") },
  @{ id = "disp";    d = @("-DHP_GATE_DISP=1") },
  @{ id = "mlen2";   d = @("-DHP_GATE_MLEN2=1") },
  @{ id = "argmax";  d = @("-DHP_GATE_ARGMAX=1") },
  @{ id = "sgrp";    d = @("-DHP_SEN_GROUP=1") },
  @{ id = "fword";   d = @("-DHP_FIRST_WORD=1") },
  @{ id = "posg";    d = @("-DHP_POS_GATE=1") },
  @{ id = "bd";      d = @("-DHP_GATE_BRANCH=1", "-DHP_GATE_DISP=1") },
  @{ id = "bdm";     d = @("-DHP_GATE_BRANCH=1", "-DHP_GATE_DISP=1", "-DHP_GATE_MLEN2=1") },
  @{ id = "bds";     d = @("-DHP_GATE_BRANCH=1", "-DHP_GATE_DISP=1", "-DHP_GATE_SHAPE=1") },
  @{ id = "v6bdm";   d = @("-DHP_SENWORD=1", "-DHP_GATE_BRANCH=1", "-DHP_GATE_DISP=1", "-DHP_GATE_MLEN2=1") }
)

foreach ($j in $jobs) {
  $exe = "hp/build/hp_g_$($j.id).exe"
  Write-Host "BUILD $($j.id)"
  & g++ @base @($j.d) -o $exe hp/src/main.cpp
  if ($LASTEXITCODE -ne 0) { Write-Host "FAIL build $($j.id)"; exit 1 }
}

$v5 = 231849
$run = {
  param($exe, $out, $id, $v5)
  & $exe c --mem 22 data/enwik8.1mb $out | Out-Null
  $c = (Get-Item $out).Length
  "$id $c $($c - $v5)"
}

$procs = @()
foreach ($j in $jobs) {
  $exe = "hp/build/hp_g_$($j.id).exe"
  $out = "hp/build/e8_1mb_g_$($j.id).hp"
  $procs += Start-Job -ScriptBlock $run -ArgumentList $exe, $out, $j.id, $v5
}
Write-Host "RUNNING $($procs.Count) x 1MB"
$procs | Wait-Job | Out-Null
foreach ($p in $procs) {
  Receive-Job $p
  Remove-Job $p
}
