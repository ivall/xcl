# Run with powershell.exe or pwsh.exe. Exercise the real smoke block with a fixture process.
$ErrorActionPreference = 'Stop'
$smoke = Join-Path $PSScriptRoot 'smoke.ps1'
$fixture = Join-Path ([IO.Path]::GetTempPath()) ('xcl-smoke-memory-' + [guid]::NewGuid())
$null = New-Item -ItemType Directory -Path $fixture
$xcl = Join-Path $fixture 'xcl.cmd'
$log = Join-Path $fixture 'xcl.log'
$report = Join-Path $fixture 'report.json'
try {
    [IO.File]::WriteAllText($xcl, "@echo off`r`necho Log: %~dp0xcl.log`r`n")
    [IO.File]::WriteAllText($log, "(1 frames shown, 1 PresentX calls)`n")
    function Assert-Memory($json, $expected) {
        if ($null -eq $json) {
            Remove-Item -LiteralPath $report -ErrorAction SilentlyContinue
        } else {
            [IO.File]::WriteAllText($report, $json)
        }
        $block = & $smoke -Game 'D:\Games\SomeTitle' -Seconds 1 -Xcl $xcl -Title 'fixture'
        $line = $block -split "`n" | Where-Object { $_ -like 'coherent memory:*' } | ForEach-Object { $_.TrimEnd("`r") }
        if ($line -ne "coherent memory: $expected") {
            throw "coherent memory was '$line', expected '$expected'"
        }
        if ($json -like '*Fixture GPU*' -and $block -notmatch '(?m)^gpu: Fixture GPU\r?$') {
            throw 'The adapter from report.json was lost.'
        }
    }
    Assert-Memory '{"d3d12":{"adapter":"Fixture GPU (12 MiB VRAM)","uma_mb":512}}' '512 MB'
    Assert-Memory '{"d3d12":{"adapter":"Fixture GPU (12 MiB VRAM)","uma_mb":4095}}' '4095 MB'
    Assert-Memory '{"d3d12":{"adapter":"Fixture GPU (12 MiB VRAM)","uma_mb":0}}' '0 MB'
    Assert-Memory '{"d3d12":{"adapter":"Fixture GPU (12 MiB VRAM)"}}' 'n/a'
    Assert-Memory '{"d3d12":{"adapter":"Fixture GPU (12 MiB VRAM)","uma_mb":null}}' 'n/a'
    Assert-Memory '{"d3d12":{"adapter":"Fixture GPU (12 MiB VRAM)","uma_mb":"bad"}}' 'n/a'
    Assert-Memory '{"d3d12":' 'n/a'
    Assert-Memory $null 'n/a'
    'smoke-memory: ok (8 cases)'
} finally {
    Remove-Item -LiteralPath $report, $log, $xcl -Force -ErrorAction SilentlyContinue
    Remove-Item -LiteralPath $fixture -Force
}
