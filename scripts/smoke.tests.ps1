# Run with powershell.exe or pwsh.exe. A stand-in for xcl.exe prints the path of a log written here.
$ErrorActionPreference = 'Stop'
$smoke = Join-Path $PSScriptRoot 'smoke.ps1'
$fixture = Join-Path ([IO.Path]::GetTempPath()) ('xcl-smoke-' + [guid]::NewGuid())
$null = New-Item -ItemType Directory -Path $fixture
$xcl = Join-Path $fixture 'xcl.cmd'
$log = Join-Path $fixture 'xcl.log'
$report = Join-Path $fixture 'report.json'
$game = Join-Path $fixture 'SomeTitle'
$null = New-Item -ItemType Directory -Path $game
function Get-Block { ((& $smoke -Game $game -Seconds 1 -Xcl $xcl) -join "`n") -replace "`r", '' }
function Assert-Line($block, $line) { if (!$block.Contains("$line`n")) { throw "Expected '$line' in the block:`n$block" } }
try {
    "@echo off`r`necho Log:    $log" | Set-Content -LiteralPath $xcl -Encoding Ascii
    @(
        'xcl: activity (2 s): 60 presents, 240 submits'
        'xcl: memory: 1234 MB committed by the process'
        'xcl: timed run over (118 frames shown, 120 PresentX calls)'
    ) | Set-Content -LiteralPath $log -Encoding utf8
    $first = (Get-CimInstance Win32_VideoController | Where-Object { $_.Name -notmatch 'Basic' } | Select-Object -First 1).Name

    # a run on a machine's second adapter: the block names the one in the run's report
    '{"d3d12":{"adapter":"Synthetic Adapter 9000 (32187 MiB VRAM)"}}' | Set-Content -LiteralPath $report -Encoding utf8
    $block = Get-Block
    foreach ($line in 'gpu: Synthetic Adapter 9000', 'title: SomeTitle', 'seconds: 1', 'exit: 0x0',
        'frames shown: 118 (PresentX 120)', 'presents per 2 s (last 3): 60', 'device removals: 0', 'crashes: 0',
        'peak commit: 1234 MB') {
        Assert-Line $block $line
    }

    # a report without an adapter (the null backend), and no report at all: the first video controller
    '{"d3d12":{"adapter":null}}' | Set-Content -LiteralPath $report -Encoding utf8
    Assert-Line (Get-Block) "gpu: $first"
    Remove-Item -LiteralPath $report
    Assert-Line (Get-Block) "gpu: $first"

    # a crash and a device removal in the log are counted
    Add-Content -LiteralPath $log -Value 'xcl: D3D12.X: device removed (0x887a0006)', 'xcl: unhandled exception 0xc0000005' -Encoding utf8
    $block = Get-Block
    Assert-Line $block 'device removals: 1'
    Assert-Line $block 'crashes: 1'
} finally {
    foreach ($path in $xcl, $log, $report, $game, $fixture) {
        if (Test-Path -LiteralPath $path) { Remove-Item -LiteralPath $path }
    }
}
