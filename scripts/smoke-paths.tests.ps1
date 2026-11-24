# Run with powershell.exe or pwsh.exe. No build or game files are required.
$ErrorActionPreference = 'Stop'
$smoke = Join-Path $PSScriptRoot 'smoke.ps1'
$fixture = Join-Path ([IO.Path]::GetTempPath()) ('xcl-smoke-paths-' + [guid]::NewGuid())
$bracketed = Join-Path $fixture '[xcl]'
$null = New-Item -ItemType Directory -Path $bracketed
$xcl = Join-Path $bracketed 'xcl.cmd'
$log = Join-Path $bracketed 'xcl.log'
$report = Join-Path $bracketed 'report.json'
$network, $seconds, $activity = $env:XCL_NETWORK, $env:XCL_RUN_SECONDS, $env:XCL_ACTIVITY
try {
    @'
@echo off
echo Log: %~dp0xcl.log
'@ | Set-Content -LiteralPath $xcl -Encoding Ascii
    '(7 frames shown, 8 PresentX calls)' | Set-Content -LiteralPath $log -Encoding Ascii
    '{"d3d12":{"adapter":"XCL test adapter (64 MiB VRAM)"}}' | Set-Content -LiteralPath $report -Encoding Ascii

    $output = & $smoke -Game $fixture -Seconds 1 -Xcl $xcl
    if ($output -notmatch '(?m)^frames shown: 7 \(PresentX 8\)\r?$') {
        throw "Expected the smoke block to use the log in $bracketed."
    }
    if ($output -notmatch '(?m)^gpu: XCL test adapter\r?$') {
        throw "Expected the smoke block to use the report in $bracketed."
    }
} finally {
    $env:XCL_NETWORK, $env:XCL_RUN_SECONDS, $env:XCL_ACTIVITY = $network, $seconds, $activity
    foreach ($path in $xcl, $log, $report, $bracketed, $fixture) {
        if (Test-Path -LiteralPath $path) { Remove-Item -LiteralPath $path }
    }
}
