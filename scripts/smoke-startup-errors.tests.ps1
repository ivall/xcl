# A stand-in for xcl.exe exits before printing a log path.
param([string]$Smoke = (Join-Path $PSScriptRoot 'smoke.ps1'))
$ErrorActionPreference = 'Stop'
$fixture = Join-Path ([IO.Path]::GetTempPath()) ('xcl-smoke-error-' + [guid]::NewGuid())
$null = New-Item -ItemType Directory -Path $fixture
$xcl = Join-Path $fixture 'xcl.cmd'
$network, $seconds, $activity = $env:XCL_NETWORK, $env:XCL_RUN_SECONDS, $env:XCL_ACTIVITY
try {
    @'
@echo off
echo synthetic startup output
>&2 echo xcl: synthetic startup failure
exit /b 2
'@ | Set-Content -LiteralPath $xcl -Encoding Ascii
    $diagnostic = ''
    try {
        $null = & $Smoke -Game $fixture -Seconds 1 -Xcl $xcl
    } catch {
        $diagnostic = $_.Exception.Message
    }
    foreach ($expected in 'XCL printed no log path', 'exit: 0x2', 'synthetic startup output', 'xcl: synthetic startup failure') {
        if (!$diagnostic.Contains($expected)) {
            throw "Expected '$expected' in the failure diagnostic. Got '$diagnostic'."
        }
    }
    Write-Host 'Passed: early failure reports the exit code, stdout and stderr.'
} finally {
    $env:XCL_NETWORK, $env:XCL_RUN_SECONDS, $env:XCL_ACTIVITY = $network, $seconds, $activity
    Remove-Item -LiteralPath $fixture -Recurse -Force
}
