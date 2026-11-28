# Run with powershell.exe or pwsh.exe. Check both sides of the smoke runner's process boundary.
$ErrorActionPreference = 'Stop'
$smoke = Join-Path $PSScriptRoot 'smoke.ps1'
$fixture = Join-Path ([IO.Path]::GetTempPath()) ('xcl-smoke-env-' + [guid]::NewGuid())
$null = New-Item -ItemType Directory -Path $fixture
$xcl = Join-Path $fixture 'xcl.cmd'
$log = Join-Path $fixture 'xcl.log'
$report = Join-Path $fixture 'report.json'
$network, $seconds, $activity = $env:XCL_NETWORK, $env:XCL_RUN_SECONDS, $env:XCL_ACTIVITY
try {
    @'
@echo off
>"%~dp0xcl.log" echo %XCL_NETWORK% %XCL_RUN_SECONDS% %XCL_ACTIVITY%
echo Log: %~dp0xcl.log
'@ | Set-Content -LiteralPath $xcl -Encoding Ascii
    '{"d3d12":{"adapter":"fixture"}}' | Set-Content -LiteralPath $report -Encoding Ascii

    foreach ($initial in '17', $null) {
        $env:XCL_NETWORK = $env:XCL_RUN_SECONDS = $env:XCL_ACTIVITY = $initial
        $null = & $smoke -Game $fixture -Seconds 2 -Xcl $xcl
        if ((Get-Content -LiteralPath $log -Raw).Trim() -ne '0 2 1') {
            throw 'The child did not receive the smoke run settings.'
        }
        foreach ($name in 'XCL_NETWORK', 'XCL_RUN_SECONDS', 'XCL_ACTIVITY') {
            if ([Environment]::GetEnvironmentVariable($name) -cne $initial) {
                throw "The smoke run changed $name in the calling shell."
            }
        }
    }
    # An existing directory passes Test-Path but cannot be launched as an executable.
    try {
        $null = & $smoke -Game $fixture -Seconds 2 -Xcl $fixture
        throw 'Expected Process.Start to fail.'
    } catch [System.Management.Automation.MethodInvocationException] {
        if ($null -ne $env:XCL_NETWORK -or $null -ne $env:XCL_RUN_SECONDS -or $null -ne $env:XCL_ACTIVITY) {
            throw 'The failed launch left smoke settings in the calling shell.'
        }
    }
} finally {
    $env:XCL_NETWORK, $env:XCL_RUN_SECONDS, $env:XCL_ACTIVITY = $network, $seconds, $activity
    foreach ($path in $xcl, $log, $report, $fixture) {
        if (Test-Path -LiteralPath $path) { Remove-Item -LiteralPath $path }
    }
}
