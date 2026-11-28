# Run in PowerShell 5.1 or 7. Simulate a test run on a machine without access to CIM.
param([string]$Smoke = (Join-Path $PSScriptRoot 'smoke.ps1'))
$ErrorActionPreference = 'Stop'
$fixture = Join-Path ([IO.Path]::GetTempPath()) ('xcl-smoke-adapter-' + [guid]::NewGuid())
$null = New-Item -ItemType Directory -Path $fixture
$xcl = Join-Path $fixture 'xcl.cmd'
$log = Join-Path $fixture 'xcl.log'
$report = Join-Path $fixture 'report.json'
$artifacts = Join-Path $fixture 'artifacts'

function Get-CimInstance {
    [CmdletBinding()]
    param([Parameter(Position = 0)][string]$ClassName)
    if ($cimMode -eq 'unavailable') { throw 'Fixture: CIM is unavailable' }
    if ($cimMode -eq 'empty') { return }
    if ($cimMode -eq 'available') {
        [pscustomobject]@{ Name = $null }
        [pscustomobject]@{ Name = '   ' }
    }
    [pscustomobject]@{ Name = 'Microsoft Basic Display Adapter' }
    if ($cimMode -eq 'available') { [pscustomobject]@{ Name = 'Fixture Controller' } }
}

function Test-SmokeAdapter([string]$Expected, [switch]$SaveArtifacts) {
    $options = @{}
    if ($SaveArtifacts) { $options.OutDir = $artifacts }
    $block = & $Smoke -Game 'D:\Synthetic' -Seconds 2 -Xcl $xcl -Title 'fixture' @options
    if ($block -notmatch ('(?m)^gpu: ' + [regex]::Escape($Expected) + '\r?$')) {
        throw "Smoke did not report '$Expected'"
    }
    if ($block -notmatch '(?m)^device removals: 1\r?$' -or $block -notmatch '(?m)^crashes: 1\r?$' -or
        $block -notmatch '(?m)^peak commit: 512 MB\r?$') {
        throw 'Smoke lost the failure or memory diagnostics'
    }
    if ($SaveArtifacts) {
        $manifest = Get-Content -LiteralPath (Join-Path $artifacts 'manifest.json') -Raw -Encoding utf8 | ConvertFrom-Json
        if ($manifest.gpu -ne $Expected -or $manifest.removals -ne 1 -or $manifest.crashes -ne 1 -or
            $manifest.peak_commit_mb -ne 512 -or -not (Test-Path -LiteralPath (Join-Path $artifacts 'xcl.log'))) {
            throw 'Smoke lost the failure diagnostics in the saved artifacts'
        }
    }
}

try {
    [IO.File]::WriteAllText($xcl, "@echo off`r`necho Log: %~dp0xcl.log`r`n")
    [IO.File]::WriteAllLines($log, @('memory: 512 MB committed', 'device removed', 'unhandled exception'))

    $cimMode = 'unavailable'
    Test-SmokeAdapter 'unknown' -SaveArtifacts
    $cimMode = 'empty'
    Test-SmokeAdapter 'unknown'
    $cimMode = 'basic'
    Test-SmokeAdapter 'unknown'
    $cimMode = 'available'
    Test-SmokeAdapter 'Fixture Controller'

    [IO.File]::WriteAllText($report, '{"d3d12":{"adapter":"   "}}')
    Test-SmokeAdapter 'Fixture Controller'
    $cimMode = 'unavailable'
    Test-SmokeAdapter 'unknown'

    $cimMode = 'available'
    [IO.File]::WriteAllText($report, '{"d3d12":{"adapter":"Report GPU (4096 MiB VRAM)"}}')
    Test-SmokeAdapter 'Report GPU'
    'smoke-adapter: ok'
} finally {
    if (Test-Path -LiteralPath $artifacts) {
        [IO.File]::Delete((Join-Path $artifacts 'manifest.json'))
        [IO.File]::Delete((Join-Path $artifacts 'xcl.log'))
        [IO.Directory]::Delete($artifacts)
    }
    foreach ($path in $xcl,$log,$report) { [IO.File]::Delete($path) }
    [IO.Directory]::Delete($fixture)
}
