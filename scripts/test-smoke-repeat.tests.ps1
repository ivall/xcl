# Run with powershell.exe or pwsh.exe.
$ErrorActionPreference = 'Stop'
$runner = Join-Path $PSScriptRoot 'smoke.ps1'
$fixture = Join-Path ([IO.Path]::GetTempPath()) ('xcl-smoke-repeat-' + [guid]::NewGuid())
$fixture = (New-Item -ItemType Directory -Path $fixture).FullName
$log = Join-Path $fixture 'source.log'
$launches = Join-Path $fixture 'launches.txt'
$mockXcl = Join-Path $fixture 'mock_xcl.cmd'
$artifacts = Join-Path $fixture 'artifacts'

try {
    '(90 frames shown, 90 PresentX calls)' | Set-Content -LiteralPath $log -Encoding utf8
    '{"d3d12": {"adapter": "Synthetic Adapter"}}' |
        Set-Content -LiteralPath (Join-Path $fixture 'report.json') -Encoding utf8
    @"
@echo off
echo launch>>"$launches"
echo Log: $log
exit /b 0
"@ | Set-Content -LiteralPath $mockXcl -Encoding Ascii

    $output = & $runner -Game $fixture -Xcl $mockXcl -Seconds 5 -Title RepeatTest -OutDir $artifacts -Repeat 3
    if (@(Get-Content -LiteralPath $launches).Count -ne 3) { throw 'Expected three independent launches.' }
    if ([regex]::Matches(($output -join "`n"), '(?m)^xcl-smoke v1\r?$').Count -ne 3) {
        throw 'Expected three complete smoke blocks.'
    }
    foreach ($run in 1..3) {
        $dir = Join-Path $artifacts ('run-{0:d3}' -f $run)
        $manifest = Get-Content -LiteralPath (Join-Path $dir 'manifest.json') -Raw | ConvertFrom-Json
        if ($manifest.title -ne 'RepeatTest' -or $manifest.seconds -ne 5 -or $manifest.exit -ne '0x0') {
            throw "Run $run did not preserve the requested title and duration."
        }
        foreach ($name in 'xcl.log', 'report.json') {
            if (-not (Test-Path -LiteralPath (Join-Path $dir $name))) { throw "Run $run is missing $name." }
        }
    }
    $manifestPaths = 1..3 | ForEach-Object { Join-Path $artifacts ('run-{0:d3}/manifest.json' -f $_) }
    $before = Get-FileHash -LiteralPath $manifestPaths | Select-Object -ExpandProperty Hash
    $reuseError = $null
    try {
        & $runner -Game $fixture -Xcl $mockXcl -Seconds 5 -Title OverwriteTest -OutDir $artifacts -Repeat 3 |
            Out-Null
    } catch {
        $reuseError = $_.Exception.Message
    }
    if ($reuseError -notmatch 'run-001' -or $reuseError -notmatch 'fresh') {
        throw "Expected repeat output directory reuse to fail. Got: $reuseError"
    }
    if (@(Get-Content -LiteralPath $launches).Count -ne 3) { throw 'Reuse must not start another launch.' }
    $after = Get-FileHash -LiteralPath $manifestPaths | Select-Object -ExpandProperty Hash
    if (Compare-Object $before $after -SyncWindow 0) { throw 'Reuse changed a previous manifest.' }

    $partialArtifacts = Join-Path $fixture 'partial-artifacts'
    $null = New-Item -ItemType Directory -Path (Join-Path $partialArtifacts 'run-002') -Force
    $partialError = $null
    try {
        & $runner -Game $fixture -Xcl $mockXcl -Seconds 5 -Title RepeatTest -OutDir $partialArtifacts -Repeat 2 |
            Out-Null
    } catch {
        $partialError = $_.Exception.Message
    }
    if ($partialError -notmatch 'run-002' -or @(Get-Content -LiteralPath $launches).Count -ne 3) {
        throw 'A later occupied destination must stop the entire batch before launching.'
    }

    $launches = Join-Path $fixture 'mixed-launches.txt'
    $firstRun = Join-Path $fixture 'first-run'
    $secondRun = Join-Path $fixture 'second-run'
    $mixedArtifacts = Join-Path $fixture 'mixed-artifacts'
    @"
@echo off
echo launch>>"$launches"
if not exist "$firstRun" goto first
if not exist "$secondRun" goto second
echo Log: $log
exit /b 0
:first
type nul > "$firstRun"
echo Log: $log
exit /b 0
:second
type nul > "$secondRun"
exit /b 3
"@ | Set-Content -LiteralPath $mockXcl -Encoding Ascii

    $blocks = [System.Collections.Generic.List[string]]::new()
    $failure = $null
    try {
        & $runner -Game $fixture -Xcl $mockXcl -Seconds 5 -Title RepeatTest -OutDir $mixedArtifacts -Repeat 3 |
            ForEach-Object { $blocks.Add($_) }
    } catch {
        $failure = $_.Exception.Message
    }
    if (@(Get-Content -LiteralPath $launches).Count -ne 3) { throw 'Expected all three launches after a failure.' }
    if ([regex]::Matches(($blocks -join "`n"), '(?m)^xcl-smoke v1?$').Count -ne 2) {
        throw 'Expected both successful runs to print smoke blocks.'
    }
    if ($failure -notmatch '1 of 3' -or $failure -notmatch 'run-002' -or $failure -notmatch 'no log path') {
        throw "Expected an aggregate error naming the failed run. Got: $failure"
    }
    foreach ($run in 'run-001', 'run-003') {
        $manifest = Get-Content -LiteralPath (Join-Path $mixedArtifacts "$run/manifest.json") -Raw | ConvertFrom-Json
        if ($manifest.exit -ne '0x0') { throw "Missing successful manifest for $run." }
    }
    if ((Get-Content -LiteralPath (Join-Path $mixedArtifacts 'run-002/error.txt') -Raw) -notmatch 'no log path') {
        throw 'Expected the failed run to retain its diagnostic.'
    }
    Write-Host 'smoke-repeat: ok'
} finally {
    if ((Resolve-Path -LiteralPath $fixture).Path -eq $fixture) {
        Remove-Item -LiteralPath $fixture -Recurse -Force
    }
}
