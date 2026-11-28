# Run with powershell.exe or pwsh.exe; no Rust toolchain or GPU is needed.
$ErrorActionPreference = 'Stop'
$runner = Join-Path $PSScriptRoot 'test.ps1'
$fixture = Join-Path ([IO.Path]::GetTempPath()) ('xcl-test-totals-' + [guid]::NewGuid())
$null = New-Item -ItemType Directory -Path $fixture
$cargo = Join-Path $fixture 'cargo.cmd'
$inputLog = Join-Path $fixture 'input.txt'
$outputLog = Join-Path $fixture 'output.txt'
$oldPath = $env:PATH
Push-Location

function Check-Run($name, $lines, $buildExit, $testExit, $summary, $skipSummary = '') {
    $command = @"
@echo off
if "%1"=="build" exit /b $buildExit
if not "%1"=="test" exit /b 127
echo forwarded: %*
type "%~dp0input.txt"
exit /b $testExit
"@
    [IO.File]::WriteAllText($cargo, $command, [Text.Encoding]::ASCII)
    [IO.File]::WriteAllText($inputLog, ($lines -join "`r`n") + "`r`n", [Text.Encoding]::ASCII)
    if ((Get-Command cargo).Path -ne $cargo) { throw 'The Cargo fixture was not selected.' }
    & $runner --skip omitted *> $outputLog
    $expectedExit = if ($buildExit) { $buildExit } else { $testExit }
    if ($LASTEXITCODE -ne $expectedExit) { throw "${name}: expected exit $expectedExit, got $LASTEXITCODE" }
    $output = Get-Content -LiteralPath $outputLog -Raw
    if (-not $buildExit) {
        $plainOutput = $output -replace '\x1B\[[0-?]*[ -/]*[@-~]', ''
        foreach ($line in $lines) {
            if (!$plainOutput.Contains(($line -replace '\x1B\[[0-?]*[ -/]*[@-~]', ''))) { throw "${name}: lost source output: $line" }
        }
    }
    $found = @($output -split '\r?\n' | Where-Object { $_ -like 'Completed test suites:*' })
    if ($summary) {
        if ($found.Count -ne 1 -or $found[0] -ne $summary) { throw "${name}: incorrect summary: $($found -join ', ')" }
    } elseif ($found.Count) { throw "${name}: invented totals without a completed suite" }
    if ($skipSummary -and !$output.Contains($skipSummary)) { throw "${name}: lost the skip summary:`n$output" }
    if ($buildExit) {
        if ($output.Contains('forwarded:')) { throw "${name}: ran tests after a failed build" }
    } elseif ($output -cnotmatch '(?m)^forwarded: test (?:--locked )?--workspace -- --test-threads=4 --skip omitted\r?$') {
        throw "${name}: changed the forwarded test arguments"
    }
    Write-Host "PASS $name"
}

try {
    $env:PATH = $fixture
    Check-Run 'multiple suites and skips' @(
        'SKIPPED gpu_fixture: no wave64',
        'test result: ok. 5 passed; 0 failed; 1 ignored; 0 measured; 0 filtered out; finished in 0.01s',
        'diagnostic: test result: ok. 999 passed; 0 failed; 0 ignored;',
        "$([char]27)[32mtest result: ok. 3 passed; 0 failed; 2 ignored; 0 measured; 5 filtered out; finished in 0.01s$([char]27)[0m"
    ) 0 0 'Completed test suites: 2; reported totals: 8 passed, 0 failed, 3 ignored.' 'GPU tests skipped on this adapter: 1 (no wave64)'
    Check-Run 'failed suite keeps the Cargo exit code' @(
        'test result: ok. 2 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.01s',
        'test result: FAILED. 4 passed; 1 failed; 2 ignored; 0 measured; 0 filtered out; finished in 0.01s'
    ) 0 101 'Completed test suites: 2; reported totals: 6 passed, 1 failed, 2 ignored.'
    Check-Run 'failure before any summary' @('running 3 tests', 'process aborted') 0 101 ''
    Check-Run 'empty completed suite' @(
        'test result: ok. 0 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s'
    ) 0 0 'Completed test suites: 1; reported totals: 0 passed, 0 failed, 0 ignored.'
    Check-Run 'build failure does not run tests' @() 7 0 ''
    Check-Run 'interrupted suite is not counted' @(
        'test result: ok. 2 passed; 0 failed; 1 ignored; 0 measured; 0 filtered out; finished in 0.01s',
        'running 5 tests', 'test unfinished ...'
    ) 0 101 'Completed test suites: 1; reported totals: 2 passed, 0 failed, 1 ignored.'
    Check-Run 'standalone and inline skips keep their reasons' @(
        'SKIPPED gpu_one: no wave64',
        'test gpu_two ... SKIPPED gpu_two: no mesh shaders',
        "$([char]27)[33mtest gpu_three ... SKIPPED gpu_three: no mesh shaders$([char]27)[0m",
        'diagnostic: SKIPPED not_a_test: no mesh shaders',
        'test result: ok. 3 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.01s'
    ) 0 0 'Completed test suites: 1; reported totals: 3 passed, 0 failed, 0 ignored.' 'GPU tests skipped on this adapter: 2 (no mesh shaders), 1 (no wave64)'
} finally {
    Pop-Location
    $env:PATH = $oldPath
    foreach ($path in $cargo, $inputLog, $outputLog, $fixture) {
        if (Test-Path -LiteralPath $path) { Remove-Item -LiteralPath $path }
    }
}
