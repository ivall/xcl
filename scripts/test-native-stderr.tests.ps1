# Run with powershell.exe or pwsh.exe. The fixture writes stderr through a native process.
$ErrorActionPreference = 'Stop'
$runner = Join-Path $PSScriptRoot 'test.ps1'
$fixture = Join-Path ([IO.Path]::GetTempPath()) ('xcl-native-stderr-' + [guid]::NewGuid())
$null = New-Item -ItemType Directory -Path $fixture
$cargo = Join-Path $fixture 'cargo.cmd'
$log = Join-Path $fixture 'runner.log'
$calls = Join-Path $fixture 'calls.log'
$wrapper = Join-Path $fixture 'run.ps1'
$shell = (Get-Process -Id $PID).Path
$oldPath = $env:PATH
try {
    # Keep redirection inside the child shell to exercise PowerShell 5.1's native stderr handling.
    [IO.File]::WriteAllText($wrapper, @'
param($Runner, $Log)
& $Runner *> $Log
exit $LASTEXITCODE
'@)
    $nativeCargo = @'
@echo off
echo %1>>"%~dp0calls.log"
if "%1"=="build" (
    >&2 echo synthetic build progress
    exit /b {0}
)
if "%1"=="test" (
    >&2 echo SKIPPED native_stderr: synthetic reason
    exit /b {1}
)
exit /b 127
'@
    $env:PATH = $fixture
    $cases = @(
        @{ Name = 'success'; Build = 0; Test = 0; Exit = 0; Calls = 'build,test' }
        @{ Name = 'build failure'; Build = 17; Test = 0; Exit = 17; Calls = 'build' }
        @{ Name = 'test failure'; Build = 0; Test = 101; Exit = 101; Calls = 'build,test' }
        @{ Name = 'missing Cargo'; Build = 0; Test = 0; Exit = 1; Calls = '' }
    )
    foreach ($case in $cases) {
        [IO.File]::WriteAllText($calls, '')
        [IO.File]::WriteAllText($log, '')
        if ($case.Name -eq 'missing Cargo') {
            Remove-Item -LiteralPath $cargo
        } else {
            [IO.File]::WriteAllText($cargo, ($nativeCargo -f $case.Build, $case.Test), [Text.Encoding]::ASCII)
            if ((Get-Command cargo).Path -ne $cargo) { throw 'The native Cargo fixture was not selected.' }
        }

        # A failing child may write stderr too; do not let the parent turn it into a terminating error.
        $ErrorActionPreference = 'Continue'
        try {
            $shellOutput = & $shell -NoProfile -NonInteractive -File $wrapper $runner $log 2>&1
            $code = $LASTEXITCODE
        } finally {
            $ErrorActionPreference = 'Stop'
        }
        $output = [IO.File]::ReadAllText($log)
        $diagnostics = $output + "`n" + ($shellOutput -join "`n")
        if ($code -ne $case.Exit) {
            throw "$($case.Name): expected exit $($case.Exit), got $code.`n$diagnostics"
        }
        $actualCalls = [IO.File]::ReadAllLines($calls) -join ','
        if ($actualCalls -ne $case.Calls) {
            throw "$($case.Name): expected Cargo calls '$($case.Calls)', got '$actualCalls'."
        }
        if ($case.Calls -and !$output.Contains('synthetic build progress')) {
            throw "$($case.Name): missing native build output in the redirected log."
        }
        if ($case.Calls -eq 'build,test' -and
            !$output.Contains('GPU tests skipped on this adapter: 1 (synthetic reason)')) {
            throw "$($case.Name): missing GPU skip summary in the redirected log."
        }
        if (!$case.Calls -and !$diagnostics.Contains('cargo')) {
            throw 'Missing Cargo did not produce a diagnostic naming cargo.'
        }
        Write-Host "PASS: $($case.Name)"
    }
} finally {
    $env:PATH = $oldPath
    foreach ($path in $cargo, $log, $calls, $wrapper, $fixture) {
        if (Test-Path -LiteralPath $path) { Remove-Item -LiteralPath $path }
    }
}
# Expected failing children must not leave a failure code for CI's PowerShell wrapper.
exit 0
