# Run with powershell.exe or pwsh.exe. The fixture writes stderr through a native process.
$ErrorActionPreference = 'Stop'
$runner = Join-Path $PSScriptRoot 'test.ps1'
$fixture = Join-Path ([IO.Path]::GetTempPath()) ('xcl-native-stderr-' + [guid]::NewGuid())
$null = New-Item -ItemType Directory -Path $fixture
$cargo = Join-Path $fixture 'cargo.cmd'
$log = Join-Path $fixture 'runner.log'
$calls = Join-Path $fixture 'calls.log'
$oldPath = $env:PATH
Push-Location
try {
    @'
@echo off
echo %*>>"%~dp0calls.log"
if "%1"=="build" (
    >&2 echo synthetic build progress
    exit /b 0
)
if "%1"=="test" (
    >&2 echo SKIPPED native_stderr: synthetic reason
    exit /b 0
)
exit /b 127
'@ | Set-Content -LiteralPath $cargo -Encoding Ascii
    $env:PATH = $fixture
    if ((Get-Command cargo).Path -ne $cargo) { throw 'The native Cargo fixture was not selected.' }

    & $runner --skip synthetic_filter --nocapture *> $log
    if ($LASTEXITCODE -ne 0) { throw "Runner exited $LASTEXITCODE on a successful build." }
    $expectedBuild = 'build --locked --workspace'
    $expectedTest = 'test --locked --workspace -- --test-threads=4 --skip synthetic_filter --nocapture'
    $actual = (Get-Content -LiteralPath $calls) -join "`n"
    if ($actual -cne "$expectedBuild`n$expectedTest") {
        throw "Unexpected Cargo calls.`nExpected:`n$expectedBuild`n$expectedTest`nActual:`n$actual"
    }
    $output = Get-Content -LiteralPath $log -Raw
    if (!$output.Contains('synthetic build progress') -or
        !$output.Contains('GPU tests skipped on this adapter: 1 (synthetic reason)')) {
        throw 'Expected native build output and the GPU skip summary in the redirected log.'
    }
} finally {
    Pop-Location
    $env:PATH = $oldPath
    foreach ($path in $cargo, $log, $calls, $fixture) {
        if (Test-Path -LiteralPath $path) { Remove-Item -LiteralPath $path }
    }
}
