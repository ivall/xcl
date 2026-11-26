# Run with powershell.exe or pwsh.exe. A stand-in records the forwarded Cargo command.
$ErrorActionPreference = 'Stop'
$runner = Join-Path $PSScriptRoot 'test.ps1'
$temp = (Resolve-Path -LiteralPath ([IO.Path]::GetTempPath())).Path
$fixture = Join-Path $temp ('xcl-test-threads-' + [guid]::NewGuid())
$null = New-Item -ItemType Directory -Path $fixture
$oldPath = $env:PATH
$oldTestThreads = $env:RUST_TEST_THREADS
Push-Location
try {
    $cargo = Join-Path $fixture 'cargo.cmd'
    @'
@echo off
if "%1"=="test" (
    echo forwarded: %*
    echo test-thread-default: %RUST_TEST_THREADS%
)
exit /b 0
'@ | Set-Content -LiteralPath $cargo -Encoding Ascii
    $env:PATH = $fixture
    if ((Get-Command cargo).Path -ne $cargo) { throw 'The native Cargo fixture was not selected.' }

    $cases = @(
        @{ Arguments = @('--test-threads=1', '--skip', 'omitted'); Expected = '--test-threads=1 --skip omitted'; Initial = '9' },
        @{ Arguments = @('--test-threads', '2'); Expected = '--test-threads 2'; Initial = $null },
        @{ Arguments = @('--skip', '--test-threads=1'); Expected = '--skip --test-threads=1'; Initial = '9' },
        @{ Arguments = @('--', '--test-threads=1'); Expected = '-- --test-threads=1'; Initial = $null }
    )
    foreach ($case in $cases) {
        $arguments = $case.Arguments
        $env:RUST_TEST_THREADS = $case.Initial
        $output = & $runner @arguments *>&1
        if ($LASTEXITCODE -ne 0) { throw "Runner exited $LASTEXITCODE." }
        if ($env:RUST_TEST_THREADS -ne $case.Initial) { throw 'The caller test-thread setting was not restored.' }
        if (($output -join "`n") -notmatch 'test-thread-default: 4') { throw 'The test process did not inherit the four-thread default.' }
        $forwarded = $output | Where-Object { "$_" -like 'forwarded:*' }
        $expected = "forwarded: test --workspace -- $($case.Expected)"
        if ("$forwarded" -ne $expected) { throw "Expected '$expected', got '$forwarded'." }
    }
    'test-threads: ok'
} finally {
    Pop-Location
    $env:PATH = $oldPath
    $env:RUST_TEST_THREADS = $oldTestThreads
    $resolved = (Resolve-Path -LiteralPath $fixture).Path
    if ((Split-Path -Parent $resolved) -eq $temp) { Remove-Item -LiteralPath $resolved -Recurse -Force }
}
