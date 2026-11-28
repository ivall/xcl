# Run with powershell.exe or pwsh.exe. No Rust build, GPU or game files needed.
# A run that smoke.ps1 kills on timeout never prints `Log:` (xcl run prints it when it ends), so the
# block has to come from the log in the `data:` folder XCL printed at the start.
param([string]$SmokeScript = (Join-Path $PSScriptRoot 'smoke.ps1'))
$ErrorActionPreference = 'Stop'
$fixture = Join-Path (Split-Path $PSScriptRoot -Parent) ('target\smoke-timeout-log-' + [guid]::NewGuid())
$null = New-Item -ItemType Directory -Path (Join-Path $fixture 'scripts') -Force
$shell = (Get-Process -Id $PID).Path
try {
    # Exercise the script itself, shortening only its 120-second grace period in a private copy.
    $source = [IO.File]::ReadAllText($SmokeScript)
    $deadline = '($Seconds + 120) * 1000'
    if (-not $source.Contains($deadline)) { throw 'Smoke timeout expression changed; update the fixture.' }
    $runner = Join-Path $fixture 'scripts\smoke.ps1'
    [IO.File]::WriteAllText($runner, $source.Replace($deadline, '3000'), [Text.UTF8Encoding]::new($false))

    # A stand-in for `xcl run` that hangs after its start lines, as a title that stops responding does.
    $cs = Join-Path $fixture 'fixture.cs'
    $exe = Join-Path $fixture 'fixture.exe'
    [IO.File]::WriteAllText($cs, @'
using System;
using System.Threading;
class Fixture {
    static int Main(string[] args) {
        Console.WriteLine("XCL 0.0.0 - starting 'synthetic'");
        Console.WriteLine("  exe:   synthetic.exe");
        Console.WriteLine("  data:  " + Environment.GetEnvironmentVariable("XCL_TEST_DATA"));
        Console.Out.Flush();
        Thread.Sleep(60000);
        return 0;
    }
}
'@)
    $compiler = Join-Path $env:SystemRoot 'Microsoft.NET\Framework64\v4.0.30319\csc.exe'
    & $compiler /nologo /target:exe "/out:$exe" $cs
    if ($LASTEXITCODE -ne 0) { throw 'Could not compile the process fixture.' }

    $data = Join-Path $fixture 'data'
    $null = New-Item -ItemType Directory -Path (Join-Path $data 'logs') -Force
    [IO.File]::WriteAllText((Join-Path $data 'logs\xcl.log'), @'
[    2.0][1][INFO ] activity (2 s): 30 presents, 120 submits, 90 draws, 0 dispatches
[    2.0][1][INFO ] memory: 1234 MB committed by the process
[    4.0][1][INFO ] activity (2 s): 60 presents, 240 submits, 780 draws, 60 dispatches
[    4.0][1][INFO ] memory: 2345 MB committed by the process
'@)
    $game = Join-Path $fixture 'game'
    $null = New-Item -ItemType Directory -Path $game

    $psi = New-Object System.Diagnostics.ProcessStartInfo
    $psi.FileName = $shell
    $psi.Arguments = '-NoProfile -NonInteractive -ExecutionPolicy Bypass -File "' + $runner + '" -Xcl "' + $exe + '" -Game "' + $game + '" -Seconds 1'
    $psi.UseShellExecute = $false
    $psi.CreateNoWindow = $true
    $psi.RedirectStandardOutput = $true
    $psi.RedirectStandardError = $true
    $psi.EnvironmentVariables['XCL_TEST_DATA'] = $data
    $p = [System.Diagnostics.Process]::Start($psi)
    $stdout = $p.StandardOutput.ReadToEndAsync()
    $stderr = $p.StandardError.ReadToEndAsync()
    if (-not $p.WaitForExit(60000)) { $p.Kill(); throw 'smoke.ps1 did not finish within 60 s' }
    $out = $stdout.Result + $stderr.Result
    if ($p.ExitCode -ne 0) { throw "smoke.ps1 failed (exit $($p.ExitCode)):`n$out" }
    foreach ($want in 'exit: killed (no exit)', 'presents per 2 s (last 3): 30 60', 'peak commit: 2345 MB') {
        if (-not $out.Contains($want)) { throw "missing '$want' in:`n$out" }
    }
    'smoke timeout log: 1 passed'
} finally {
    Remove-Item -LiteralPath $fixture -Recurse -Force -ErrorAction SilentlyContinue
}
