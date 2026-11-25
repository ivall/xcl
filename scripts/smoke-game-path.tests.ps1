# A native fixture records the argv that smoke.ps1 passes to xcl.exe.
param([string]$Smoke = (Join-Path $PSScriptRoot 'smoke.ps1'))
$ErrorActionPreference = 'Stop'
$fixture = Join-Path ([IO.Path]::GetTempPath()) ('xcl-smoke-path-' + [guid]::NewGuid())
$null = New-Item -ItemType Directory -Path $fixture
$source = Join-Path $fixture 'fixture.cs'
$xcl = Join-Path $fixture 'xcl.exe'
$game = Join-Path $fixture 'Game With Spaces'
$null = New-Item -ItemType Directory -Path $game
$network, $seconds, $activity = $env:XCL_NETWORK, $env:XCL_RUN_SECONDS, $env:XCL_ACTIVITY
try {
    @'
using System;
using System.IO;

static class Fixture
{
    static void Main(string[] args)
    {
        string dir = AppDomain.CurrentDomain.BaseDirectory;
        File.WriteAllLines(Path.Combine(dir, "argv.txt"), args);
        Console.WriteLine("Log: " + Path.Combine(dir, "xcl.log"));
    }
}
'@ | Set-Content -LiteralPath $source -Encoding Ascii
    $compiler = Join-Path $env:SystemRoot 'Microsoft.NET\Framework64\v4.0.30319\csc.exe'
    & $compiler /nologo /target:exe "/out:$xcl" $source
    if ($LASTEXITCODE -ne 0) { throw "Fixture compilation exited $LASTEXITCODE." }
    'xcl: timed run over (118 frames shown, 120 PresentX calls)' |
        Set-Content -LiteralPath (Join-Path $fixture 'xcl.log') -Encoding Ascii
    '{"d3d12":{"adapter":"Synthetic Adapter"}}' |
        Set-Content -LiteralPath (Join-Path $fixture 'report.json') -Encoding Ascii

    foreach ($path in @($game, ($game + '\'), ($game + '\\'))) {
        $block = & $Smoke -Game $path -Seconds 1 -Xcl $xcl
        $received = @(Get-Content -LiteralPath (Join-Path $fixture 'argv.txt'))
        if ($received.Count -ne 2 -or $received[0] -cne 'run' -or $received[1] -cne $path) {
            throw "Expected argv 'run', '$path'; received '$($received -join "', '")'."
        }
        if ($block -notmatch '(?m)^exit: 0x0\r?$') { throw "Expected a successful smoke block." }
    }
    Write-Host 'Passed: paths with spaces and zero, one or two trailing backslashes.'
} finally {
    $env:XCL_NETWORK, $env:XCL_RUN_SECONDS, $env:XCL_ACTIVITY = $network, $seconds, $activity
    Remove-Item -LiteralPath $fixture -Recurse -Force
}
