# Run with powershell.exe or pwsh.exe. No Rust build, GPU or game files needed.
param([string]$SmokeScript = (Join-Path $PSScriptRoot 'smoke.ps1'))
$ErrorActionPreference = 'Stop'
$fixture = Join-Path (Split-Path $PSScriptRoot -Parent) ('target\smoke-timeout-' + [guid]::NewGuid())
$null = New-Item -ItemType Directory -Path (Join-Path $fixture 'scripts') -Force
$shell = (Get-Process -Id $PID).Path
$processes = @()
function Stop-FixtureProcesses($directory) {
    foreach ($pidPath in (Get-ChildItem -LiteralPath $directory -Filter '*.pid' -Recurse)) {
        $leftover = Get-Process -Id ([int][IO.File]::ReadAllText($pidPath.FullName)) -ErrorAction SilentlyContinue
        if ($leftover -and $leftover.Path -eq $exe) { $leftover.Kill(); $leftover.WaitForExit(); $leftover.Dispose() }
    }
}
try {
    # Exercise the script itself, shortening only its 120-second grace period in a private copy.
    $source = [IO.File]::ReadAllText($SmokeScript)
    $deadline = '($Seconds + 120) * 1000'
    if (-not $source.Contains($deadline)) { throw 'Smoke timeout expression changed; update the fixture.' }
    $runner = Join-Path $fixture 'scripts\smoke.ps1'
    [IO.File]::WriteAllText($runner, $source.Replace($deadline, '5000'), [Text.UTF8Encoding]::new($false))
    $cs = Join-Path $fixture 'fixture.cs'
    $exe = Join-Path $fixture 'fixture.exe'
    [IO.File]::WriteAllText($cs, @'
using System;
using System.Diagnostics;
using System.IO;
using System.Reflection;
using System.Threading;
class Fixture {
    static int Main(string[] args) {
        if (Path.GetFileName(Assembly.GetExecutingAssembly().Location) == "taskkill.exe") {
            if (Environment.GetEnvironmentVariable("XCL_TEST_TASKKILL") == "failure") return 7;
            Process.GetProcessById(int.Parse(args[1])).Kill(); // leave the inherited pipes open
            return 0;
        }
        string dir = args[1];
        if (args[0] == "child") {
            File.WriteAllText(Path.Combine(dir, "child.pid"), Process.GetCurrentProcess().Id.ToString());
            Thread.Sleep(60000);
            return 0;
        }
        File.WriteAllText(Path.Combine(dir, "parent.pid"), Process.GetCurrentProcess().Id.ToString());
        Console.WriteLine("Log: " + Path.Combine(dir, "xcl.log"));
        Console.Error.WriteLine("synthetic title stderr");
        if (Path.GetFileName(dir) == "normal") return 0;
        var child = Process.Start(new ProcessStartInfo {
            FileName = Assembly.GetExecutingAssembly().Location,
            Arguments = "child \"" + dir + "\"",
            UseShellExecute = false,
            CreateNoWindow = true
        });
        child.WaitForExit();
        return 0;
    }
}
'@)
    $compiler = Join-Path $env:SystemRoot 'Microsoft.NET\Framework64\v4.0.30319\csc.exe'
    & $compiler /nologo /target:exe "/out:$exe" $cs
    if ($LASTEXITCODE -ne 0) { throw 'Could not compile the native process fixture.' }
    $fakeWindows = Join-Path $fixture 'fake-windows'
    $null = New-Item -ItemType Directory -Path (Join-Path $fakeWindows 'System32')
    [IO.File]::Copy($exe, (Join-Path $fakeWindows 'System32\taskkill.exe'))
    foreach ($case in 'normal', 'timeout', 'failure', 'open-pipes') {
        $game = Join-Path $fixture $case
        $null = New-Item -ItemType Directory -Path $game
        [IO.File]::WriteAllText((Join-Path $game 'xcl.log'), "synthetic log`n")
        [IO.File]::WriteAllText((Join-Path $game 'report.json'), '{"d3d12":{"adapter":"synthetic adapter"}}')
        $caseRunner = $runner
        if ($case -eq 'failure' -or $case -eq 'open-pipes') {
            # Inject failure only at the taskkill boundary in this private copy.
            $caseRunner = Join-Path $fixture "scripts\$case.ps1"
            $killPath = '"$env:SystemRoot\System32\taskkill.exe"'
            if (-not $source.Contains($killPath)) { throw 'taskkill invocation changed; update the fixture.' }
            $fakeKillPath = '"' + (Join-Path $fakeWindows 'System32\taskkill.exe') + '"'
            [IO.File]::WriteAllText($caseRunner, $source.Replace($deadline, '5000').Replace($killPath, $fakeKillPath), [Text.UTF8Encoding]::new($false))
        }
        $psi = New-Object System.Diagnostics.ProcessStartInfo
        $psi.FileName = $shell
        $psi.Arguments = '-NoProfile -NonInteractive -ExecutionPolicy Bypass -File "' + $caseRunner + '" -Xcl "' + $exe + '" -Game "' + $game + '" -Seconds 1'
        $psi.UseShellExecute = $false
        $psi.CreateNoWindow = $true
        $psi.RedirectStandardOutput = $true
        $psi.RedirectStandardError = $true
        if ($case -eq 'failure' -or $case -eq 'open-pipes') {
            $psi.EnvironmentVariables['XCL_TEST_TASKKILL'] = $case
        }
        $p = New-Object System.Diagnostics.Process
        $p.StartInfo = $psi
        $processes += $p
        $null = $p.Start()
        $stdout = $p.StandardOutput.ReadToEndAsync()
        $stderr = $p.StandardError.ReadToEndAsync()
        if (-not $p.WaitForExit(30000)) { throw "$case`: smoke.ps1 did not return after the deadline." }
        # Injected kill failures deliberately leave fixtures alive; close their inherited handles
        # after verifying that the smoke script itself returned, before collecting its diagnostics.
        if ($case -eq 'failure' -or $case -eq 'open-pipes') { Stop-FixtureProcesses $game }
        if (-not $stdout.Wait(1000) -or -not $stderr.Wait(1000)) { throw "$case`: output pipes stayed open." }
        if ($case -eq 'failure' -or $case -eq 'open-pipes') {
            $message = if ($case -eq 'failure') { 'could not terminate the XCL process tree' } else { 'output pipes did not close' }
            if ($p.ExitCode -eq 0 -or -not $stderr.Result.Contains($message)) {
                throw "$case`: expected a bounded failure, got exit $($p.ExitCode): $($stderr.Result)"
            }
            Write-Host "PASS $case"
            continue
        }
        if ($p.ExitCode -ne 0) { throw "$case`: smoke.ps1 exited $($p.ExitCode): $($stderr.Result)" }
        $expected = if ($case -eq 'timeout') { 'exit: killed (no exit)' } else { 'exit: 0x0' }
        if (-not $stdout.Result.Contains($expected)) { throw "$case`: expected '$expected', got $($stdout.Result)" }
        foreach ($name in 'parent', 'child') {
            $pidPath = Join-Path $game "$name.pid"
            if (-not (Test-Path -LiteralPath $pidPath)) {
                if ($name -eq 'parent' -or $case -eq 'timeout') { throw "$case`: $name never started." }
                continue
            }
            $alive = Get-Process -Id ([int][IO.File]::ReadAllText($pidPath)) -ErrorAction SilentlyContinue
            if ($alive) { throw "$case`: $name survived the smoke run." }
        }
        Write-Host "PASS $case"
    }
} finally {
    # A failing baseline leaves the title alive after its parent has gone: clean it up by PID.
    Stop-FixtureProcesses $fixture
    foreach ($p in $processes) {
        if (-not $p.HasExited) { $p.Kill(); $p.WaitForExit() }
        $p.Dispose()
    }
    $target = [IO.Path]::GetFullPath((Join-Path (Split-Path $PSScriptRoot -Parent) 'target')) + '\'
    if (-not [IO.Path]::GetFullPath($fixture).StartsWith($target, [StringComparison]::OrdinalIgnoreCase)) {
        throw 'Refusing to clean a fixture outside the target directory.'
    }
    Remove-Item -LiteralPath $fixture -Recurse -Force
}
