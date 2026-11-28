# Run in Windows PowerShell 5.1. The stand-in records the exact child arguments and flags.
$ErrorActionPreference = 'Stop'
$launcher = Join-Path $PSScriptRoot 'xcl-launcher.ps1'
$oldLocal = $env:LOCALAPPDATA
$oldHost = $env:XCL_HOST
$root = Join-Path ([IO.Path]::GetTempPath()) ('xcl-launcher-' + [guid]::NewGuid())
$title = Join-Path $root "game with spaces and an apostrophe's"
$exe = Join-Path $root 'stand-in.exe'
$null = [IO.Directory]::CreateDirectory($title)
try {
    $env:LOCALAPPDATA = $root
    $env:XCL_HOST = 'must not reach the child'
    Add-Type -OutputAssembly $exe -OutputType ConsoleApplication -TypeDefinition @'
using System;
using System.IO;
public class LauncherProbe {
    public static void Main(string[] args) {
        File.WriteAllLines("probe.txt", new[] {args[0], args[1],
            Environment.GetEnvironmentVariable("XCL_THREAD_DUMP_AT"),
            Environment.GetEnvironmentVariable("XCL_TRACE_FAULTS"),
            Environment.GetEnvironmentVariable("XCL_HOST")});
    }
}
'@
    & $launcher -Launch -Xcl $exe -Game $title -Preset Debugging -Options @{ XCL_THREAD_DUMP_AT = '5,10' }
    if ($LASTEXITCODE -ne 0) { throw 'First launch failed' }
    $probe = [IO.File]::ReadAllLines((Join-Path $root 'probe.txt'))
    if ($probe[0] -cne 'run' -or $probe[1] -cne $title -or $probe[2] -cne '5,10' -or
        $probe[3] -cne '1' -or $probe[4] -ne '') { throw "Unexpected arguments or environment: $probe" }
    & $launcher -Launch -Options @{ XCL_TRACE_FAULTS = '' }
    if ($LASTEXITCODE -ne 0) { throw 'Launch with remembered locations failed' }
    $probe = [IO.File]::ReadAllLines((Join-Path $root 'probe.txt'))
    if ($probe[1] -cne $title -or $probe[2] -cne '5,10' -or $probe[3] -ne '') { throw 'Remembered options mismatch' }
    $saved = [IO.File]::ReadAllText((Join-Path $root 'XCL\developer-launcher.json')) | ConvertFrom-Json
    if ($saved.Game -cne $title -or $saved.Xcl -cne $exe -or $saved.Preset -cne 'Debugging') { throw 'Settings mismatch' }
    $saved.Presets | Add-Member -NotePropertyName 'Short trace' -NotePropertyValue @{ XCL_THREAD_DUMP_AT = '3,6' }
    [IO.File]::WriteAllText((Join-Path $root 'XCL\developer-launcher.json'), ($saved | ConvertTo-Json -Depth 6))
    & $launcher -Launch -Preset 'Short trace'
    $probe = [IO.File]::ReadAllLines((Join-Path $root 'probe.txt'))
    if ($LASTEXITCODE -ne 0 -or $probe[2] -cne '3,6' -or $probe[3] -ne '') { throw 'Custom preset mismatch' }
    'Launcher checks passed.'
} finally {
    $env:LOCALAPPDATA = $oldLocal
    $env:XCL_HOST = $oldHost
    $resolved = (Resolve-Path -LiteralPath $root).Path
    if ($resolved -eq $root) { Remove-Item -LiteralPath $resolved -Recurse -Force }
}
