# Run with powershell.exe or pwsh.exe. -Xcl paths are relative to the shell's location.
$ErrorActionPreference = 'Stop'
$smoke = Join-Path $PSScriptRoot 'smoke.ps1'
$temp = (Resolve-Path -LiteralPath ([IO.Path]::GetTempPath())).Path
$fixture = Join-Path $temp ('xcl-smoke-relative-' + [guid]::NewGuid())
$null = New-Item -ItemType Directory -Path $fixture
try {
    $xcl = Join-Path $fixture 'xcl.cmd'
    [IO.File]::WriteAllText($xcl, "@echo off`r`necho Log: %~dp0xcl.log`r`n")
    [IO.File]::WriteAllText((Join-Path $fixture 'xcl.log'), "(1 frames shown, 1 PresentX calls)`n")
    [IO.File]::WriteAllText((Join-Path $fixture 'report.json'), '{"d3d12":{"adapter":"Fixture GPU"}}')
    Push-Location -LiteralPath $fixture
    try {
        $block = & $smoke -Game $fixture -Xcl '.\xcl.cmd' -Seconds 1 -Title 'Fixture'
        if (($block -join "`n") -notmatch 'frames shown: 1 \(PresentX 1\)') {
            throw 'The relative launcher did not produce its smoke result.'
        }
    } finally { Pop-Location }
    'smoke-relative-xcl: ok'
} finally {
    $resolved = (Resolve-Path -LiteralPath $fixture).Path
    if ((Split-Path -Parent $resolved) -eq $temp) { Remove-Item -LiteralPath $resolved -Recurse -Force }
}
