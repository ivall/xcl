# Run with powershell.exe or pwsh.exe. A stand-in launcher resolves the game argument.
$ErrorActionPreference = 'Stop'
$smoke = Join-Path $PSScriptRoot 'smoke.ps1'
$temp = (Resolve-Path -LiteralPath ([IO.Path]::GetTempPath())).Path
$fixture = Join-Path $temp ('xcl-smoke-game-' + [guid]::NewGuid())
$game = Join-Path $fixture 'Game With Spaces'
$null = New-Item -ItemType Directory -Path $game -Force
try {
    $xcl = Join-Path $fixture 'xcl.cmd'
    @'
@echo off
if not exist "%~2\sentinel.txt" exit /b 51
echo Log: %~dp0xcl.log
'@ | Set-Content -LiteralPath $xcl -Encoding ascii
    [IO.File]::WriteAllText((Join-Path $game 'sentinel.txt'), 'fixture')
    [IO.File]::WriteAllText((Join-Path $fixture 'xcl.log'), "(1 frames shown, 1 PresentX calls)`n")
    [IO.File]::WriteAllText((Join-Path $fixture 'report.json'), '{"d3d12":{"adapter":"Fixture GPU"}}')

    $absolute = & $smoke -Game $game -Xcl $xcl -Seconds 1
    Push-Location -LiteralPath $fixture
    try { $relative = & $smoke -Game '.\Game With Spaces' -Xcl $xcl -Seconds 1 }
    finally { Pop-Location }
    Push-Location -LiteralPath $game
    try { $dot = & $smoke -Game '.' -Xcl $xcl -Seconds 1 }
    finally { Pop-Location }

    $title = $absolute -split "`n" | Where-Object { $_ -like 'title:*' }
    foreach ($block in @($relative, $dot)) {
        if (($block -split "`n" | Where-Object { $_ -like 'title:*' }) -ne $title) {
            throw 'The same game folder produced a different title hash.'
        }
        if ($block -notmatch 'frames shown: 1 \(PresentX 1\)') { throw 'Missing smoke result.' }
    }
    'smoke-relative-game: ok'
} finally {
    $resolved = (Resolve-Path -LiteralPath $fixture).Path
    if ((Split-Path -Parent $resolved) -eq $temp) { Remove-Item -LiteralPath $resolved -Recurse -Force }
}
