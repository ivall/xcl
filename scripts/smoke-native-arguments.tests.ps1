# Run with powershell.exe or pwsh.exe. Python stands in for native xcl.exe.
$ErrorActionPreference = 'Stop'
$smoke = Join-Path $PSScriptRoot 'smoke.ps1'
$python = (Get-Command python.exe -CommandType Application | Select-Object -First 1).Source
$temp = (Resolve-Path -LiteralPath ([IO.Path]::GetTempPath())).ProviderPath
$fixture = Join-Path $temp ('xcl-smoke-native-' + [guid]::NewGuid())
$originalDirectory = [Environment]::CurrentDirectory
$null = New-Item -ItemType Directory -Path $fixture
try {
    $game = Join-Path $fixture 'Mock Game'
    $null = New-Item -ItemType Directory -Path $game
    $received = Join-Path $fixture 'argv.json'
    [IO.File]::WriteAllText((Join-Path $fixture 'xcl.log'), '(1 frames shown, 1 PresentX calls)')
    [IO.File]::WriteAllText((Join-Path $fixture 'report.json'), '{"d3d12":{"adapter":"Synthetic"}}')
    [IO.File]::WriteAllText((Join-Path $fixture 'run'), @'
import json
import sys
from pathlib import Path

assert len(sys.argv) == 2, sys.argv
Path("argv.json").write_text(json.dumps(sys.argv[1]), encoding="ascii")
print("Log: " + str(Path("xcl.log").resolve()))
'@)

    [Environment]::CurrentDirectory = $fixture
    $driveRoot = [IO.Path]::GetPathRoot($fixture)
    $failures = foreach ($path in @($game, ($game + '\'), ($game + '\\'), $driveRoot)) {
        $null = & $smoke -Game $path -Title 'Synthetic argv' -Xcl $python -Seconds 1
        $actual = Get-Content -LiteralPath $received -Raw | ConvertFrom-Json
        if ($actual -cne $path) { "expected [$path], received [$actual]" }
    }
    if ($failures) { throw ($failures -join "`n") }
    'smoke-native-arguments: ok'
} finally {
    [Environment]::CurrentDirectory = $originalDirectory
    $resolved = (Resolve-Path -LiteralPath $fixture).ProviderPath
    if ((Split-Path -Parent $resolved) -eq $temp) { Remove-Item -LiteralPath $resolved -Recurse -Force }
}
