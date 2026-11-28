# Run with powershell.exe or pwsh.exe. Tests the smoke block's build-input dirty marker.
$ErrorActionPreference = 'Stop'
$temp = (Resolve-Path -LiteralPath ([IO.Path]::GetTempPath())).Path
$fixture = Join-Path $temp ('xcl-smoke-dirty-' + [guid]::NewGuid())
$paths = @('scripts', 'crates', 'target')
foreach ($path in $paths) { $null = New-Item -ItemType Directory -Path (Join-Path $fixture $path) -Force }
try {
    $runner = Join-Path $fixture 'scripts\smoke.ps1'
    Copy-Item -LiteralPath (Join-Path $PSScriptRoot 'smoke.ps1') -Destination $runner
    $xcl = Join-Path $fixture 'xcl.cmd'
    [IO.File]::WriteAllText($xcl, "@echo off`r`necho Log: %~dp0xcl.log`r`n")
    [IO.File]::WriteAllText((Join-Path $fixture 'xcl.log'), "(1 frames shown, 1 PresentX calls)`n")
    [IO.File]::WriteAllText((Join-Path $fixture 'report.json'), '{"d3d12":{"adapter":"Fixture GPU"}}')
    [IO.File]::WriteAllText((Join-Path $fixture '.gitignore'), "ignored.rs`n")

    git -C $fixture init --quiet
    if ($LASTEXITCODE -ne 0) { throw 'Could not initialize the fixture repository.' }
    git -C $fixture -c core.autocrlf=false add .gitignore
    if ($LASTEXITCODE -ne 0) { throw 'Could not stage the fixture ignore rules.' }
    git -C $fixture -c user.name=Fixture -c user.email=fixture@example.invalid -c commit.gpgsign=false commit --quiet -m 'Fixture'
    if ($LASTEXITCODE -ne 0) { throw 'Could not commit the fixture ignore rules.' }
    $head = git -C $fixture rev-parse HEAD

    function Get-SmokeCommit {
        $block = & $runner -Game (Join-Path $fixture 'Game') -Xcl $xcl -Seconds 1 -Title 'Fixture'
        $block -split "`n" | Where-Object { $_ -like 'commit:*' } | ForEach-Object { $_.Trim() }
    }

    [IO.File]::WriteAllText((Join-Path $fixture 'crates\ignored.rs'), '')
    [IO.File]::WriteAllText((Join-Path $fixture 'target\build.txt'), '')
    if ((Get-SmokeCommit) -ne "commit: $head") { throw 'Ignored or out-of-scope files made the smoke block dirty.' }

    [IO.File]::WriteAllText((Join-Path $fixture 'crates\local.rs'), '')
    if ((Get-SmokeCommit) -ne "commit: $head (uncommitted changes)") { throw 'An untracked crate file did not mark the smoke block dirty.' }
    'smoke-dirty: ok'
} finally {
    $resolved = (Resolve-Path -LiteralPath $fixture).Path
    if ((Split-Path -Parent $resolved) -eq $temp) { Remove-Item -LiteralPath $resolved -Recurse -Force }
}
