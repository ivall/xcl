# Run with powershell.exe or pwsh.exe. Tests scripts/smoke.ps1 -OutDir artifact saving.
$ErrorActionPreference = 'Stop'
$runner = Join-Path $PSScriptRoot 'smoke.ps1'
$fixture = Join-Path ([IO.Path]::GetTempPath()) ('xcl-smoke-test-' + [guid]::NewGuid())
$null = New-Item -ItemType Directory -Path $fixture
$mockXcl = Join-Path $fixture 'mock_xcl.cmd'
$gameDir = Join-Path $fixture 'MockGame'
$null = New-Item -ItemType Directory -Path $gameDir
$runDir = Join-Path $fixture 'run'
$null = New-Item -ItemType Directory -Path $runDir
$mockLog = Join-Path $runDir 'mock.log'
$mockReport = Join-Path $runDir 'report.json'
$outDir1 = Join-Path $fixture 'artifacts1'
$outDir2 = Join-Path $fixture 'artifacts2'

try {
    # Generate mock log and report
    @"
[00:00:01] activity (2 s): 30 presents
[00:00:03] activity (2 s): 30 presents
[00:00:05] activity (2 s): 30 presents
[00:00:05] memory: 128 MB committed
(90 frames shown, 90 PresentX calls)
"@ | Set-Content -LiteralPath $mockLog -Encoding utf8

    '{"d3d12": {"adapter": "Mock Synthetic Adapter (16384 MB VRAM)"}}' |
        Set-Content -LiteralPath $mockReport -Encoding utf8

    # Mock XCL script outputs Log: path
    @"
@echo off
echo Running mock game...
echo Log: $mockLog
exit /b 0
"@ | Set-Content -LiteralPath $mockXcl -Encoding Ascii

    # Run 1: with report.json present
    $output1 = & $runner -Game $gameDir -Xcl $mockXcl -Seconds 5 -OutDir $outDir1 -Title 'MockRun1'
    if ($LASTEXITCODE -ne 0) { throw "smoke.ps1 exited with code $LASTEXITCODE on run 1" }

    if (-not (Test-Path (Join-Path $outDir1 'xcl.log'))) { throw 'run 1 did not preserve xcl.log' }
    if (-not (Test-Path (Join-Path $outDir1 'report.json'))) { throw 'run 1 did not preserve report.json' }
    if (-not (Test-Path (Join-Path $outDir1 'manifest.json'))) { throw 'run 1 did not write manifest.json' }

    $manifest1 = Get-Content (Join-Path $outDir1 'manifest.json') -Raw -Encoding utf8 | ConvertFrom-Json
    if ($manifest1.gpu -ne 'Mock Synthetic Adapter') { throw "unexpected gpu: $($manifest1.gpu)" }
    if ($manifest1.driver -ne 'unknown') { throw "unexpected driver: $($manifest1.driver)" }
    if ($manifest1.build -ne 'unknown') { throw "unexpected build: $($manifest1.build)" }
    if ($manifest1.seconds -ne 5) { throw "unexpected seconds: $($manifest1.seconds)" }
    if ($manifest1.frames -ne '90 (PresentX 90)') { throw "unexpected frames: $($manifest1.frames)" }
    if ($manifest1.title -ne 'MockRun1') { throw "unexpected title: $($manifest1.title)" }

    $smokeBlock = $output1 -join "`n"
    if ($smokeBlock -notmatch 'xcl-smoke v1') { throw "smoke block missing header: $smokeBlock" }
    if ($smokeBlock -notmatch 'gpu: Mock Synthetic Adapter') { throw "smoke block missing mock gpu: $smokeBlock" }
    if ($smokeBlock -notmatch '(?m)^driver: unknown\r?$') { throw "smoke block missing driver: $smokeBlock" }
    if ($smokeBlock -notmatch '(?m)^build: unknown\r?$') { throw "smoke block missing build: $smokeBlock" }

    # Run 2: remove report.json to test missing report case; reuse $outDir1 to verify stale report.json is purged
    Remove-Item -LiteralPath $mockReport -Force
    $output2 = & $runner -Game $gameDir -Xcl $mockXcl -Seconds 10 -Build Release -OutDir $outDir1 -Title 'MockRun2'
    if ($LASTEXITCODE -ne 0) { throw "smoke.ps1 exited with code $LASTEXITCODE on run 2" }

    if (-not (Test-Path (Join-Path $outDir1 'xcl.log'))) { throw 'run 2 did not preserve xcl.log' }
    if (Test-Path (Join-Path $outDir1 'report.json')) { throw 'run 2 did not clean up stale report.json' }
    if (-not (Test-Path (Join-Path $outDir1 'manifest.json'))) { throw 'run 2 did not write manifest.json' }

    $manifest2 = Get-Content (Join-Path $outDir1 'manifest.json') -Raw -Encoding utf8 | ConvertFrom-Json
    if ($manifest2.seconds -ne 10) { throw "unexpected seconds: $($manifest2.seconds)" }
    if ($manifest2.title -ne 'MockRun2') { throw "unexpected title: $($manifest2.title)" }
    if ($manifest2.build -ne 'release') { throw "unexpected explicit build: $($manifest2.build)" }
    $expectedController = $null
    try {
        $expectedController = Get-CimInstance Win32_VideoController -ErrorAction Stop | Where-Object { $_.Name -notmatch 'Basic' } | Select-Object -First 1
    } catch { }
    $expectedDriver = if ($expectedController -and -not [string]::IsNullOrWhiteSpace($expectedController.DriverVersion)) { $expectedController.DriverVersion.Trim() } else { 'unknown' }
    if ($manifest2.driver -ne $expectedDriver) { throw "unexpected fallback driver '$($manifest2.driver)', expected '$expectedDriver'" }

    # Run 3: infer build profile from debug/release parent directory and reject invalid -Build values
    foreach ($profile in @('debug', 'release')) {
        $profileDir = Join-Path $fixture $profile
        $null = New-Item -ItemType Directory -Path $profileDir
        $profileXcl = Join-Path $profileDir 'xcl.cmd'
        Copy-Item -LiteralPath $mockXcl -Destination $profileXcl -Force
        $inferredBlock = (& $runner -Game $gameDir -Xcl $profileXcl -Seconds 1 -Title 'InferredBuild') -join "`n"
        if ($inferredBlock -notmatch "(?m)^build: $profile\r?$") { throw "failed to infer build '$profile': $inferredBlock" }
    }

    $threw = $false
    try { $null = & $runner -Game $gameDir -Xcl $mockXcl -Seconds 1 -Build 'invalid' } catch { $threw = $true }
    if (-not $threw) { throw '-Build invalid was accepted' }

    # Run 4: positional invocation preserves 5th slot for OutDir
    $null = & $runner $gameDir 1 $mockXcl 'PositionalRun' $outDir2
    if ($LASTEXITCODE -ne 0) { throw "smoke.ps1 exited with code $LASTEXITCODE on positional run" }
    if (-not (Test-Path (Join-Path $outDir2 'manifest.json'))) { throw 'positional run did not write manifest.json in OutDir' }
    $manifestPos = Get-Content (Join-Path $outDir2 'manifest.json') -Raw -Encoding utf8 | ConvertFrom-Json
    if ($manifestPos.title -ne 'PositionalRun') { throw "unexpected positional title: $($manifestPos.title)" }

    Write-Host "test-smoke-artifacts: all assertions passed successfully."
} finally {
    if (Test-Path -LiteralPath $fixture) {
        Remove-Item -LiteralPath $fixture -Recurse -Force
    }
}
