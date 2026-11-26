# Run with powershell.exe or pwsh.exe. Checks cold-pso-bench.ps1 against a fixture xcl.exe: the
# environment each run gets (driver blobs and implicit caches off, the thread count, its own
# times file, an exe name of its own that is removed afterwards), and the table it makes from
# XCL_PIPELINE_TIMES rows.
$ErrorActionPreference = 'Stop'
$bench = Join-Path $PSScriptRoot 'cold-pso-bench.ps1'
$fixture = Join-Path ([IO.Path]::GetTempPath()) ('xcl-cold-pso-' + [guid]::NewGuid())
$null = New-Item -ItemType Directory -Path $fixture
$savedAppData = $env:LOCALAPPDATA
$savedTimes = $env:XCL_PIPELINE_TIMES
$savedCacheOff = $env:XCL_PIPELINE_CACHE_OFF
try {
    # The fixture must never clear the user's driver cache or append to a real timing file.
    $env:LOCALAPPDATA = Join-Path $fixture 'local-appdata'
    $cache = Join-Path $env:LOCALAPPDATA 'NVIDIA\DXCache'
    $null = New-Item -ItemType Directory -Path $cache
    $sentinel = Join-Path $cache 'sentinel'
    [IO.File]::WriteAllText($sentinel, 'temporary fixture cache')
    Remove-Item Env:XCL_PIPELINE_TIMES, Env:XCL_PIPELINE_CACHE_OFF -ErrorAction SilentlyContinue
    $xcl = Join-Path $fixture 'xcl.cmd'
    # A pipelines run (XCL_PIPELINE_TIMES set) writes four builds and the environment it saw.
    [IO.File]::WriteAllText($xcl, (@(
        '@echo off'
        'if not defined XCL_PIPELINE_TIMES exit /b 0'
        '>> "%XCL_PIPELINE_TIMES%" echo 0000000000000001,1000,made,1.0,100.0'
        '>> "%XCL_PIPELINE_TIMES%" echo 0000000000000002,100000,made,0.5,300.0'
        '>> "%XCL_PIPELINE_TIMES%" echo 0000000000000003,2000000,made,0.0,5000.0'
        '>> "%XCL_PIPELINE_TIMES%" echo 0000000000000004,500,refused,0.0,1.0'
        '>> "%~dp0env-%XCL_PIPELINE_THREADS%.txt" echo %~n0 %XCL_PSO_DISK_CACHE% %XCL_PIPELINE_MAX_KB% %XCL_PIPELINE_CACHE_OFF% %*'
        'exit /b 0'
    ) -join "`r`n") + "`r`n")
    $out = Join-Path $fixture 'out'
    $table = (& $bench -Shaders 'D:\pack' -Threads 4, 8 -Limit 10 -Xcl $xcl -Out $out) -join "`n"

    $names = @()
    foreach ($n in 4, 8) {
        $seen = (Get-Content (Join-Path $fixture "env-$n.txt") -Raw).Trim()
        if ($seen -notmatch "^xcl-cold-$n-[0-9a-f]{8} 0 0 1 shaders D:\\pack --precompile --only cs --limit 10 --pipelines$") { throw "threads ${n}: exe name, environment (cache, 8 MiB cap off, cache switch) or arguments wrong: $seen" }
        $names += ($seen -split ' ')[0]
        # made: 0.1 + 0.3 + 5.0 s; p50 of three is the second, p95 and max the third
        if ($table -notmatch "\| $n \| [\d.]+ \| [\d.]+ \| \d+ \| 3 \| 1 \| 5\.4 \| 0\.3 \| 5 \| 5 \|") { throw "threads ${n}: summary row wrong:`n$table" }
        foreach ($row in @("\| $n \| < 64 KiB \| 1 \| 0\.1 \|", "\| $n \| 64-256 KiB \| 1 \| 0\.3 \|", "\| $n \| 1-4 MiB \| 1 \| 5 \|")) {
            if ($table -notmatch $row) { throw "threads ${n}: bucket row missing ($row):`n$table" }
        }
    }
    if (-not (Test-Path (Join-Path $out 'summary.csv'))) { throw 'no summary.csv' }
    foreach ($row in (Import-Csv -LiteralPath (Join-Path $out 'summary.csv'))) {
        if ([long]$row.peak_commit_mb -le 0) { throw "no peak commit for short child at $($row.threads) threads" }
    }
    # a new exe name for each run (the AMD driver's cache key), none left behind
    if ($names[0] -eq $names[1]) { throw "two runs used one exe name: $($names[0])" }
    $left = @(Get-ChildItem $fixture -Filter 'xcl-cold-*')
    if ($left.Count) { throw "exe copies left: $($left.Name -join ', ')" }

    # from cmd or bash (`-File`) "3,5" arrives as one string: two runs, not one of 35 threads
    $shell = (Get-Process -Id $PID).Path
    $null = & $shell -NoProfile -ExecutionPolicy Bypass -File $bench -Shaders 'D:\pack' -Threads '3,5' -Xcl $xcl -Out (Join-Path $fixture 'out3')
    if ($LASTEXITCODE) { throw "-File -Threads 3,5 failed ($LASTEXITCODE)" }
    if (-not ((Test-Path (Join-Path $fixture 'env-3.txt')) -and (Test-Path (Join-Path $fixture 'env-5.txt')))) { throw '-File -Threads 3,5 did not run 3 and 5 threads' }
    if (Test-Path (Join-Path $fixture 'env-35.txt')) { throw '-File -Threads 3,5 ran 35 threads' }
    $refused = $false
    try { & $bench -Shaders 'D:\pack' -Threads '4,x' -Xcl $xcl -Out (Join-Path $fixture 'out4') *> $null } catch { $refused = $_.Exception.Message -match "-Threads: 'x'" }
    if (-not $refused) { throw '-Threads 4,x was not refused' }

    # -ClearNvCache replaces the implicit-cache switch, also when the caller has it set
    $env:XCL_PIPELINE_CACHE_OFF = '1'
    & $bench -Shaders 'D:\pack' -Threads 6 -Xcl $xcl -Out (Join-Path $fixture 'out2') -ClearNvCache *> $null
    $seen = (Get-Content (Join-Path $fixture 'env-6.txt') -Raw).Trim()
    if ($seen -notmatch '^xcl-cold-6-[0-9a-f]{8} 0 0 0 shaders D:\\pack --precompile --only cs --pipelines$') { throw "-ClearNvCache: caches switch still set: $seen" }
    if (Test-Path -LiteralPath $sentinel) { throw '-ClearNvCache: temporary cache was not cleared' }
    if ($env:XCL_PIPELINE_CACHE_OFF -ne '1') { throw "-ClearNvCache changed the caller's XCL_PIPELINE_CACHE_OFF" }
    'cold-pso-bench.ps1: ok'
} finally {
    $env:LOCALAPPDATA = $savedAppData
    $env:XCL_PIPELINE_TIMES = $savedTimes
    $env:XCL_PIPELINE_CACHE_OFF = $savedCacheOff
    Remove-Item -Recurse -Force $fixture -ErrorAction SilentlyContinue
}
