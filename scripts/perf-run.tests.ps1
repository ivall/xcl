# Run on Windows with powershell.exe or pwsh.exe. Uses a clean temporary repository and mock
# cargo/XCL commands; no game executable is run.
$ErrorActionPreference = 'Stop'
$runnerSource = Join-Path $PSScriptRoot 'perf-run.ps1'
$fixture = Join-Path ([IO.Path]::GetTempPath()) ('xcl-perf-run-test-' + [guid]::NewGuid())
$repo = Join-Path $fixture 'repo'
$scripts = Join-Path $repo 'scripts'
$game = Join-Path $fixture 'Mock Game'
$out = Join-Path $fixture 'artifacts'
$otherGame = Join-Path (Join-Path $fixture 'other-build') 'Mock Game'
$otherOut = Join-Path $fixture 'other-artifacts'
$featureOut = Join-Path $fixture 'feature-artifacts'
$featureTarget = Join-Path $fixture 'feature target'
$bin = Join-Path $fixture 'bin'
$userData = Join-Path $fixture 'user-data'
$userDxil = Join-Path $fixture 'user-dxil'
$cargoMarker = Join-Path $fixture 'cargo.txt'
$runLog = Join-Path $fixture 'run.log'
$mockXcl = Join-Path $bin 'mock-xcl.cmd'
$oldPath = $env:PATH
$oldTest = $env:XCL_PERF_RUN_TESTING
$oldTestXcl = $env:XCL_PERF_RUN_TEST_XCL
$oldData = $env:XCL_DATA_ROOT
$oldDxil = $env:XCL_DXIL_CACHE
$oldPress = $env:XCL_PRESS_AT
$oldGpuTiming = $env:XCL_GPU_TIMING
$oldGpuTimingMs = $env:XCL_GPU_TIMING_MS
$oldPipelineThreads = $env:XCL_PIPELINE_THREADS
$oldPrepareAll = $env:XCL_PREPARE_ALL

try {
    $null = New-Item -ItemType Directory -Path $scripts, $game, $bin, $userData, $userDxil
    Copy-Item -LiteralPath $runnerSource -Destination (Join-Path $scripts 'perf-run.ps1')
    'target/' | Set-Content -LiteralPath (Join-Path $repo '.gitignore') -Encoding Ascii
    'not copied' | Set-Content -LiteralPath (Join-Path $game 'game-data.bin') -Encoding Ascii
    'keep' | Set-Content -LiteralPath (Join-Path $userData 'sentinel.txt') -Encoding Ascii
    'keep' | Set-Content -LiteralPath (Join-Path $userDxil 'sentinel.txt') -Encoding Ascii

    @'
@echo off
echo %* > "%MOCK_CARGO_MARKER%"
>&2 echo synthetic build progress
exit /b 0
'@ | Set-Content -LiteralPath (Join-Path $bin 'cargo.cmd') -Encoding Ascii
    @'
@echo off
echo rustc 1.test
echo host: x86_64-pc-windows-msvc
exit /b 0
'@ | Set-Content -LiteralPath (Join-Path $bin 'rustc.cmd') -Encoding Ascii
    @'
@echo off
set STATE=warm
if not exist "%XCL_DXIL_CACHE%\seen" set STATE=cold
if exist "%XCL_DXIL_CACHE%\seen" goto marker_done
echo seen>"%XCL_DXIL_CACHE%\seen"
:marker_done
if not exist "%XCL_DATA_ROOT%\MockTitle\logs" mkdir "%XCL_DATA_ROOT%\MockTitle\logs"
echo cache=%STATE%>"%XCL_DATA_ROOT%\MockTitle\logs\xcl.log"
echo gpu_timing=%XCL_GPU_TIMING% threshold=%XCL_GPU_TIMING_MS%>>"%XCL_DATA_ROOT%\MockTitle\logs\xcl.log"
if not "%MOCK_XCL_CRASH%"=="1" goto no_crash
echo Game exited with code 0xc0000005 (3221225477)
echo No report - see the log: %XCL_DATA_ROOT%\MockTitle\logs\xcl.log
exit /b -1073741819
:no_crash
echo {"d3d12":{"adapter":"Mock GPU","submissions":2,"slowest_submission_ms":1.5,"slowest_submission_lists":3}}>"%XCL_DATA_ROOT%\MockTitle\logs\report.json"
echo Log: %XCL_DATA_ROOT%\MockTitle\logs\xcl.log
if not "%MOCK_XCL_LINGER%"=="1" exit /b 0
start "" /b cmd /d /c ""%~dp0mock-child.cmd""
ping -n 11 127.0.0.1 >nul
exit /b 0
'@ | Set-Content -LiteralPath $mockXcl -Encoding Ascii
    # Started by the mock and sharing its output: writes the log 5 s later unless it is ended.
    @'
@echo off
ping -n 6 127.0.0.1 >nul
echo late child write>>"%XCL_DATA_ROOT%\MockTitle\logs\xcl.log"
'@ | Set-Content -LiteralPath (Join-Path $bin 'mock-child.cmd') -Encoding Ascii

    & git.exe -C $repo init | Out-Null
    & git.exe -C $repo add .
    & git.exe -C $repo -c user.name='XCL test' -c user.email='xcl-test@example.invalid' -c commit.gpgsign=false commit -m fixture | Out-Null
    if ($LASTEXITCODE -ne 0) { throw 'could not create clean fixture repository' }

    $env:PATH = $bin + [IO.Path]::PathSeparator + $oldPath
    $env:MOCK_CARGO_MARKER = $cargoMarker
    $env:XCL_PERF_RUN_TESTING = '1'
    $env:XCL_PERF_RUN_TEST_XCL = $mockXcl
    $env:XCL_DATA_ROOT = $userData
    $env:XCL_DXIL_CACHE = $userDxil
    $env:XCL_PRESS_AT = '20:A,24:A'
    $env:XCL_GPU_TIMING = 'caller-value'
    $env:XCL_GPU_TIMING_MS = '999'
    $env:XCL_PIPELINE_THREADS = '1'
    $env:XCL_PREPARE_ALL = '1'

    # Redirected like a queued or logged run: Windows PowerShell then turns native stderr (cargo's
    # progress) into error records, which must not stop the script.
    & (Join-Path $scripts 'perf-run.ps1') -Game $game -OutDir $out -Seconds 2 -StartupTimeoutSeconds 30 *> $runLog
    if ($LASTEXITCODE -ne 0) { throw "perf-run.ps1 exited with $LASTEXITCODE" }
    if (-not (Get-Content -LiteralPath $runLog -Raw).Contains('synthetic build progress')) { throw 'native build output was not kept in the redirected log' }

    $manifest = Get-Content -LiteralPath (Join-Path $out 'manifest.json') -Raw -Encoding utf8 | ConvertFrom-Json
    $manifestRaw = Get-Content -LiteralPath (Join-Path $out 'manifest.json') -Raw -Encoding utf8
    if ($manifest.schema -ne 'xcl-perf-run' -or $manifest.version -ne 1) { throw 'wrong manifest schema/version' }
    if ($manifest.dirty -ne $false -or $manifest.build.profile -ne 'release') { throw 'build identity was not recorded' }
    if (($manifest.run_order -join ',') -ne 'xcl-cold,xcl-warm') { throw 'run order was not cold then warm' }
    if ($manifest.cache.driver -ne 'uncontrolled') { throw 'driver cache was not labelled uncontrolled' }
    if ($manifest.environment.XCL_PRESS_AT -ne '20:A,24:A') { throw 'input schedule was not recorded' }
    if ($manifest.environment.XCL_GPU_TIMING -ne '1' -or $manifest.environment.XCL_GPU_TIMING_MS -ne '50') {
        throw 'submission timing settings were not forced and recorded'
    }
    if ($manifest.environment.XCL_PIPELINE_THREADS -ne '1') { throw 'an inherited XCL setting was not recorded' }
    if ($manifest.environment.XCL_PREPARE_ALL -ne '1') { throw 'the shader preparation switch was not recorded' }
    $recordedNames = @($manifest.environment.PSObject.Properties.Name)
    if (@($recordedNames | Where-Object { $_ -in @('XCL_DATA_ROOT', 'XCL_DXIL_CACHE') -or $_ -like 'XCL_PERF_RUN_*' }).Count -ne 0) {
        throw 'per-run cache paths or test hooks were recorded'
    }
    if ($manifest.workload_id_algorithm -ne 'sha256-relative-path-length-mtime-utc-ticks' -or
        $manifest.workload_id_version -ne 1) { throw 'workload identity algorithm/version was not recorded' }
    if ($manifest.workload_id -notmatch '^[0-9a-f]{64}$') { throw 'workload identity is not a full SHA-256 value' }
    if ($manifestRaw.Contains($game) -or $manifestRaw.Contains('game-data.bin')) {
        throw 'workload inventory details leaked into the manifest'
    }
    if ($manifest.runs.Count -ne 2 -or $manifest.runs[0].cache_state -ne 'xcl-cold' -or $manifest.runs[1].cache_state -ne 'xcl-warm') { throw 'run records are incomplete' }
    if (@($manifest.runs | Where-Object { $_.startup_timeout_seconds -ne 30 }).Count -ne 0) { throw 'startup timeout was not recorded' }
    if ((Get-Content -LiteralPath (Join-Path $out 'runs\01-cold\xcl.log') -Raw) -notmatch 'cache=cold') { throw 'cold run did not start with a fresh DXIL cache' }
    if ((Get-Content -LiteralPath (Join-Path $out 'runs\02-warm\xcl.log') -Raw) -notmatch 'cache=warm') { throw 'warm run did not reuse the cold cache' }
    if ((Get-Content -LiteralPath (Join-Path $out 'runs\01-cold\xcl.log') -Raw) -notmatch 'gpu_timing=1 threshold=50') {
        throw 'submission timing settings were not passed to XCL'
    }
    if (-not (Test-Path -LiteralPath (Join-Path $out 'runs\01-cold\report.json')) -or -not (Test-Path -LiteralPath (Join-Path $out 'runs\02-warm\report.json'))) { throw 'reports were not preserved' }
    $timing = (Get-Content -LiteralPath (Join-Path $out 'runs\01-cold\report.json') -Raw -Encoding utf8 | ConvertFrom-Json).d3d12
    if ($timing.submissions -ne 2 -or $timing.slowest_submission_ms -ne 1.5 -or $timing.slowest_submission_lists -ne 3) {
        throw 'submission timing fields were not preserved'
    }
    $artifactFiles = @(Get-ChildItem -LiteralPath $out -Recurse -File)
    $unexpectedArtifacts = @($artifactFiles | Where-Object { $_.Name -notin @('manifest.json', 'xcl.log', 'report.json') })
    if ($artifactFiles.Count -ne 5 -or $unexpectedArtifacts.Count -ne 0) { throw 'the artifact folder contains something other than logs, reports, and the manifest' }
    if ($env:XCL_DATA_ROOT -ne $userData -or $env:XCL_DXIL_CACHE -ne $userDxil -or
        $env:XCL_GPU_TIMING -ne 'caller-value' -or $env:XCL_GPU_TIMING_MS -ne '999') {
        throw 'the caller environment was not restored'
    }
    if (-not (Test-Path -LiteralPath (Join-Path $userData 'sentinel.txt')) -or -not (Test-Path -LiteralPath (Join-Path $userDxil 'sentinel.txt'))) { throw 'a user cache was touched' }
    if (Test-Path -LiteralPath (Join-Path ([IO.Path]::GetTempPath()) ("xcl-perf-$($manifest.cache.identity)"))) { throw 'the ephemeral cache was not removed' }
    if ((Get-Content -LiteralPath $cargoMarker -Raw) -notmatch 'build --locked --release -p xcl') { throw 'release build command was not run' }

    & (Join-Path $scripts 'perf-run.ps1') -Game $game -OutDir $featureOut -Seconds 2 `
        -CargoFeatures @('zeta,alpha', 'alpha') -CargoTargetDir $featureTarget
    if ($LASTEXITCODE -ne 0) { throw "feature perf-run.ps1 exited with $LASTEXITCODE" }
    $featureManifest = Get-Content -LiteralPath (Join-Path $featureOut 'manifest.json') -Raw -Encoding utf8 | ConvertFrom-Json
    if ((@($featureManifest.build.cargo_features) -join ',') -ne 'alpha,zeta' -or
        $featureManifest.build.target_dir_kind -ne 'explicit' -or
        $featureManifest.build.executable_sha256 -notmatch '^[0-9a-f]{64}$') {
        throw 'the canonical Cargo build variant was not recorded'
    }
    $cargoInvocation = Get-Content -LiteralPath $cargoMarker -Raw
    if ($cargoInvocation -notmatch '--features=alpha,zeta' -or
        $cargoInvocation -notmatch '--target-dir' -or
        $cargoInvocation -notmatch ([regex]::Escape($featureTarget))) {
        throw 'Cargo features/target directory were not passed as canonical arguments'
    }

    $null = New-Item -ItemType Directory -Path $otherGame -Force
    'a different title build with the same folder leaf' | Set-Content -LiteralPath (Join-Path $otherGame 'game-data.bin') -Encoding Ascii
    & (Join-Path $scripts 'perf-run.ps1') -Game $otherGame -OutDir $otherOut -Seconds 2
    if ($LASTEXITCODE -ne 0) { throw "second perf-run.ps1 exited with $LASTEXITCODE" }
    $otherManifest = Get-Content -LiteralPath (Join-Path $otherOut 'manifest.json') -Raw -Encoding utf8 | ConvertFrom-Json
    if ($otherManifest.workload_id -eq $manifest.workload_id) {
        throw 'different builds with the same folder leaf received the same workload identity'
    }

    # A crash before the report: the log is still kept and the exit code recorded.
    $crashOut = Join-Path $fixture 'crash-artifacts'
    $env:MOCK_XCL_CRASH = '1'
    try {
        & (Join-Path $scripts 'perf-run.ps1') -Game $game -OutDir $crashOut -Seconds 2 *> $runLog
    } finally {
        Remove-Item Env:MOCK_XCL_CRASH -ErrorAction SilentlyContinue
    }
    $crashManifest = Get-Content -LiteralPath (Join-Path $crashOut 'manifest.json') -Raw -Encoding utf8 | ConvertFrom-Json
    if ($crashManifest.runs[0].exit_code -ne -1073741819 -or $null -ne $crashManifest.runs[0].artifacts.report) { throw 'a crashed run was not recorded with its exit code and without a report' }
    if ((Get-Content -LiteralPath (Join-Path $crashOut 'runs\01-cold\xcl.log') -Raw) -notmatch 'cache=cold') { throw 'the crashed run''s log was not kept' }

    # A timeout ends the whole process tree: a child of the launcher does not outlive the run,
    # keep its output open or write the log after the run's time is taken.
    $lingerOut = Join-Path $fixture 'linger-artifacts'
    $env:MOCK_XCL_LINGER = '1'
    try {
        & (Join-Path $scripts 'perf-run.ps1') -Game $game -OutDir $lingerOut -Seconds 1 -StartupTimeoutSeconds 1 *> $runLog
    } finally {
        Remove-Item Env:MOCK_XCL_LINGER -ErrorAction SilentlyContinue
    }
    $lingerManifest = Get-Content -LiteralPath (Join-Path $lingerOut 'manifest.json') -Raw -Encoding utf8 | ConvertFrom-Json
    foreach ($run in $lingerManifest.runs) {
        if (-not $run.timed_out -or $run.duration_actual_ms -ge 4000) { throw "a timed-out run was not ended at its deadline: $($run.duration_actual_ms) ms" }
    }
    foreach ($runName in @('01-cold', '02-warm')) {
        if ((Get-Content -LiteralPath (Join-Path $lingerOut "runs\$runName\xcl.log") -Raw) -match 'late child write') {
            throw "a process left by the $runName run wrote its log after the timeout"
        }
    }

    Add-Content -LiteralPath (Join-Path $repo '.gitignore') -Value '# dirty'
    $refused = $false
    try {
        & (Join-Path $scripts 'perf-run.ps1') -Game $game -OutDir (Join-Path $fixture 'dirty-out') -Seconds 2
    } catch {
        $refused = $_.Exception.Message -match 'clean worktree'
    }
    if (-not $refused) { throw 'a dirty worktree was not refused' }
    Set-Content -LiteralPath (Join-Path $repo '.gitignore') -Value 'target/' -Encoding Ascii

    # A real executable parses its command line the way XCL does (a command file does not): a game
    # folder with a space and a trailing separator must arrive as given. The shim records its
    # arguments, then hands over to the mock.
    $nativeXcl = Join-Path $bin 'mock-xcl.exe'
    $nativeArgs = Join-Path $fixture 'native-args.txt'
    $shimSource = Join-Path $fixture 'shim.cs'
    @'
using System;
using System.Diagnostics;
using System.IO;
static class Shim {
    static int Main(string[] args) {
        File.WriteAllLines(Environment.GetEnvironmentVariable("MOCK_XCL_ARGS"), args);
        var cmd = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "mock-xcl.cmd");
        var psi = new ProcessStartInfo("cmd.exe", "/d /c \"\"" + cmd + "\"\"") { UseShellExecute = false };
        using (var p = Process.Start(psi)) { p.WaitForExit(); return p.ExitCode; }
    }
}
'@ | Set-Content -LiteralPath $shimSource -Encoding Ascii
    $csc = Join-Path $env:WINDIR 'Microsoft.NET\Framework64\v4.0.30319\csc.exe'
    & $csc /nologo /target:exe "/out:$nativeXcl" $shimSource | Out-Null
    if ($LASTEXITCODE -ne 0) { throw 'could not compile the native mock' }
    $env:XCL_PERF_RUN_TEST_XCL = $nativeXcl
    $env:MOCK_XCL_ARGS = $nativeArgs
    $slashGame = $game + '\'
    & (Join-Path $scripts 'perf-run.ps1') -Game $slashGame -OutDir (Join-Path $fixture 'native-artifacts') -Seconds 2 *> $runLog
    $received = @(Get-Content -LiteralPath $nativeArgs)
    if ($received.Count -ne 2 -or $received[0] -ne 'run' -or $received[1] -ne $slashGame) {
        throw "the native launch received [$($received -join '] [')] instead of [run] [$slashGame]"
    }

    Write-Host 'perf-run tests: all assertions passed'
} finally {
    $env:PATH = $oldPath
    $env:XCL_PERF_RUN_TESTING = $oldTest
    $env:XCL_PERF_RUN_TEST_XCL = $oldTestXcl
    $env:XCL_DATA_ROOT = $oldData
    $env:XCL_DXIL_CACHE = $oldDxil
    $env:XCL_PRESS_AT = $oldPress
    $env:XCL_GPU_TIMING = $oldGpuTiming
    $env:XCL_GPU_TIMING_MS = $oldGpuTimingMs
    $env:XCL_PIPELINE_THREADS = $oldPipelineThreads
    $env:XCL_PREPARE_ALL = $oldPrepareAll
    Remove-Item Env:MOCK_CARGO_MARKER, Env:MOCK_XCL_CRASH, Env:MOCK_XCL_ARGS -ErrorAction SilentlyContinue
    if (Test-Path -LiteralPath $fixture) { Remove-Item -LiteralPath $fixture -Recurse -Force }
}
