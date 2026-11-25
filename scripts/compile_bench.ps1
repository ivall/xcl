# Compile-time benchmark on generated RDNA2 programs (crates/xcl/tests/compile_bench): no game
# data, DXC on the CPU, no GPU needed. Builds release, compiles each program one at a time and
# writes compile_bench.json / .csv and the log to the output folder.
#   .\scripts\compile_bench.ps1                       # full set, 3 compiles per program
#   .\scripts\compile_bench.ps1 -Set small -Reps 1    # the set every test run uses
#   .\scripts\compile_bench.ps1 -Set large            # only the multi-megabyte programs
#   .\scripts\compile_bench.ps1 -DxcArgs '-O1'        # extra DXC arguments
# Compare two runs (e.g. main and a PR's head, same machine):
#   python -I scripts/compile_bench_compare.py <base>\compile_bench.json <head>\compile_bench.json
param(
    [ValidateSet('full', 'large', 'small')][string]$Set = 'full',
    [int]$Reps = 0,
    [string]$DxcArgs = '',
    [string]$Out = ''
)
$ErrorActionPreference = 'Stop'
Set-Location (Join-Path $PSScriptRoot '..')
$null = Get-Command cargo -ErrorAction Stop
$commit = (git rev-parse HEAD).Trim()
$dirty = [bool](git status --porcelain)
if ($dirty) { Write-Warning 'uncommitted changes: the results say "dirty": true and do not stand for a commit' }
if (-not $Out) { $Out = Join-Path 'target\compile_bench' ($commit.Substring(0, 10) + $(if ($dirty) { '-dirty' } else { '' })) }
New-Item -ItemType Directory -Force $Out | Out-Null
$log = Join-Path $Out 'compile_bench.log'

$env:XCL_COMPILE_BENCH = if ($Set -eq 'small') { '' } else { $Set }
$env:XCL_COMPILE_BENCH_OUT = (Resolve-Path $Out).Path
$env:XCL_COMPILE_BENCH_REPS = if ($Reps -gt 0) { "$Reps" } else { '' }
$env:XCL_COMPILE_BENCH_DXC_ARGS = $DxcArgs
# Windows PowerShell 5.1 treats redirected native stderr as an error even when Cargo succeeds
$ErrorActionPreference = 'Continue'
try {
    # one compile at a time: nothing else in the test binary runs beside the measurement
    cargo test --release -p xcl --test compile_bench dxc_compiles_the_generated_programs -- --exact --test-threads=1 --nocapture 2>&1 |
        ForEach-Object { "$_" } | Tee-Object -Variable lines | Write-Host
    $code = $LASTEXITCODE
} finally {
    foreach ($v in 'XCL_COMPILE_BENCH', 'XCL_COMPILE_BENCH_OUT', 'XCL_COMPILE_BENCH_REPS', 'XCL_COMPILE_BENCH_DXC_ARGS') { Remove-Item "env:$v" -ErrorAction SilentlyContinue }
}
$lines | Out-File -FilePath $log -Encoding utf8
if ($code -eq 0 -and -not (Test-Path (Join-Path $Out 'compile_bench.json'))) {
    Write-Warning 'no compile_bench.json: the test was skipped (no DXC?), see the log'
    $code = 1
}
Write-Host "commit $commit$(if ($dirty) { ' (dirty)' }), results and log: $Out"
exit $code
