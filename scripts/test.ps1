# Build everything (the e2e test needs the test title + DLL) and run all tests.
# Arguments go to the test binaries (after `--`), e.g. `.\scripts\test.ps1 --skip <name>`.
$ErrorActionPreference = 'Stop'
Set-Location (Join-Path $PSScriptRoot '..')
# tests on game data pass without running anything when their variable is not set
$missing = 'XCL_SAMPLE_RPF', 'XCL_SAMPLE_DLLS', 'XCL_SAMPLE_TTPF', 'XCL_SAMPLE_BCPACK', 'XCL_SAMPLE_BK2', 'XCL_SAMPLE_BVH4', 'XCL_SAMPLE_AWC', 'XCL_SAMPLE_XG', 'XCL_SAMPLE_DXC' | Where-Object { -not (Test-Path "env:$_") }
if ($missing) { Write-Host "not set, their tests are skipped: $($missing -join ', ')" }
# Windows PowerShell 5.1 treats redirected native stderr as an error even when Cargo succeeds.
$null = Get-Command cargo -ErrorAction Stop
$ErrorActionPreference = 'Continue'
cargo build --workspace
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
# Tests that cannot run pass too: each writes "SKIPPED <test>: <reason>" to stderr – GPU
# capability skips (XCL_REQUIRE_GPU=wave64 / =mesh makes them fail) and missing sample data
# are counted separately at the end
$skipped = @()
$sampleSkipped = @()
cargo test --workspace -- --test-threads=4 @args 2>&1 | ForEach-Object {
    $line = "$_"
    if ($line -match '^SKIPPED \S+: (no XCL_SAMPLE_.*)$') { $sampleSkipped += $Matches[1] }
    elseif ($line -match '^SKIPPED \S+: (.+)$') { $skipped += $Matches[1] }
    Write-Host $line
}
$code = $LASTEXITCODE
if ($skipped) {
    $counts = $skipped | Group-Object | Sort-Object Count -Descending | ForEach-Object { "$($_.Count) ($($_.Name))" }
    Write-Host "GPU tests skipped on this adapter: $($counts -join ', ')"
}
if ($sampleSkipped) {
    $counts = $sampleSkipped | Group-Object | Sort-Object Count -Descending | ForEach-Object { "$($_.Count) ($($_.Name))" }
    Write-Host "sample-data tests skipped (XCL_SAMPLE_* not set): $($counts -join ', ')"
}
exit $code
