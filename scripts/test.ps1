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
# GPU tests that cannot run on this adapter pass too: each prints "SKIPPED <test>: <reason>"
# to stderr (XCL_REQUIRE_GPU=wave64 / =mesh makes them fail); counted at the end
$skipped = @()
$totals = @{ Passed = 0L; Failed = 0L; Ignored = 0L; Suites = 0 }
cargo test --workspace -- --test-threads=4 @args 2>&1 | ForEach-Object {
    $line = "$_"
    $plain = $line -replace '\x1B\[[0-?]*[ -/]*[@-~]', ''
    if ($plain -match '^(?:test \S+ \.\.\. )?SKIPPED \S+: (.+)$') { $skipped += $Matches[1] }
    if ($plain -match '^test result: (?:ok|FAILED)\. (\d+) passed; (\d+) failed; (\d+) ignored;') {
        $totals.Passed += [long]$Matches[1]
        $totals.Failed += [long]$Matches[2]
        $totals.Ignored += [long]$Matches[3]
        $totals.Suites++
    }
    Write-Host $line
}
$code = $LASTEXITCODE
if ($totals.Suites) {
    Write-Host "Completed test suites: $($totals.Suites); reported totals: $($totals.Passed) passed, $($totals.Failed) failed, $($totals.Ignored) ignored."
}
if ($skipped) {
    $counts = $skipped | Group-Object | Sort-Object Count -Descending | ForEach-Object { "$($_.Count) ($($_.Name))" }
    Write-Host "GPU tests skipped on this adapter: $($counts -join ', ')"
}
exit $code
