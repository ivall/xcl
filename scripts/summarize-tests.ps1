# Summarize a saved Cargo/libtest log without running tests or modifying the log.
param(
    [Parameter(Mandatory = $true, Position = 0)]
    [string]$LiteralPath,
    [switch]$Json
)
$ErrorActionPreference = 'Stop'
$file = Get-Item -LiteralPath $LiteralPath
if ($file.PSIsContainer) { throw 'Expected a test log file, not a directory.' }
$totals = @{ Passed = 0L; Failed = 0L; Ignored = 0L; Suites = 0 }
$pendingSuites = 0
$skips = [Collections.Generic.List[object]]::new()
$failed = [Collections.Generic.HashSet[string]]::new([StringComparer]::Ordinal)
$missing = [Collections.Generic.HashSet[string]]::new([StringComparer]::Ordinal)
# StreamReader detects UTF-8 and UTF-16 BOMs, including Windows PowerShell logs.
$reader = [IO.StreamReader]::new($file.FullName)
try {
    while ($null -ne ($line = $reader.ReadLine())) {
        $line = $line -replace '\x1B\[[0-?]*[ -/]*[@-~]', ''
        if ($line -match '^running \d+ tests?$') { $pendingSuites++ }
        if ($line -match '^test result: (?:ok|FAILED)\. (\d+) passed; (\d+) failed; (\d+) ignored;') {
            $totals.Passed += [long]$Matches[1]
            $totals.Failed += [long]$Matches[2]
            $totals.Ignored += [long]$Matches[3]
            $totals.Suites++
            if ($pendingSuites -gt 0) { $pendingSuites-- }
        }
        if ($line -match '^(?:test \S+ \.\.\. )?SKIPPED (\S+): (.+)$') {
            $skips.Add([pscustomobject]@{ Test = $Matches[1]; Reason = $Matches[2] })
        }
        if ($line -match '^test (\S+) \.\.\. FAILED$') { $null = $failed.Add($Matches[1]) }
        if ($line -match '^not set, their tests are skipped: (.+)$') {
            foreach ($variable in [regex]::Matches($Matches[1], '\bXCL_SAMPLE_[A-Z0-9_]+\b')) {
                $null = $missing.Add($variable.Value)
            }
        }
    }
} finally {
    $reader.Dispose()
}
if (!$totals.Suites) { throw 'No completed libtest summaries found; this is not evidence of a successful test run.' }
$notes = @(
    'Totals cover completed suites, not unique tests across repeated runs.',
    'Passed counts may include skipped checks. Skip records are reported separately, not subtracted.',
    'The original process exit code and whole-run completion cannot be established from these summaries.'
)
if ($pendingSuites -gt 0) { $notes += 'A suite start has no matching result; the log may be incomplete.' }
$report = [pscustomobject][ordered]@{
    source = $file.FullName
    completed_suites = $totals.Suites
    passed = $totals.Passed
    failed = $totals.Failed
    ignored = $totals.Ignored
    skip_records = @($skips.ToArray())
    failed_tests = @($failed | Sort-Object)
    missing_sample_inputs = @($missing | Sort-Object)
    notes = $notes
}
if ($Json) {
    $report | ConvertTo-Json -Depth 5
} else {
    "Source: $($report.source)"
    "Completed test suites: $($report.completed_suites); reported totals: $($report.passed) passed, $($report.failed) failed, $($report.ignored) ignored."
    "SKIPPED records: $($skips.Count) (absence of records does not prove that all checks ran)."
    foreach ($skip in $skips) { "  $($skip.Test): $($skip.Reason)" }
    foreach ($test in $report.failed_tests) { "FAILED: $test" }
    if ($missing.Count) { "Sample inputs not set: $($report.missing_sample_inputs -join ', ')" }
    $notes
}
# Success means the log was summarized, not that the logged tests passed.
