# Run with powershell.exe or pwsh.exe; exercises the command as a separate process.
$ErrorActionPreference = 'Stop'
$script = Join-Path $PSScriptRoot 'summarize-tests.ps1'
$hostExe = (Get-Process -Id $PID).Path
$fixture = Join-Path ([IO.Path]::GetTempPath()) ('xcl-log-summary-' + [guid]::NewGuid())
$null = New-Item -ItemType Directory -Path $fixture
$log = Join-Path $fixture 'test [1].log'

function Read-Report($path, $expectFailure = $false) {
    $ErrorActionPreference = 'Continue'
    $output = & $hostExe -NoProfile -ExecutionPolicy Bypass -File $script -LiteralPath $path -Json 2>&1
    $code = $LASTEXITCODE
    $ErrorActionPreference = 'Stop'
    if ($expectFailure) {
        if (!$code) { throw 'Expected a failing exit code for an unusable input.' }
    } else {
        if ($code) { throw "Summary failed: $output" }
        ($output -join "`n") | ConvertFrom-Json
    }
}
try {
    $esc = [char]27
    $text = @"
not set, their tests are skipped: XCL_SAMPLE_RPF, XCL_SAMPLE_DLLS
running 4 tests
SKIPPED gpu_one: no wave64
test gpu_two ... SKIPPED gpu_two: no mesh shaders
test failing_test ... FAILED
${esc}[31mtest result: FAILED. 2 passed; 1 failed; 1 ignored; 0 measured; 0 filtered out; finished in 0.01s${esc}[0m
running 3 tests
test result: ok. 3 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.01s
diagnostic: test result: ok. 999 passed; 0 failed; 0 ignored;
"@
    foreach ($encoding in @([Text.UTF8Encoding]::new($false), [Text.Encoding]::Unicode)) {
        [IO.File]::WriteAllText($log, $text, $encoding)
        $r = Read-Report $log
        if ($r.completed_suites -ne 2 -or $r.passed -ne 5 -or $r.failed -ne 1 -or $r.ignored -ne 1) {
            throw 'Incorrect totals or diagnostic text counted as a summary.'
        }
        if ($r.skip_records.Count -ne 2 -or $r.skip_records[1].Test -ne 'gpu_two') { throw 'Lost skip records.' }
        if ($r.failed_tests.Count -ne 1 -or $r.failed_tests[0] -ne 'failing_test') { throw 'Lost failed test name.' }
        if ($r.missing_sample_inputs.Count -ne 2) { throw 'Lost missing sample inputs.' }
        if ($r.notes.Count -ne 3) { throw 'Missing limitations or false truncation warning.' }
        Write-Host "PASS $($encoding.WebName), ANSI, literal path, skips and failures"
    }
    [IO.File]::WriteAllText($log, $text + "`nrunning 5 tests`n", [Text.Encoding]::UTF8)
    $r = Read-Report $log
    if ($r.completed_suites -ne 2 -or $r.notes[-1] -notlike '*incomplete*') { throw 'Missed incomplete suite.' }
    Write-Host 'PASS incomplete log'
    # A capture can start at one suite's result and end while the next suite is running.
    $partial = "test result: ok. 2 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out;`nrunning 1 test"
    [IO.File]::WriteAllText($log, $partial, [Text.Encoding]::UTF8)
    $r = Read-Report $log
    if ($r.completed_suites -ne 1 -or $r.passed -ne 2 -or $r.notes[-1] -notlike '*incomplete*') {
        throw 'An earlier summary masked a later unfinished suite.'
    }
    Write-Host 'PASS incomplete capture starting at a summary'
    [IO.File]::WriteAllText($log, 'test result: ok. 0 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out;', [Text.Encoding]::UTF8)
    $r = Read-Report $log
    if ($r.completed_suites -ne 1 -or $r.skip_records.Count -ne 0 -or $r.failed_tests.Count -ne 0) { throw 'Empty arrays or suite were lost.' }
    Write-Host 'PASS empty completed suite'
    [IO.File]::WriteAllText($log, 'running 3 tests', [Text.Encoding]::UTF8)
    Read-Report $log $true
    Write-Host 'PASS no completed summaries rejected'
    Read-Report (Join-Path $fixture 'missing.log') $true
    Write-Host 'PASS missing file rejected'
    Read-Report $fixture $true
    Write-Host 'PASS directory rejected'
} finally {
    foreach ($path in $log, $fixture) {
        if (Test-Path -LiteralPath $path) { Remove-Item -LiteralPath $path }
    }
}
