$ErrorActionPreference = 'Stop'
$script = Join-Path $PSScriptRoot 'compare-perf.ps1'
$fixture = Join-Path ([IO.Path]::GetTempPath()) ('xcl-compare-perf-' + [guid]::NewGuid())
$null = New-Item -ItemType Directory -Path $fixture
$base = Join-Path $fixture 'base[1].log'
$candidate = Join-Path $fixture 'candidate.log'
try {
    [IO.File]::WriteAllLines($base, @(
        'game: private-path'
        '[    1.123][5][INFO ] present: first game frame shown (1280x720)'
        '[    2.002][5][INFO ] activity (2 s): 12 presents, 3 submits; DirectStorage 5 requests, 2 MB, 0 failed'
        '[    2.003][5][INFO ] memory: 100 MB committed by the process; video memory 80 MB local'
        '[    4.004][5][INFO ] activity (2 s): 9 presents, 3 submits; DirectStorage 3 requests, 2 MB, 0 failed'
        '[    4.005][5][INFO ] memory: 999 MB committed by the process; video memory 80 MB local'
    ))
    [IO.File]::WriteAllLines($candidate, @(
        '[    2.001][5][INFO ] activity (2 s): 14 presents, 3 submits; DirectStorage 8 requests, 2 MB, 0 failed'
    ))
    $output = & $script -Baseline $base -Candidate $candidate | Out-String -Width 240
    if ($output -notmatch 'Compared 1 two-second samples.*baseline 2, candidate 1' -or
        $output -notmatch 'First frame shown: baseline 1\.123 s, candidate n/a' -or
        $output -notmatch 'Peak sampled process commit \(MB\): baseline 100, candidate n/a' -or
        $output -notmatch '(?m)^\s*1\s+2\.002\s+2\.001\s+12\s+14\s+5\s+8\s*$' -or
        $output -match 'private-path|999') {
        throw "Unexpected comparison output: $output"
    }

    [IO.File]::WriteAllText($candidate, 'No activity samples')
    try {
        & $script -Baseline $base -Candidate $candidate
        throw 'Expected an error for a log without activity samples.'
    } catch {
        if ($_.Exception.Message -notmatch 'Candidate log contains no two-second activity samples') { throw }
    }
    # A writer may still own the file. Read an unterminated UTF-16 sample through a relative path.
    $sample = '[    2.500][5][INFO ] activity (2 s): 6 presents, 1 submits; DirectStorage 4 requests, 0 MB, 0 failed'
    [IO.File]::WriteAllText($base, $sample, [Text.Encoding]::Unicode)
    $writer = [IO.File]::Open($base, [IO.FileMode]::Open, [IO.FileAccess]::Write, [IO.FileShare]::ReadWrite)
    try {
        Push-Location -LiteralPath $fixture
        try { $output = & $script -Baseline 'base[1].log' -Candidate 'base[1].log' | Out-String -Width 240 }
        finally { Pop-Location }
    } finally { $writer.Dispose() }
    if ($output -notmatch '(?m)^\s*1\s+2\.500\s+2\.500\s+6\s+6\s+4\s+4\s*$') { throw $output }
    'compare-perf: all three fixture checks passed'
} finally {
    Remove-Item -LiteralPath $base, $candidate -ErrorAction SilentlyContinue
    Remove-Item -LiteralPath $fixture
}
