$ErrorActionPreference = 'Stop'
$smoke = Join-Path $PSScriptRoot 'smoke.ps1'
$fixture = Join-Path ([IO.Path]::GetTempPath()) ('xcl-smoke-log-' + [guid]::NewGuid())
$null = New-Item -ItemType Directory -Path $fixture
$log = Join-Path $fixture 'run.log'
$run = Join-Path $fixture 'run.txt'
$report = Join-Path $fixture 'report.json'
Push-Location -LiteralPath $fixture
try {
    # findstr acts as the launcher. It prints the Log: line containing "run" from the fixture.
    'Log: .\run.log' | Set-Content -LiteralPath $run -Encoding Ascii
    '{"d3d12":{"adapter":"Test GPU"}}' | Set-Content -LiteralPath $report -Encoding Ascii
    $xcl = (Get-Command findstr.exe).Source
    $shell = (Get-Process -Id $PID).Path
    function Check-Report([string[]]$lines, [string[]]$expected) {
        [IO.File]::WriteAllLines($log, $lines)
        $block = & $shell -NoProfile -ExecutionPolicy Bypass -File $smoke -Game $run -Xcl $xcl -Seconds 1
        if ($LASTEXITCODE -ne 0) { throw "Smoke script exited $LASTEXITCODE" }
        foreach ($field in $expected) {
            if (($block -split '\r?\n') -notcontains $field) { throw "Missing '$field' in smoke block:`n$block" }
        }
    }
    Check-Report @(
        'memory: 900 MB committed','activity (2 s): 42 presents',
        'activity (2 s): 3 presents','activity (2 s): 6 presents','activity (2 s): 0 presents',
        'memory: 400 MB committed', 'unhandled exception; process ends with status; device removed',
        'device removed', '(51 frames shown, 52 PresentX calls)', '(2 frames shown, 3 PresentX calls)'
    ) @('frames shown: 51 (PresentX 52)', 'presents per 2 s (last 3): 3 6 0',
        'peak commit: 900 MB', 'device removals: 2', 'crashes: 2')
    Check-Report @('some launcher output') @('frames shown: none logged (the run did not reach its end)',
        'presents per 2 s (last 3): ', 'peak commit: n/a MB', 'device removals: 0', 'crashes: 0')
    Check-Report @('memory: 0 MB committed', 'activity (2 s): 0 presents') @(
        'presents per 2 s (last 3): 0', 'peak commit: 0 MB')
    '3 smoke-log cases passed'
} finally {
    Pop-Location
    foreach ($path in @($log, $run, $report, $fixture)) {
        if ($path -and (Test-Path -LiteralPath $path)) { Remove-Item -LiteralPath $path }
    }
}
