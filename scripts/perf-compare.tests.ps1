$ErrorActionPreference = 'Stop'
$tool = Join-Path $PSScriptRoot 'perf-compare.ps1'
$fixture = Join-Path ([IO.Path]::GetTempPath()) ('xcl-perf-compare-test-' + [guid]::NewGuid())
$baseline = Join-Path $fixture 'baseline'
$candidate = Join-Path $fixture 'candidate'

function Write-Fixture([string]$Root, [string]$Commit, [int]$ColdMs, $WarmMs) {
    $null = New-Item -ItemType Directory -Path (Join-Path $Root 'runs\01-cold'), (Join-Path $Root 'runs\02-warm') -Force
    $manifest = [ordered]@{
        schema = 'xcl-perf-run'
        version = 1
        commit = $Commit
        dirty = $false
        build = [ordered]@{ profile = 'release' }
        workload_id = 'same-workload'
        run_order = @('xcl-cold', 'xcl-warm')
        environment = [ordered]@{ XCL_PRESS_AT = '20:A' }
        runs = @(
            [ordered]@{ order = 1; cache_state = 'xcl-cold'; duration_requested_seconds = 60; duration_actual_ms = $ColdMs; exit_code = 0; timed_out = $false },
            [ordered]@{ order = 2; cache_state = 'xcl-warm'; duration_requested_seconds = 60; duration_actual_ms = $WarmMs; exit_code = $null; timed_out = $true }
        )
    }
    $manifest | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath (Join-Path $Root 'manifest.json') -Encoding utf8
}

function Write-Report([string]$Path, [int]$Count, [long]$Total, [long]$Maximum, [switch]$ExtraKind) {
    $aggregates = @(
        [ordered]@{ kind = 'dxc_compile'; count = $Count; total_ns = $Total; max_ns = $Maximum; histogram_log2_ns = @(0, $Count) }
    )
    if ($ExtraKind) {
        $aggregates += [ordered]@{ kind = 'gpu_wait'; count = 1; total_ns = 9; max_ns = 9; histogram_log2_ns = @(0, 0, 0, 1) }
    }
    [ordered]@{ stalls = [ordered]@{ version = 1; aggregates = $aggregates; slow_events = @() } } |
        ConvertTo-Json -Depth 6 | Set-Content -LiteralPath $Path -Encoding utf8
}

try {
    Write-Fixture $baseline 'base-commit' 1000 0
    Write-Fixture $candidate 'candidate-commit' 850 0
    Write-Report (Join-Path $baseline 'runs\01-cold\report.json') 2 100 70
    Write-Report (Join-Path $candidate 'runs\01-cold\report.json') 3 80 40 -ExtraKind
    [ordered]@{ d3d12 = [ordered]@{} } | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $baseline 'runs\02-warm\report.json') -Encoding utf8

    $json = (& $tool -Baseline $baseline -Candidate $candidate | Out-String) | ConvertFrom-Json
    if ($json.schema -ne 'xcl-perf-comparison' -or $json.version -ne 1) { throw 'wrong output schema' }
    if ($json.runs[0].duration_ms.delta -ne -150) { throw 'cold duration delta is wrong' }
    if ($json.runs[1].duration_ms.delta -ne 0) { throw 'zero durations were not compared as raw values' }
    if ($json.runs[1].exit_code.candidate -ne $null -or $json.runs[1].timed_out.candidate -ne $true) { throw 'missing exit/timeout was not preserved' }
    $dxc = $json.runs[0].stalls | Where-Object { $_.kind -eq 'dxc_compile' }
    if ($dxc.delta.total_ns -ne -20 -or $dxc.delta.count -ne 1 -or $dxc.delta.max_ns -ne -30) { throw 'stall deltas are wrong' }
    $gpu = $json.runs[0].stalls | Where-Object { $_.kind -eq 'gpu_wait' }
    if ($null -ne $gpu.baseline -or $gpu.candidate.total_ns -ne 9 -or $null -ne $gpu.delta.total_ns) { throw 'one-sided stall data was not labelled missing' }
    if ($json.runs[1].stalls_status.baseline -ne 'missing_stalls' -or $json.runs[1].stalls_status.candidate -ne 'missing_report') { throw 'missing report/stalls states are wrong' }
    if (($json | ConvertTo-Json -Depth 12) -match 'percent|ratio|significant') { throw 'output invented a derived significance metric' }

    $candidateManifest = Get-Content -LiteralPath (Join-Path $candidate 'manifest.json') -Raw -Encoding utf8 | ConvertFrom-Json
    $candidateManifest.workload_id = 'other-workload'
    $candidateManifest | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath (Join-Path $candidate 'manifest.json') -Encoding utf8
    $refused = $false
    try { $null = & $tool -Baseline $baseline -Candidate $candidate } catch { $refused = $_.Exception.Message -match 'workload_id differs' }
    if (-not $refused) { throw 'different workloads were not refused' }

    Write-Fixture $candidate 'candidate-commit' 850 0
    $candidateManifest = Get-Content -LiteralPath (Join-Path $candidate 'manifest.json') -Raw -Encoding utf8 | ConvertFrom-Json
    $candidateManifest.runs[1].cache_state = 'xcl-cold'
    $candidateManifest | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath (Join-Path $candidate 'manifest.json') -Encoding utf8
    $refused = $false
    try { $null = & $tool -Baseline $baseline -Candidate $candidate } catch { $refused = $_.Exception.Message -match 'does not match run_order' }
    if (-not $refused) { throw 'mismatched run labels were not refused' }

    Write-Fixture $candidate 'candidate-commit' 850 0
    $candidateManifest = Get-Content -LiteralPath (Join-Path $candidate 'manifest.json') -Raw -Encoding utf8 | ConvertFrom-Json
    $candidateManifest.version = 2
    $candidateManifest | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath (Join-Path $candidate 'manifest.json') -Encoding utf8
    $refused = $false
    try { $null = & $tool -Baseline $baseline -Candidate $candidate } catch { $refused = $_.Exception.Message -match 'unsupported manifest schema/version' }
    if (-not $refused) { throw 'an unsupported manifest schema was not refused' }

    Write-Fixture $candidate 'candidate-commit' 850 0
    $candidateManifest = Get-Content -LiteralPath (Join-Path $candidate 'manifest.json') -Raw -Encoding utf8 | ConvertFrom-Json
    $candidateManifest | Add-Member -NotePropertyName system -NotePropertyValue ([pscustomobject]@{ cpu = 'only candidate recorded this' })
    $candidateManifest | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath (Join-Path $candidate 'manifest.json') -Encoding utf8
    $refused = $false
    try { $null = & $tool -Baseline $baseline -Candidate $candidate } catch { $refused = $_.Exception.Message -match 'system information differs' }
    if (-not $refused) { throw 'system information present on only one side was not refused' }

    Write-Fixture $baseline 'base-commit' 1000 0
    Write-Fixture $candidate 'candidate-commit' 850 0
    $baselineManifest = Get-Content -LiteralPath (Join-Path $baseline 'manifest.json') -Raw -Encoding utf8 | ConvertFrom-Json
    $candidateManifest = Get-Content -LiteralPath (Join-Path $candidate 'manifest.json') -Raw -Encoding utf8 | ConvertFrom-Json
    $baselineManifest.runs[0] | Add-Member -NotePropertyName gpu -NotePropertyValue 'GPU A'
    $candidateManifest.runs[0] | Add-Member -NotePropertyName gpu -NotePropertyValue 'GPU B'
    $baselineManifest | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath (Join-Path $baseline 'manifest.json') -Encoding utf8
    $candidateManifest | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath (Join-Path $candidate 'manifest.json') -Encoding utf8
    $refused = $false
    try { $null = & $tool -Baseline $baseline -Candidate $candidate } catch { $refused = $_.Exception.Message -match 'xcl-cold GPU differs' }
    if (-not $refused) { throw 'different GPUs were not refused' }

    Write-Fixture $baseline 'base-commit' 1000 0
    Write-Fixture $candidate 'candidate-commit' 850 0
    $candidateManifest = Get-Content -LiteralPath (Join-Path $candidate 'manifest.json') -Raw -Encoding utf8 | ConvertFrom-Json
    $candidateManifest.runs[1] | Add-Member -NotePropertyName gpu_driver -NotePropertyValue 'only candidate recorded this'
    $candidateManifest | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath (Join-Path $candidate 'manifest.json') -Encoding utf8
    $refused = $false
    try { $null = & $tool -Baseline $baseline -Candidate $candidate } catch { $refused = $_.Exception.Message -match 'xcl-warm GPU driver differs' }
    if (-not $refused) { throw 'GPU driver present on only one side was not refused' }

    Write-Host 'perf-compare tests: all assertions passed'
} finally {
    if (Test-Path -LiteralPath $fixture) { Remove-Item -LiteralPath $fixture -Recurse -Force }
}
