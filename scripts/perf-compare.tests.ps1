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
        workload_id_algorithm = 'sha256-relative-path-length-mtime-utc-ticks'
        workload_id_version = 1
        run_order = @('xcl-cold', 'xcl-warm')
        environment = [ordered]@{ XCL_PRESS_AT = '20:A' }
        runs = @(
            [ordered]@{ order = 1; cache_state = 'xcl-cold'; duration_requested_seconds = 60; duration_actual_ms = $ColdMs; exit_code = 0; timed_out = $false },
            [ordered]@{ order = 2; cache_state = 'xcl-warm'; duration_requested_seconds = 60; duration_actual_ms = $WarmMs; exit_code = $null; timed_out = $true }
        )
    }
    $manifest | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath (Join-Path $Root 'manifest.json') -Encoding utf8
}

function Write-Report([string]$Path, [int]$Count, [long]$Total, [long]$Maximum,
        $Preparation = $null, [int]$PreparationVersion = 1, $RankedPreloads = $null,
        $AllocatorSubmits = $null, [switch]$ExtraKind) {
    $aggregates = @(
        [ordered]@{ kind = 'dxc_compile'; count = $Count; total_ns = $Total; max_ns = $Maximum; histogram_log2_ns = @(0, $Count) }
    )
    if ($ExtraKind) {
        $aggregates += [ordered]@{ kind = 'gpu_wait'; count = 1; total_ns = 9; max_ns = 9; histogram_log2_ns = @(0, 0, 0, 1) }
    }
    $report = [ordered]@{ stalls = [ordered]@{ version = 1; aggregates = $aggregates; slow_events = @() } }
    if ($null -ne $Preparation) {
        $report['preparation'] = [ordered]@{
            version = $PreparationVersion
            compute_build_coalescing = $Preparation
        }
        if ($null -ne $RankedPreloads) {
            $report['preparation']['ranked_preloads'] = $RankedPreloads
        }
        if ($null -ne $AllocatorSubmits) {
            $report['preparation']['allocator_submits'] = $AllocatorSubmits
        }
    }
    $report |
        ConvertTo-Json -Depth 6 | Set-Content -LiteralPath $Path -Encoding utf8
}

try {
    Write-Fixture $baseline 'base-commit' 1000 0
    Write-Fixture $candidate 'candidate-commit' 850 0
    $beforePreparation = [ordered]@{ keyed_submissions = 12; coalescing_hits = 5; unique_keyed_builds = 7; current_in_flight = 0; peak_in_flight = 3 }
    $afterPreparation = [ordered]@{ keyed_submissions = 10; coalescing_hits = 6; unique_keyed_builds = 4; current_in_flight = 1; peak_in_flight = 5 }
    $beforeRanked = [ordered]@{ submissions = 20; started = 15; promoted = 3 }
    $afterRanked = [ordered]@{ submissions = 24; started = 20; promoted = 2 }
    $beforeAllocator = [ordered]@{ outer_execute_calls = 10; chunks = 13; multi_chunk_calls = 2; forced_wait_boundaries = 3; lists_marked = 40; backing_mark_writes = 20; signal_attempts = 13; signal_failures = 1; state_lock_wait_ns = 100; state_lock_hold_ns = 500 }
    $afterAllocator = [ordered]@{ outer_execute_calls = 12; chunks = 14; multi_chunk_calls = 1; forced_wait_boundaries = 2; lists_marked = 44; backing_mark_writes = 22; signal_attempts = 14; signal_failures = 0; state_lock_wait_ns = 80; state_lock_hold_ns = 450 }
    Write-Report (Join-Path $baseline 'runs\01-cold\report.json') 2 100 70 $beforePreparation 3 $beforeRanked $beforeAllocator
    Write-Report (Join-Path $candidate 'runs\01-cold\report.json') 3 80 40 $afterPreparation 3 $afterRanked $afterAllocator -ExtraKind
    [ordered]@{ d3d12 = [ordered]@{} } | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $baseline 'runs\02-warm\report.json') -Encoding utf8

    $json = (& $tool -Baseline $baseline -Candidate $candidate | Out-String) | ConvertFrom-Json
    if ($json.schema -ne 'xcl-perf-comparison' -or $json.version -ne 1) { throw 'wrong output schema' }
    if ($json.workload_id_algorithm -ne 'sha256-relative-path-length-mtime-utc-ticks' -or
        $json.workload_id_version -ne 1) { throw 'workload identity contract is missing from output' }
    if ($json.runs[0].duration_ms.delta -ne -150) { throw 'cold duration delta is wrong' }
    if ($json.runs[1].duration_ms.delta -ne 0) { throw 'zero durations were not compared as raw values' }
    if ($json.runs[1].exit_code.candidate -ne $null -or $json.runs[1].timed_out.candidate -ne $true) { throw 'missing exit/timeout was not preserved' }
    $dxc = $json.runs[0].stalls | Where-Object { $_.kind -eq 'dxc_compile' }
    if ($dxc.delta.total_ns -ne -20 -or $dxc.delta.count -ne 1 -or $dxc.delta.max_ns -ne -30) { throw 'stall deltas are wrong' }
    $gpu = $json.runs[0].stalls | Where-Object { $_.kind -eq 'gpu_wait' }
    if ($null -ne $gpu.baseline -or $gpu.candidate.total_ns -ne 9 -or $null -ne $gpu.delta.total_ns) { throw 'one-sided stall data was not labelled missing' }
    if ($json.runs[1].stalls_status.baseline -ne 'missing_stalls' -or $json.runs[1].stalls_status.candidate -ne 'missing_report') { throw 'missing report/stalls states are wrong' }
    $preparation = $json.runs[0].compute_build_coalescing
    if ($preparation.baseline.keyed_submissions -ne 12 -or $preparation.candidate.keyed_submissions -ne 10 -or
        $preparation.baseline.coalescing_hits -ne 5 -or $preparation.candidate.coalescing_hits -ne 6 -or
        $preparation.baseline.unique_keyed_builds -ne 7 -or $preparation.candidate.unique_keyed_builds -ne 4 -or
        $preparation.baseline.current_in_flight -ne 0 -or $preparation.candidate.current_in_flight -ne 1 -or
        $preparation.baseline.peak_in_flight -ne 3 -or $preparation.candidate.peak_in_flight -ne 5 -or
        $preparation.delta.keyed_submissions -ne -2 -or $preparation.delta.coalescing_hits -ne 1 -or
        $preparation.delta.unique_keyed_builds -ne -3 -or $preparation.delta.current_in_flight -ne 1 -or
        $preparation.delta.peak_in_flight -ne 2) { throw 'preparation counters or deltas are wrong' }
    $ranked = $json.runs[0].ranked_preloads
    if ($json.runs[0].ranked_preloads_status.baseline -ne 'present' -or
        $json.runs[0].ranked_preloads_status.candidate -ne 'present' -or
        $ranked.baseline.submissions -ne 20 -or $ranked.candidate.submissions -ne 24 -or
        $ranked.baseline.started -ne 15 -or $ranked.candidate.started -ne 20 -or
        $ranked.baseline.promoted -ne 3 -or $ranked.candidate.promoted -ne 2 -or
        $ranked.delta.submissions -ne 4 -or $ranked.delta.started -ne 5 -or
        $ranked.delta.promoted -ne -1) { throw 'ranked preload counters, status, or deltas are wrong' }
    $allocator = $json.runs[0].allocator_submits
    if ($json.runs[0].allocator_submits_status.baseline -ne 'present' -or
        $json.runs[0].allocator_submits_status.candidate -ne 'present' -or
        $allocator.baseline.outer_execute_calls -ne 10 -or $allocator.candidate.outer_execute_calls -ne 12 -or
        $allocator.delta.chunks -ne 1 -or $allocator.delta.multi_chunk_calls -ne -1 -or
        $allocator.delta.lists_marked -ne 4 -or $allocator.delta.backing_mark_writes -ne 2 -or
        $allocator.delta.state_lock_hold_ns -ne -50) {
        throw 'allocator submit counters, status, or deltas are wrong'
    }
    if ($json.runs[1].preparation_status.baseline -ne 'missing_preparation' -or
        $json.runs[1].preparation_status.candidate -ne 'missing_report' -or
        $null -ne $json.runs[1].compute_build_coalescing.delta.keyed_submissions -or
        $json.runs[1].ranked_preloads_status.baseline -ne 'missing_preparation' -or
        $json.runs[1].ranked_preloads_status.candidate -ne 'missing_report' -or
        $null -ne $json.runs[1].ranked_preloads.delta.submissions -or
        $json.runs[1].allocator_submits_status.baseline -ne 'missing_preparation' -or
        $json.runs[1].allocator_submits_status.candidate -ne 'missing_report' -or
        $null -ne $json.runs[1].allocator_submits.delta.chunks) {
        throw 'missing preparation data was treated as zero'
    }
    if (($json | ConvertTo-Json -Depth 12) -match '"(percent|percentage|ratio|significance|significant)"\s*:') {
        throw 'output invented a derived significance metric'
    }

    Write-Report (Join-Path $baseline 'runs\01-cold\report.json') 2 100 70 $beforePreparation
    Write-Report (Join-Path $candidate 'runs\01-cold\report.json') 3 80 40 $afterPreparation -ExtraKind
    $versionOne = (& $tool -Baseline $baseline -Candidate $candidate | Out-String) | ConvertFrom-Json
    if ($versionOne.runs[0].preparation_status.baseline -ne 'present' -or
        $versionOne.runs[0].compute_build_coalescing.candidate.coalescing_hits -ne 6 -or
        $versionOne.runs[0].ranked_preloads_status.baseline -ne 'not_in_version' -or
        $versionOne.runs[0].ranked_preloads_status.candidate -ne 'not_in_version' -or
        $versionOne.runs[0].allocator_submits_status.baseline -ne 'not_in_version' -or
        $versionOne.runs[0].allocator_submits_status.candidate -ne 'not_in_version' -or
        $null -ne $versionOne.runs[0].ranked_preloads.delta.started) {
        throw 'version 1 preparation support regressed'
    }
    Write-Report (Join-Path $baseline 'runs\01-cold\report.json') 2 100 70 $beforePreparation 2 $beforeRanked
    Write-Report (Join-Path $candidate 'runs\01-cold\report.json') 3 80 40 $afterPreparation 2 $afterRanked -ExtraKind

    Write-Report (Join-Path $candidate 'runs\02-warm\report.json') 1 1 1 $afterPreparation 4
    $unsupported = (& $tool -Baseline $baseline -Candidate $candidate | Out-String) | ConvertFrom-Json
    if ($unsupported.runs[0].allocator_submits_status.baseline -ne 'not_in_version' -or
        $unsupported.runs[0].allocator_submits_status.candidate -ne 'not_in_version' -or
        $unsupported.runs[1].preparation_status.candidate -ne 'unsupported_version' -or
        $unsupported.runs[1].ranked_preloads_status.candidate -ne 'unsupported_version' -or
        $unsupported.runs[1].allocator_submits_status.candidate -ne 'unsupported_version' -or
        $null -ne $unsupported.runs[1].compute_build_coalescing.candidate -or
        $null -ne $unsupported.runs[1].ranked_preloads.candidate -or
        $null -ne $unsupported.runs[1].allocator_submits.candidate) {
        throw 'unsupported preparation version was not preserved as a state'
    }

    $badReportPath = Join-Path $candidate 'runs\01-cold\report.json'
    Write-Report $badReportPath 3 80 40 $afterPreparation 2 -ExtraKind
    $refused = $false
    try { $null = & $tool -Baseline $baseline -Candidate $candidate } catch { $refused = $_.Exception.Message -match 'malformed ranked preload' }
    if (-not $refused) { throw 'missing version 2 ranked preload counters were not refused' }

    Write-Report $badReportPath 3 80 40 $afterPreparation 2 $afterRanked -ExtraKind
    $badReport = Get-Content -LiteralPath $badReportPath -Raw -Encoding utf8 | ConvertFrom-Json
    $badReport.preparation.ranked_preloads.promoted = -1
    $badReport | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath $badReportPath -Encoding utf8
    $refused = $false
    try { $null = & $tool -Baseline $baseline -Candidate $candidate } catch { $refused = $_.Exception.Message -match 'malformed ranked preloads' }
    if (-not $refused) { throw 'a malformed ranked preload counter was not refused' }

    Write-Report $badReportPath 3 80 40 $afterPreparation 2 $afterRanked -ExtraKind
    $badReport = Get-Content -LiteralPath $badReportPath -Raw -Encoding utf8 | ConvertFrom-Json
    $badReport.preparation.compute_build_coalescing.coalescing_hits = -1
    $badReport | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath $badReportPath -Encoding utf8
    $refused = $false
    try { $null = & $tool -Baseline $baseline -Candidate $candidate } catch { $refused = $_.Exception.Message -match 'malformed preparation' }
    if (-not $refused) { throw 'a malformed preparation counter was not refused' }
    Write-Report $badReportPath 3 80 40 $afterPreparation 3 $afterRanked -ExtraKind
    $refused = $false
    try { $null = & $tool -Baseline $baseline -Candidate $candidate } catch { $refused = $_.Exception.Message -match 'malformed allocator submit' }
    if (-not $refused) { throw 'missing version 3 allocator submit counters were not refused' }
    Write-Report $badReportPath 3 80 40 $afterPreparation 2 $afterRanked -ExtraKind

    $candidateManifest = Get-Content -LiteralPath (Join-Path $candidate 'manifest.json') -Raw -Encoding utf8 | ConvertFrom-Json
    $candidateManifest.workload_id = 'other-workload'
    $candidateManifest | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath (Join-Path $candidate 'manifest.json') -Encoding utf8
    $refused = $false
    try { $null = & $tool -Baseline $baseline -Candidate $candidate } catch { $refused = $_.Exception.Message -match 'workload_id differs' }
    if (-not $refused) { throw 'different workloads were not refused' }

    Write-Fixture $candidate 'candidate-commit' 850 0
    $candidateManifest = Get-Content -LiteralPath (Join-Path $candidate 'manifest.json') -Raw -Encoding utf8 | ConvertFrom-Json
    $candidateManifest.workload_id_algorithm = 'other-algorithm'
    $candidateManifest | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath (Join-Path $candidate 'manifest.json') -Encoding utf8
    $refused = $false
    try { $null = & $tool -Baseline $baseline -Candidate $candidate } catch { $refused = $_.Exception.Message -match 'workload identity algorithm differs' }
    if (-not $refused) { throw 'different workload identity algorithms were not refused' }

    Write-Fixture $candidate 'candidate-commit' 850 0
    $candidateManifest = Get-Content -LiteralPath (Join-Path $candidate 'manifest.json') -Raw -Encoding utf8 | ConvertFrom-Json
    $candidateManifest.workload_id_version = 2
    $candidateManifest | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath (Join-Path $candidate 'manifest.json') -Encoding utf8
    $refused = $false
    try { $null = & $tool -Baseline $baseline -Candidate $candidate } catch { $refused = $_.Exception.Message -match 'workload identity version differs' }
    if (-not $refused) { throw 'different workload identity versions were not refused' }

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
