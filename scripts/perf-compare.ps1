# Compare two version-1 perf-run artifact folders without changing either one. Concise text goes
# to the host; the success stream is one machine-readable JSON document.
param(
    [Parameter(Mandatory = $true)][string]$Baseline,
    [Parameter(Mandatory = $true)][string]$Candidate
)
$ErrorActionPreference = 'Stop'

function Read-Json([string]$Path) {
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) { throw "missing artifact: $Path" }
    try {
        return Get-Content -LiteralPath $Path -Raw -Encoding utf8 | ConvertFrom-Json
    } catch {
        throw "invalid JSON in $($Path): $($_.Exception.Message)"
    }
}

function Has-Property($Object, [string]$Name) {
    return $null -ne $Object -and $null -ne $Object.PSObject.Properties[$Name]
}

function Require-Same-Property($Before, $After, [string]$Name, [string]$Description) {
    $beforeHas = Has-Property $Before $Name
    $afterHas = Has-Property $After $Name
    if ($beforeHas -ne $afterHas) { throw "$Description differs; runs are incomparable" }
    if ($beforeHas) {
        $beforeValue = $Before.$Name | ConvertTo-Json -Compress -Depth 6
        $afterValue = $After.$Name | ConvertTo-Json -Compress -Depth 6
        if ($beforeValue -ne $afterValue) { throw "$Description differs; runs are incomparable" }
    }
}

function Read-Manifest([string]$Root, [string]$Side) {
    $manifest = Read-Json (Join-Path $Root 'manifest.json')
    if ($manifest.schema -ne 'xcl-perf-run' -or [int]$manifest.version -ne 1) {
        throw "$Side has unsupported manifest schema/version"
    }
    if ($manifest.dirty -ne $false -or $manifest.build.profile -ne 'release') {
        throw "$Side is not a clean release run"
    }
    if ([string]::IsNullOrWhiteSpace([string]$manifest.commit) -or -not (Has-Property $manifest 'environment')) {
        throw "$Side manifest is missing commit/environment"
    }
    if ([string]::IsNullOrWhiteSpace([string]$manifest.workload_id)) {
        throw "$Side has no workload_id"
    }
    $labels = @($manifest.run_order)
    if ($labels.Count -ne 2 -or $labels[0] -ne 'xcl-cold' -or $labels[1] -ne 'xcl-warm') {
        throw "$Side run_order must be xcl-cold, xcl-warm"
    }
    $runs = @($manifest.runs)
    if ($runs.Count -ne 2) { throw "$Side must contain exactly two run records" }
    for ($i = 0; $i -lt 2; $i++) {
        if ([int]$runs[$i].order -ne $i + 1 -or $runs[$i].cache_state -ne $labels[$i]) {
            throw "$Side run record $($i + 1) does not match run_order"
        }
        if (-not (Has-Property $runs[$i] 'duration_requested_seconds') -or $null -eq $runs[$i].duration_requested_seconds) {
            throw "$Side run record $($i + 1) has no requested duration"
        }
    }
    return $manifest
}

function Read-Stalls([string]$Root, [int]$Order, [string]$Label) {
    $mode = $Label.Substring(4)
    $report = Join-Path $Root (Join-Path 'runs' (Join-Path ('{0:d2}-{1}' -f $Order, $mode) 'report.json'))
    if (-not (Test-Path -LiteralPath $report -PathType Leaf)) {
        return [ordered]@{ status = 'missing_report'; aggregates = @{} }
    }
    $json = Read-Json $report
    if (-not (Has-Property $json 'stalls') -or $null -eq $json.stalls) {
        return [ordered]@{ status = 'missing_stalls'; aggregates = @{} }
    }
    if ([int]$json.stalls.version -ne 1) { throw "unsupported stall summary in $report" }
    $aggregates = @{}
    foreach ($aggregate in @($json.stalls.aggregates)) {
        $kind = [string]$aggregate.kind
        if ([string]::IsNullOrWhiteSpace($kind) -or $aggregates.ContainsKey($kind)) {
            throw "invalid or duplicate stall kind in $report"
        }
        $aggregates[$kind] = $aggregate
    }
    return [ordered]@{ status = 'present'; aggregates = $aggregates }
}

function Number-Or-Null($Object, [string]$Name) {
    if ($null -eq $Object -or -not (Has-Property $Object $Name) -or $null -eq $Object.$Name) { return $null }
    return [long]$Object.$Name
}

function Delta-Or-Null($Before, $After) {
    if ($null -eq $Before -or $null -eq $After) { return $null }
    return [long]$After - [long]$Before
}

function Bool-Or-Null($Object, [string]$Name) {
    if ($null -eq $Object -or -not (Has-Property $Object $Name) -or $null -eq $Object.$Name) { return $null }
    return [bool]$Object.$Name
}

function Aggregate-Or-Null($Aggregate) {
    if ($null -eq $Aggregate) { return $null }
    return [ordered]@{
        count = Number-Or-Null $Aggregate 'count'
        total_ns = Number-Or-Null $Aggregate 'total_ns'
        max_ns = Number-Or-Null $Aggregate 'max_ns'
        histogram_log2_ns = @($Aggregate.histogram_log2_ns | ForEach-Object { [long]$_ })
    }
}

function Compare-Aggregates($BeforeSet, $AfterSet) {
    $kinds = @($BeforeSet.Keys) + @($AfterSet.Keys) | Sort-Object -Unique
    $out = @()
    foreach ($kind in $kinds) {
        $beforeRaw = if ($BeforeSet.ContainsKey($kind)) { $BeforeSet[$kind] } else { $null }
        $afterRaw = if ($AfterSet.ContainsKey($kind)) { $AfterSet[$kind] } else { $null }
        $before = Aggregate-Or-Null $beforeRaw
        $after = Aggregate-Or-Null $afterRaw
        $histogramDelta = $null
        if ($null -ne $before -and $null -ne $after -and $before.histogram_log2_ns.Count -eq $after.histogram_log2_ns.Count) {
            $histogramDelta = for ($i = 0; $i -lt $before.histogram_log2_ns.Count; $i++) {
                $after.histogram_log2_ns[$i] - $before.histogram_log2_ns[$i]
            }
        }
        $out += [ordered]@{
            kind = $kind
            baseline = $before
            candidate = $after
            delta = [ordered]@{
                count = Delta-Or-Null $before.count $after.count
                total_ns = Delta-Or-Null $before.total_ns $after.total_ns
                max_ns = Delta-Or-Null $before.max_ns $after.max_ns
                histogram_log2_ns = $histogramDelta
            }
        }
    }
    return $out
}

function Display-Value($Value, [string]$Suffix) {
    if ($null -eq $Value) { return 'missing' }
    return "$Value$Suffix"
}

$baselineRoot = [IO.Path]::GetFullPath($Baseline)
$candidateRoot = [IO.Path]::GetFullPath($Candidate)
$beforeManifest = Read-Manifest $baselineRoot 'baseline'
$afterManifest = Read-Manifest $candidateRoot 'candidate'

if ($beforeManifest.workload_id -ne $afterManifest.workload_id) { throw 'workload_id differs; runs are incomparable' }
if ((@($beforeManifest.run_order) -join ',') -ne (@($afterManifest.run_order) -join ',')) { throw 'run order differs; runs are incomparable' }
Require-Same-Property $beforeManifest $afterManifest 'system' 'system information'
if (($beforeManifest.environment | ConvertTo-Json -Compress -Depth 6) -ne ($afterManifest.environment | ConvertTo-Json -Compress -Depth 6)) {
    throw 'runtime environment differs; runs are incomparable'
}

$comparisons = @()
for ($i = 0; $i -lt 2; $i++) {
    $beforeRun = @($beforeManifest.runs)[$i]
    $afterRun = @($afterManifest.runs)[$i]
    Require-Same-Property $beforeRun $afterRun 'gpu' "$($beforeRun.cache_state) GPU"
    Require-Same-Property $beforeRun $afterRun 'gpu_driver' "$($beforeRun.cache_state) GPU driver"
    if ([long]$beforeRun.duration_requested_seconds -ne [long]$afterRun.duration_requested_seconds) {
        throw "$($beforeRun.cache_state) requested duration differs; runs are incomparable"
    }
    $beforeDuration = Number-Or-Null $beforeRun 'duration_actual_ms'
    $afterDuration = Number-Or-Null $afterRun 'duration_actual_ms'
    $beforeStalls = Read-Stalls $baselineRoot ($i + 1) $beforeRun.cache_state
    $afterStalls = Read-Stalls $candidateRoot ($i + 1) $afterRun.cache_state
    $stalls = @(Compare-Aggregates $beforeStalls.aggregates $afterStalls.aggregates)

    $comparison = [ordered]@{
        label = $beforeRun.cache_state
        duration_ms = [ordered]@{
            baseline = $beforeDuration
            candidate = $afterDuration
            delta = Delta-Or-Null $beforeDuration $afterDuration
        }
        exit_code = [ordered]@{
            baseline = Number-Or-Null $beforeRun 'exit_code'
            candidate = Number-Or-Null $afterRun 'exit_code'
        }
        timed_out = [ordered]@{
            baseline = Bool-Or-Null $beforeRun 'timed_out'
            candidate = Bool-Or-Null $afterRun 'timed_out'
        }
        stalls_status = [ordered]@{ baseline = $beforeStalls.status; candidate = $afterStalls.status }
        stalls = $stalls
    }
    $comparisons += $comparison

    Write-Host "$($comparison.label): duration $(Display-Value $beforeDuration ' ms') -> $(Display-Value $afterDuration ' ms'); delta $(Display-Value $comparison.duration_ms.delta ' ms')"
    Write-Host "  exit: $(Display-Value $comparison.exit_code.baseline '') -> $(Display-Value $comparison.exit_code.candidate ''); timeout: $(Display-Value $comparison.timed_out.baseline '') -> $(Display-Value $comparison.timed_out.candidate '')"
    if ($stalls.Count -eq 0) {
        Write-Host "  stalls: missing/empty (baseline $($beforeStalls.status), candidate $($afterStalls.status))"
    } else {
        foreach ($stall in $stalls) {
            Write-Host "  stall $($stall.kind): total $(Display-Value $stall.baseline.total_ns ' ns') -> $(Display-Value $stall.candidate.total_ns ' ns'); count $(Display-Value $stall.baseline.count '') -> $(Display-Value $stall.candidate.count ''); max $(Display-Value $stall.baseline.max_ns ' ns') -> $(Display-Value $stall.candidate.max_ns ' ns')"
        }
    }
}

[ordered]@{
    schema = 'xcl-perf-comparison'
    version = 1
    workload_id = $beforeManifest.workload_id
    run_order = @($beforeManifest.run_order)
    baseline = [ordered]@{ commit = $beforeManifest.commit; path = $baselineRoot }
    candidate = [ordered]@{ commit = $afterManifest.commit; path = $candidateRoot }
    runs = $comparisons
} | ConvertTo-Json -Depth 12
