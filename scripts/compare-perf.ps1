param(
    [Parameter(Mandatory = $true)][string]$Baseline,
    [Parameter(Mandatory = $true)][string]$Candidate
)

$ErrorActionPreference = 'Stop'

function Read-PerformanceLog($Path, $Label) {
    $firstFrame = $null
    $samples = [System.Collections.Generic.List[object]]::new()
    # XCL logs one activity line per interval, followed by its memory sample.
    $timestamp = '^\[\s*(?<seconds>[0-9]+\.[0-9]+)\]'
    $file = Convert-Path -LiteralPath $Path
    $share = [IO.FileShare]::ReadWrite -bor [IO.FileShare]::Delete
    $stream = [IO.File]::Open($file, [IO.FileMode]::Open, [IO.FileAccess]::Read, $share)
    $reader = [IO.StreamReader]::new($stream, [Text.Encoding]::UTF8, $true)
    try {
        while ($null -ne ($line = $reader.ReadLine())) {
            if ($line -match ($timestamp + '.*?present: first game frame shown \(')) {
                if ($null -eq $firstFrame) { $firstFrame = $Matches.seconds }
            } elseif ($line -match ($timestamp + '.*?activity \(2 s\): (?<presents>\d+) presents,.*?; DirectStorage (?<requests>\d+) requests,')) {
                $samples.Add([pscustomobject]@{
                    End = $Matches.seconds
                    Presents = [long]$Matches.presents
                    Requests = [long]$Matches.requests
                    Commit = $null
                })
            } elseif ($samples.Count -gt 0 -and $line -match 'memory: (?<commit>\d+) MB committed by the process;') {
                $samples[$samples.Count - 1].Commit = [long]$Matches.commit
            }
        }
    } finally {
        $reader.Dispose()
    }
    if ($samples.Count -eq 0) {
        throw "$Label log contains no two-second activity samples. Run XCL with XCL_ACTIVITY=1."
    }
    [pscustomobject]@{ FirstFrame = $firstFrame; Samples = $samples }
}

$base = Read-PerformanceLog $Baseline 'Baseline'
$next = Read-PerformanceLog $Candidate 'Candidate'
$count = [Math]::Min($base.Samples.Count, $next.Samples.Count)
$basePeak = $null
$nextPeak = $null
$rows = for ($index = 0; $index -lt $count; $index++) {
    $before = $base.Samples[$index]
    $after = $next.Samples[$index]
    if ($null -ne $before.Commit -and ($null -eq $basePeak -or $before.Commit -gt $basePeak)) { $basePeak = $before.Commit }
    if ($null -ne $after.Commit -and ($null -eq $nextPeak -or $after.Commit -gt $nextPeak)) { $nextPeak = $after.Commit }
    [pscustomobject]@{
        'Sample' = $index + 1
        'Base s' = $before.End
        'New s' = $after.End
        'Base presents' = $before.Presents
        'New presents' = $after.Presents
        'Base DS' = $before.Requests
        'New DS' = $after.Requests
    }
}

"Compared $count two-second samples from startup (baseline $($base.Samples.Count), candidate $($next.Samples.Count))."
$baseFrame = if ($null -eq $base.FirstFrame) { 'n/a' } else { "$($base.FirstFrame) s" }
$nextFrame = if ($null -eq $next.FirstFrame) { 'n/a' } else { "$($next.FirstFrame) s" }
if ($null -eq $basePeak) { $basePeak = 'n/a' }
if ($null -eq $nextPeak) { $nextPeak = 'n/a' }
"First frame shown: baseline $baseFrame, candidate $nextFrame"
"Peak sampled process commit (MB): baseline $basePeak, candidate $nextPeak"
"DS columns count DirectStorage requests."
$rows | Format-Table -AutoSize
