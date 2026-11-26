# Cold NVIDIA (or any) driver compile benchmark of compute pipelines, for #2088 workstream 6.
#
#   .\scripts\cold-pso-bench.ps1 -Shaders <shaders.rpf or a folder of records> [-Threads 4,8,12]
#                                [-Limit N] [-Xcl .\target\release\xcl.exe] [-Out <folder>] [-ClearNvCache]
#
# One `xcl shaders --precompile --only cs` pass fills a DXIL cache first (DXC, not measured). Then
# for every thread count `--pipelines` makes every compute pipeline of the same input again, with
# XCL's driver blobs off (XCL_PSO_DISK_CACHE=0) and the implicit D3D and driver shader caches off
# (XCL_PIPELINE_CACHE_OFF=1, needs Windows Developer Mode). -ClearNvCache instead empties
# %LOCALAPPDATA%\NVIDIA\DXCache before each run (it also drops other applications' entries).
# Every measured run starts a copy of -Xcl with a name of its own, next to it (the copy is removed
# after the run): the AMD driver keys its shader cache on the exe's file name, so a second run of
# the same name is served warm whatever was cleared (measured on RDNA4, #2129).
# The 8 MiB cap of --pipelines is off (XCL_PIPELINE_MAX_KB=0): a capped program writes no timing
# line, so the summary would leave out the biggest ones without saying so.
# Each build's driver time comes from XCL_PIPELINE_TIMES. Prints a table per thread count: wall
# time, CPU time, peak private commit, pipelines made, summed driver time, p50 / p95 / max per
# pipeline, and the same per DXIL size bucket. Warm runs are not cold numbers: do not mix them.
param(
    [Parameter(Mandatory = $true)][string]$Shaders,
    # strings: `powershell -File ... -Threads 4,8` passes one string "4,8", which [int[]] would
    # read as the number 48
    [string[]]$Threads = @(4, 6, 8, 10, 12, 16, 20, 24),
    [int]$Limit = 0,
    [string]$Xcl = (Join-Path $PSScriptRoot '..\target\release\xcl.exe'),
    [string]$Out = (Join-Path ([IO.Path]::GetTempPath()) ('xcl-cold-pso-' + (Get-Date -Format 'yyyyMMdd-HHmmss'))),
    [switch]$ClearNvCache,
    [int]$TimeoutMinutes = 120
)
$ErrorActionPreference = 'Stop'
[Threading.Thread]::CurrentThread.CurrentCulture = [Globalization.CultureInfo]::InvariantCulture
$Threads = @($Threads | ForEach-Object { $_ -split ',' } | ForEach-Object { $_.Trim() } | Where-Object { $_ })
foreach ($t in $Threads) { if ($t -notmatch '^[1-9]\d{0,3}$') { throw "-Threads: '$t' is not a thread count (give them as 4,8,12)" } }
if (-not (Test-Path $Xcl)) { throw "no $Xcl - build it first (cargo build --release -p xcl)" }
$null = New-Item -ItemType Directory -Force -Path $Out
$dxil = Join-Path $Out 'dxil'

# The value at fraction q of the sorted numbers (nearest rank).
function Get-Rank([double[]]$Sorted, [double]$Q) {
    if ($Sorted.Count -eq 0) { return 0 }
    $Sorted[[math]::Min($Sorted.Count - 1, [math]::Max(0, [math]::Ceiling($Q * $Sorted.Count) - 1))]
}

# Rows of an XCL_PIPELINE_TIMES file: hash, bytes, outcome, wait ms, driver ms.
function Read-Times([string]$Path) {
    if (-not (Test-Path $Path)) { return @() }
    Get-Content $Path | Where-Object { $_ } | ForEach-Object {
        $f = $_.Split(',')
        [pscustomobject]@{ Hash = $f[0]; Bytes = [long]$f[1]; Outcome = $f[2]; WaitMs = [double]$f[3]; DriverMs = [double]$f[4] }
    }
}

$Buckets = @(@('< 64 KiB', 0, 64KB), @('64-256 KiB', 64KB, 256KB), @('256 KiB-1 MiB', 256KB, 1MB), @('1-4 MiB', 1MB, 4MB), @('>= 4 MiB', 4MB, [long]::MaxValue))

# Summary of the driver builds of one run (the `made` rows: cold compiles).
function Get-Summary($Rows) {
    $made = @($Rows | Where-Object Outcome -eq 'made')
    $ms = [double[]]@($made | ForEach-Object DriverMs | Sort-Object)
    $s = [ordered]@{
        made = $made.Count
        refused = @($Rows | Where-Object Outcome -eq 'refused').Count
        reused = @($Rows | Where-Object Outcome -eq 'reused').Count
        driver_s = [math]::Round((($ms | Measure-Object -Sum).Sum) / 1000, 1)
        p50_s = [math]::Round((Get-Rank $ms 0.5) / 1000, 2)
        p95_s = [math]::Round((Get-Rank $ms 0.95) / 1000, 2)
        max_s = [math]::Round((Get-Rank $ms 1.0) / 1000, 2)
        buckets = @()
    }
    foreach ($b in $Buckets) {
        $in = [double[]]@($made | Where-Object { $_.Bytes -ge $b[1] -and $_.Bytes -lt $b[2] } | ForEach-Object DriverMs | Sort-Object)
        if ($in.Count -eq 0) { continue }
        $s.buckets += [pscustomobject]@{ bucket = $b[0]; made = $in.Count; driver_s = [math]::Round((($in | Measure-Object -Sum).Sum) / 1000, 1); p50_s = [math]::Round((Get-Rank $in 0.5) / 1000, 2); p95_s = [math]::Round((Get-Rank $in 0.95) / 1000, 2) }
    }
    [pscustomobject]$s
}

# Windows retains PeakPagefileUsage (peak private commit) on the process handle after exit.
if (-not ('XclColdPso.MemoryCounters' -as [type])) {
    Add-Type -TypeDefinition @'
using System;
using System.ComponentModel;
using System.Runtime.InteropServices;
using Microsoft.Win32.SafeHandles;

namespace XclColdPso {
    public static class MemoryCounters {
        [StructLayout(LayoutKind.Sequential)]
        private struct Counters {
            public uint Size, PageFaultCount;
            public UIntPtr PeakWorkingSet, WorkingSet, PeakPagedPool, PagedPool;
            public UIntPtr PeakNonPagedPool, NonPagedPool, PagefileUsage, PeakPagefileUsage;
        }

        [DllImport("kernel32.dll", SetLastError = true)]
        private static extern bool K32GetProcessMemoryInfo(SafeProcessHandle process, ref Counters counters, uint size);

        public static ulong PeakCommit(SafeProcessHandle process) {
            var counters = new Counters();
            counters.Size = (uint)Marshal.SizeOf(typeof(Counters));
            if (!K32GetProcessMemoryInfo(process, ref counters, counters.Size))
                throw new Win32Exception(Marshal.GetLastWin32Error());
            return counters.PeakPagefileUsage.ToUInt64();
        }
    }
}
'@
}

# One `xcl shaders` run with the given environment; wall and CPU time, peak private commit.
function Invoke-Xcl([string[]]$Arguments, [hashtable]$Environment, [string]$Exe = $Xcl) {
    $psi = New-Object System.Diagnostics.ProcessStartInfo
    $psi.FileName = $Exe
    $psi.Arguments = ($Arguments | ForEach-Object { if ($_ -match '\s') { '"' + $_ + '"' } else { $_ } }) -join ' '
    $psi.UseShellExecute = $false
    $psi.RedirectStandardOutput = $true
    $psi.RedirectStandardError = $true
    foreach ($k in $Environment.Keys) { $psi.EnvironmentVariables[$k] = [string]$Environment[$k] }
    $watch = [Diagnostics.Stopwatch]::StartNew()
    $p = [System.Diagnostics.Process]::Start($psi)
    try {
        $stdout = $p.StandardOutput.ReadToEndAsync(); $stderr = $p.StandardError.ReadToEndAsync()
        while (-not $p.WaitForExit(1000)) {
            if ($watch.Elapsed.TotalMinutes -gt $TimeoutMinutes) { $p.Kill(); throw "xcl ran over $TimeoutMinutes minutes" }
        }
        $p.WaitForExit()
        $wall = $p.ExitTime.ToUniversalTime() - $p.StartTime.ToUniversalTime()
        $peak = [XclColdPso.MemoryCounters]::PeakCommit($p.SafeHandle)
        [pscustomobject]@{ Exit = $p.ExitCode; WallS = [math]::Round($wall.TotalSeconds, 1); CpuS = [math]::Round($p.TotalProcessorTime.TotalSeconds, 1); PeakMB = [math]::Round($peak / 1MB); Out = $stdout.Result + $stderr.Result }
    } finally {
        $p.Dispose()
    }
}

$common = @('shaders', $Shaders, '--precompile', '--only', 'cs')
if ($Limit -gt 0) { $common += @('--limit', "$Limit") }

Write-Host "DXIL pass (not measured) into $dxil"
$prep = Invoke-Xcl $common @{ XCL_DXIL_CACHE = $dxil }
if ($prep.Exit -ne 0) { throw "the DXIL pass failed ($($prep.Exit)):`n$($prep.Out)" }

$results = @()
foreach ($n in $Threads) {
    $times = Join-Path $Out "times-$n.csv"
    Remove-Item $times -ErrorAction SilentlyContinue
    $vars = @{ XCL_DXIL_CACHE = $dxil; XCL_PSO_DISK_CACHE = '0'; XCL_PIPELINE_MAX_KB = '0'; XCL_PIPELINE_THREADS = "$n"; XCL_PIPELINE_TIMES = $times }
    if ($ClearNvCache) {
        $nv = Join-Path $env:LOCALAPPDATA 'NVIDIA\DXCache'
        if (Test-Path $nv) { Get-ChildItem $nv -Force | Remove-Item -Recurse -Force -ErrorAction SilentlyContinue }
        # set, not left out: the run would otherwise inherit the caller's XCL_PIPELINE_CACHE_OFF=1
        $vars.XCL_PIPELINE_CACHE_OFF = '0'
    } else {
        $vars.XCL_PIPELINE_CACHE_OFF = '1'
    }
    Write-Host "threads $n ..."
    # next to -Xcl, so that it finds the same DXC and Agility SDK folders
    $exe = Join-Path (Split-Path $Xcl -Parent) ('{0}-cold-{1}-{2}{3}' -f [IO.Path]::GetFileNameWithoutExtension($Xcl), $n, ([guid]::NewGuid().ToString('N').Substring(0, 8)), [IO.Path]::GetExtension($Xcl))
    Copy-Item -LiteralPath $Xcl -Destination $exe
    try { $run = Invoke-Xcl ($common + '--pipelines') $vars $exe } finally { Remove-Item -LiteralPath $exe -Force -ErrorAction SilentlyContinue }
    if ($run.Exit -ne 0) { throw "threads ${n}: xcl failed ($($run.Exit)):`n$($run.Out)" }
    $s = Get-Summary (Read-Times $times)
    $results += [pscustomobject]@{ threads = $n; wall_s = $run.WallS; cpu_s = $run.CpuS; peak_commit_mb = $run.PeakMB; made = $s.made; refused = $s.refused; reused = $s.reused; driver_s = $s.driver_s; p50_s = $s.p50_s; p95_s = $s.p95_s; max_s = $s.max_s; buckets = $s.buckets }
}

$cache = if ($ClearNvCache) { 'NVIDIA DXCache folder emptied before each run' } else { 'implicit D3D and driver caches disabled (XCL_PIPELINE_CACHE_OFF)' }
"cold compute pipelines: $Shaders$(if ($Limit) { ", first $Limit records" }); $cache; XCL driver blobs off"
'| threads | wall s | CPU s | peak commit MB | made | refused | driver s (sum) | p50 s | p95 s | max s |'
'|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|'
foreach ($r in $results) { "| $($r.threads) | $($r.wall_s) | $($r.cpu_s) | $($r.peak_commit_mb) | $($r.made) | $($r.refused) | $($r.driver_s) | $($r.p50_s) | $($r.p95_s) | $($r.max_s) |" }
''
'| threads | DXIL size | made | driver s (sum) | p50 s | p95 s |'
'|---:|---|---:|---:|---:|---:|'
foreach ($r in $results) { foreach ($b in $r.buckets) { "| $($r.threads) | $($b.bucket) | $($b.made) | $($b.driver_s) | $($b.p50_s) | $($b.p95_s) |" } }
$results | Select-Object threads, wall_s, cpu_s, peak_commit_mb, made, refused, reused, driver_s, p50_s, p95_s, max_s | Export-Csv -NoTypeInformation -Path (Join-Path $Out 'summary.csv')
