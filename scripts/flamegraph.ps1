<#
.SYNOPSIS
Creates a Windows CPU flamegraph for one of XCL's Criterion benchmark targets.

.EXAMPLE
.\scripts\flamegraph.ps1 -Bench gpu_cpu

.EXAMPLE
.\scripts\flamegraph.ps1 -Bench gpu_cpu -Filter 'recompiler_translate/linear/256' -Output .\translate.svg
#>
[CmdletBinding()]
param(
    [ValidateSet("core", "formats", "gpu_cpu", "runtime")]
    [string]$Bench = "core",

    [string]$Filter,

    [string]$Output
)

$ErrorActionPreference = "Stop"

if (-not (Get-Command cargo-flamegraph -ErrorAction SilentlyContinue)) {
    throw "cargo-flamegraph is not installed or is not on PATH. Install it with: cargo install flamegraph"
}

$runningOnWindows = [Environment]::OSVersion.Platform -eq [PlatformID]::Win32NT
if ($runningOnWindows) {
    $identity = [Security.Principal.WindowsIdentity]::GetCurrent()
    $principal = [Security.Principal.WindowsPrincipal]::new($identity)
    if (-not $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
        throw "cargo-flamegraph's Windows profiler requires an elevated PowerShell. Open PowerShell as Administrator and run this script again."
    }
}

$repoRoot = Split-Path -Parent $PSScriptRoot
if ([string]::IsNullOrWhiteSpace($Output)) {
    $Output = Join-Path $repoRoot "target\flamegraphs\$Bench.svg"
} elseif (-not [IO.Path]::IsPathRooted($Output)) {
    $Output = Join-Path $repoRoot $Output
}
$Output = [IO.Path]::GetFullPath($Output)

$outputDirectory = Split-Path -Parent $Output
if ($outputDirectory) {
    New-Item -ItemType Directory -Force -Path $outputDirectory | Out-Null
}

$cargoArguments = @(
    "flamegraph",
    "--package", "xcl",
    "--bench", $Bench,
    "--output", $Output,
    "--"
)
if (-not [string]::IsNullOrWhiteSpace($Filter)) {
    $cargoArguments += $Filter
}
$cargoArguments += "--bench"

Push-Location $repoRoot
try {
    & cargo @cargoArguments
    if ($LASTEXITCODE -ne 0) {
        throw "cargo flamegraph exited with code $LASTEXITCODE"
    }
} finally {
    Pop-Location
}

if (-not (Test-Path -LiteralPath $Output -PathType Leaf)) {
    throw "cargo flamegraph completed without creating $Output"
}

Write-Host "Flamegraph written to $Output"
