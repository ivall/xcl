[CmdletBinding()]
param(
    [string]$Game,
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$ExtraArgs
)

$ErrorActionPreference = 'Stop'

$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path

# Check for .env beside the script first, then in the repo root
$EnvFile = if (Test-Path (Join-Path $PSScriptRoot ".env")) {
    Join-Path $PSScriptRoot ".env"
} elseif (Test-Path (Join-Path $RepoRoot ".env")) {
    Join-Path $RepoRoot ".env"
} else {
    $null
}

if ($EnvFile) {
    Get-Content $EnvFile | ForEach-Object {
        $line = $_.Trim()
        if ($line -and -not $line.StartsWith("#") -and $line.Contains("=")) {
            $key, $val = $line.Split("=", 2)
            $key = $key.Trim()
            $val = $val.Trim()
            [Environment]::SetEnvironmentVariable($key, $val, 'Process')
        }
    }
}

$GamePath = if ($Game) { $Game } elseif ($env:GAME_PATH) { $env:GAME_PATH } else {
    Write-Host "Error: No game directory specified!" -ForegroundColor Red
    Write-Host "Please create a .env file (copy from .env.example) and set GAME_PATH=<path to game>." -ForegroundColor Yellow
    exit 1
}

$XclExe = Join-Path $RepoRoot "target\release\xcl.exe"

if (-not (Test-Path $XclExe)) {
    Write-Host "Building release xcl.exe..." -ForegroundColor Yellow
    Push-Location $RepoRoot
    cargo build --release -p xcl
    Pop-Location
}

Write-Host "XCL Game Launcher" -ForegroundColor Cyan
Write-Host "Game: $GamePath" -ForegroundColor White
if ($env:XCL_DXC_DIR) {
    Write-Host "DXC:  $env:XCL_DXC_DIR" -ForegroundColor Green
}
if ($env:XCL_DXC_THREADS) {
    Write-Host "Threads: $env:XCL_DXC_THREADS | Memory Reserve: $($env:XCL_MEMORY_RESERVE_GB) GB" -ForegroundColor Gray
}

& $XclExe run $GamePath $ExtraArgs

