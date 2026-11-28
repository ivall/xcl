# Run with powershell.exe or pwsh.exe. smoke.ps1 started with `-File` and no -Xcl looks for
# target\release\xcl.exe next to its own folder. Windows PowerShell 5.1 leaves $PSScriptRoot empty in
# the parameter defaults of a script with [Parameter()] attributes when it is started with -File, so a
# default there made Join-Path fail before the script could say anything.
$ErrorActionPreference = 'Stop'
$fixture = Join-Path ([IO.Path]::GetTempPath()) ('xcl-smoke-default-' + [guid]::NewGuid())
$scripts = Join-Path $fixture 'scripts'
$null = New-Item -ItemType Directory -Path $scripts
try {
    # a copy outside the repository: no xcl.exe is there, so the script stops before it runs anything
    Copy-Item -LiteralPath (Join-Path $PSScriptRoot 'smoke.ps1') -Destination $scripts
    $shell = (Get-Process -Id $PID).Path
    $ErrorActionPreference = 'Continue'
    $output = & $shell -NoProfile -NonInteractive -ExecutionPolicy Bypass -File (Join-Path $scripts 'smoke.ps1') -Game $fixture 2>&1 | Out-String
    $ErrorActionPreference = 'Stop'
    # the child's error text can be wrapped at its console width; PowerShell 7 also colours it (ANSI
    # escapes) and starts each wrapped line with `|`
    $flat = $output -replace '\x1b\[[0-?]*[ -/]*[@-~]', '' -replace '[\s|]', ''
    $expected = ('no ' + (Join-Path $scripts '..\target\release\xcl.exe') + ' - build it first') -replace '[\s|]', ''
    if (!$flat.Contains($expected)) {
        throw "smoke.ps1 without -Xcl did not look for xcl.exe next to its folder; its output:`n$output"
    }
} finally {
    $ErrorActionPreference = 'Stop'
    if (Test-Path -LiteralPath $fixture) { Remove-Item -LiteralPath $fixture -Recurse -Force }
}
