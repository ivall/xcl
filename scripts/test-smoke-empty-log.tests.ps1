# Run with powershell.exe or pwsh.exe. An empty log cannot support a smoke report.
$ErrorActionPreference = 'Stop'
$smoke = Join-Path $PSScriptRoot 'smoke.ps1'
$temp = (Resolve-Path -LiteralPath ([IO.Path]::GetTempPath())).Path
$fixture = Join-Path $temp ('xcl-smoke-empty-' + [guid]::NewGuid())
$null = New-Item -ItemType Directory -Path $fixture
try {
    $xcl = Join-Path $fixture 'xcl.cmd'
    $log = Join-Path $fixture 'xcl.log'
    [IO.File]::WriteAllText($xcl, "@echo off`r`necho Log: %~dp0xcl.log`r`n")
    [IO.File]::WriteAllText((Join-Path $fixture 'report.json'), '{"d3d12":{"adapter":"Fixture GPU"}}')

    foreach ($content in @('', " `t`r`n")) {
        [IO.File]::WriteAllText($log, $content)
        $failure = $null
        try { $null = & $smoke -Game $fixture -Xcl $xcl -Seconds 1 -Title 'Fixture' }
        catch { $failure = $_.Exception.Message }
        if ($failure -ne "XCL log is empty: $log") {
            throw "Expected the empty log diagnostic, got '$failure'."
        }
    }
    'smoke-empty-log: ok'
} finally {
    $resolved = (Resolve-Path -LiteralPath $fixture).Path
    if ((Split-Path -Parent $resolved) -eq $temp) { Remove-Item -LiteralPath $resolved -Recurse -Force }
}
