# Run with powershell.exe or pwsh.exe. No title or executable is needed.
$ErrorActionPreference = 'Stop'
$smoke = Join-Path $PSScriptRoot 'smoke.ps1'
$missingExe = Join-Path ([IO.Path]::GetTempPath()) ([guid]::NewGuid().ToString() + '.exe')

foreach ($duration in @(-121, -1, 2147483)) {
    try {
        & $smoke -Game 'unused' -Seconds $duration -Xcl $missingExe
    } catch {
        if ($_.FullyQualifiedErrorId -like 'ParameterArgumentValidationError,*') { continue }
        throw
    }
    throw "Expected parameter validation to reject -Seconds $duration."
}

foreach ($timeout in @(
    @{ Seconds = 2147364 }
    @{ Seconds = 2147363; StartupTimeoutSeconds = 121 }
)) {
    try {
        & $smoke -Game 'unused' -Xcl $missingExe @timeout
    } catch {
        if ($_.Exception.Message -eq 'The sum of -Seconds and -StartupTimeoutSeconds exceeds the Windows process wait limit.') { continue }
        throw
    }
    throw 'Expected the combined timeout to be rejected before checking the executable.'
}

foreach ($timeout in @(
    @{ Seconds = 0 }
    @{ Seconds = 1 }
    @{ Seconds = 2147363 }
    @{ Seconds = 2147482; StartupTimeoutSeconds = 1 }
    @{ Seconds = 2061083; StartupTimeoutSeconds = 86400 }
)) {
    try {
        & $smoke -Game 'unused' -Xcl $missingExe @timeout
    } catch {
        if ($_.Exception.Message.StartsWith("no $missingExe - build it first")) { continue }
        throw
    }
    throw 'Expected the combined timeout to pass validation and reach the executable check.'
}
