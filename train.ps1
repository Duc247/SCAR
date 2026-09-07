# Arguments are forwarded unchanged, e.g. .\train.ps1 --run-id m3_run01 --skip-cache
$ErrorActionPreference = 'Stop'
Push-Location $PSScriptRoot
try {
    & python run_all.py @args
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
} finally {
    Pop-Location
}
