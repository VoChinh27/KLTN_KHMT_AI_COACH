[CmdletBinding()]
param(
    [string]$RunRoot = '',
    [string]$Manifest = '',
    [string]$ProjectRoot = ''
)
$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
if (-not $ProjectRoot) {
    $candidate = Get-Item -LiteralPath $PSScriptRoot
    while ($candidate -and -not (Test-Path -LiteralPath (Join-Path $candidate.FullName 'src/gym_coach/counting/fixed_origin'))) {
        $candidate = $candidate.Parent
    }
    if (-not $candidate) { throw 'Cannot locate Gym project root' }
    $ProjectRoot = $candidate.FullName
}
if (-not $Manifest) { $Manifest = Join-Path $PSScriptRoot 'manifest.json' }
$python = Join-Path $ProjectRoot '.venv/Scripts/python.exe'
if (-not (Test-Path -LiteralPath $python)) { throw 'Missing project Python' }
if (-not (Test-Path -LiteralPath $Manifest)) { throw 'Missing prepared manifest' }
$frozen = Get-Content -LiteralPath $Manifest -Raw | ConvertFrom-Json
$launcherHash = (Get-FileHash -LiteralPath $PSCommandPath -Algorithm SHA256).Hash.ToLowerInvariant()
if ($launcherHash -ne $frozen.payload.launcher.sha256) { throw 'Launcher differs from frozen manifest' }
$arguments = @('-m', 'gym_coach.counting.fixed_origin.orchestrate', '--root', $ProjectRoot, '--manifest', $Manifest, '--user-run')
if ($RunRoot) { $arguments += @('--run-root', $RunRoot) }
Push-Location -LiteralPath $ProjectRoot
try {
    & $python @arguments
    if ($LASTEXITCODE -ne 0) { throw "Run stopped (exit $LASTEXITCODE). Preserve artifacts; use the printed runRoot to resume." }
} finally {
    Pop-Location
}
