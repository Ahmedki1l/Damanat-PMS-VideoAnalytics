#Requires -Version 5.1
<#
  run_all.ps1 - thin wrapper around the Python supervisor.

  The multi-process launch is now owned by supervisor.py (invoked as
  `python main.py --supervise`). The camera GROUPS table, per-group core
  slicing, PID file, log redirection and teardown all live there - this script
  only resolves the venv python and forwards flags, so there is ONE source of
  truth for the topology (edit $GROUPS in supervisor.py, not here).

  This blocks and supervises in the foreground: the supervisor stays up until
  a worker dies or you press Ctrl+C, at which point it tears every group down.

  USAGE
    .\run_all.ps1               # start + supervise all groups (Ctrl+C stops all)
    .\run_all.ps1 -ResetPlates  # wipe ALL plate identities once, then start
    .\run_all.ps1 -Foreground   # also mirror each group's log to this console
    Get-Content -Wait .\logs\va_b1-gate.out.log   # follow a group's log
    .\stop_all.ps1              # stop groups by PID file (if run detached)
#>
param(
    # One-shot: wipe ALL slot plate identities + per-car gallery folders ONCE,
    # before launching the groups (forwarded to the supervisor's --reset-plates,
    # which runs main.py --reset-plates-only and waits). Global and destructive;
    # VA-local (does not notify PMS-AI). Occupancy rows are left untouched.
    [switch]$ResetPlates,
    # Mirror every group's log file to this console (in addition to logs/*.log).
    [switch]$Foreground
)

$ErrorActionPreference = 'Stop'
$Root = $PSScriptRoot

# Console python (the supervisor spawns the windowless workers itself via
# CREATE_NO_WINDOW, so no pythonw juggling is needed here). Fall back to PATH.
$Python = Join-Path $Root '.venv\Scripts\python.exe'
if (-not (Test-Path $Python)) { $Python = 'python' }

$cliArgs = @('main.py', '--supervise')
if ($ResetPlates) { $cliArgs += '--reset-plates' }
if ($Foreground)  { $cliArgs += '--foreground' }

Push-Location $Root
try {
    & $Python @cliArgs
    exit $LASTEXITCODE
} finally {
    Pop-Location
}
